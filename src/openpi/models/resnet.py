# _resnet.py
from typing import Optional, Tuple
import jax.numpy as jnp
import flax.linen as nn

# ---------- small ResNet-18 backbone ----------
class BasicBlock(nn.Module):
    channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(self.channels, (3,3), self.stride, padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(self.channels, (3,3), 1, padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        if x.shape[-1] != self.channels or self.stride != 1:
            x = nn.Conv(self.channels, (1,1), self.stride, use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x + y)

def _make_layer(ch, blocks, first_stride):
    return [
        BasicBlock(ch, stride=first_stride),
        *[BasicBlock(ch, stride=1) for _ in range(blocks-1)]
    ]

def posemb_sincos_2d(h, w, dim):
    """(H, W, C) -> (H*W, C) 2D sincos PE; dim must be even."""
    assert dim % 4 == 0
    y, x = jnp.meshgrid(jnp.linspace(-1., 1., h), jnp.linspace(-1., 1., w), indexing='ij')
    # split equally across (sinx, cosx, siny, cosy)
    dim_q = dim // 4
    freq = jnp.exp(-jnp.linspace(0., 1., dim_q)) * 2.0 * jnp.pi
    sinx = jnp.sin(x[..., None] * freq)   # (H, W, dim_q)
    cosx = jnp.cos(x[..., None] * freq)
    siny = jnp.sin(y[..., None] * freq)
    cosy = jnp.cos(y[..., None] * freq)
    pe = jnp.concatenate([sinx, cosx, siny, cosy], axis=-1)  # (H, W, dim)
    return pe.reshape(h*w, dim)

class ResNet18Backbone(nn.Module):
    no_stride: bool = False
    remove_layer_num: int = 4  # 0..5 (like your PyTorch arg)

    @nn.compact
    def __call__(self, x, train: bool):
        # stem
        y = nn.Conv(64, (7,7), strides=(1 if self.no_stride else 2), padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        # y = nn.max_pool(y, (3,3), strides=(1 if self.no_stride else 2), padding='SAME')

        y = nn.max_pool(
            y,
            window_shape=(3, 3),
            strides=(1, 1) if self.no_stride else (2, 2),
            padding='SAME',
        )

        # layers (ResNet-18: [2,2,2,2] blocks with channel multipliers [64,128,256,512])
        layers = []
        layers += _make_layer(64,  2, 1 if self.no_stride else 1)
        layers += _make_layer(128, 2, 1 if self.no_stride else 2)
        layers += _make_layer(256, 2, 1 if self.no_stride else 2)
        layers += _make_layer(512, 2, 1 if self.no_stride else 2)

        # Optionally drop top layers (like remove_layer_num in your ref)
        # Keep: stem + first (4 - remove_layer_num) groups
        keep_groups = max(0, 4 - max(0, min(self.remove_layer_num, 5)))
        idx = 0
        for g in range(keep_groups):
            for _ in range(2):
                y = layers[idx](y, train=train)
                idx += 1
        # If remove_layer_num > 4, you effectively keep just the stem (no residual groups).

        return y  # (B, H', W', C')

# ---------- Image tokenizer module, SigLIP-compatible output ----------
class Module(nn.Module):
    """
    Drop-in replacement for _siglip.Module used by Pi0.
    Produces (tokens, aux), where tokens is (B, S, E) and E==out_dim.
    """
    out_dim: int                       # == paligemma_config.width
    remove_layer_num: int = 4          # match your other repo default
    no_stride: bool = False
    language_fusion: str = "none"      # "none" or "film"
    language_dim: int = 768            # for FiLM

    @nn.compact
    def __call__(self, images, train: bool = False, langs: Optional[jnp.ndarray] = None
                 ) -> Tuple[jnp.ndarray, None]:
        # Expect images in float32 [0,1]. If not, normalize here as needed.
        x = images

        # Backbone feature map
        feats = ResNet18Backbone(no_stride=self.no_stride,
                                 remove_layer_num=self.remove_layer_num)(x, train=train)  # (B, H', W', Cb)
        B, H, W, Cb = feats.shape

        # Optional FiLM conditioning (per-stage would be ideal; here we demo a single-site FiLM)
        if (self.language_fusion != "none") and (langs is not None):
            gamma_beta = nn.Dense(2 * Cb)(langs)        # (B, 2*Cb)
            gamma, beta = jnp.split(gamma_beta, 2, axis=-1)
            gamma = gamma[:, None, None, :]
            beta  = beta[:, None, None, :]
            feats = (1.0 + gamma) * feats + beta

        def _adaptive_pool_to(feats, gh, gw):
            B, H, W, C = feats.shape
            sh = max(1, H // gh)
            sw = max(1, W // gw)
            feats = nn.avg_pool(feats, window_shape=(sh, sw), strides=(sh, sw), padding='SAME')
            # Optional: if you want exactly (gh, gw), do one more crop or interp; usually this is close enough.
            return feats

        # choose a grid similar to SigLIP
        target_grid = (14, 14)   # 196 tokens
        feats = _adaptive_pool_to(feats, *target_grid)

        # Project channels to LLM width
        feats = nn.Conv(self.out_dim, (1,1), use_bias=False)(feats)  # (B, H, W, out_dim)

        # IMPORTANT: recompute shape *now*
        B, H2, W2, E = feats.shape
        tokens = feats.reshape(B, H2 * W2, E)  # (B, S, E)

        # Flatten to tokens
        # tokens = feats.reshape(B, H*W, self.out_dim)                 # (B, S, E)

        # Add 2D sin-cos PE like ViT/SigLIP (keeps LLM happy)
        # pe = posemb_sincos_2d(H, W, self.out_dim)                    # (S, E)
        pe = posemb_sincos_2d(H2, W2, E)               # (S, E)
        pe = pe.astype(tokens.dtype)                   # match dtype (bf16/fp16/etc)
        tokens = tokens + pe[None, :, :]

        return tokens, None
