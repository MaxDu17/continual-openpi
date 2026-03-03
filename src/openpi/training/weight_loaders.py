import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable
from pathlib import Path

import flax.traverse_util
from flax import traverse_util # Allow both usages
import numpy as np
import torch
import jax
import jax.numpy as jnp
from transformers import PaliGemmaForConditionalGeneration

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")

@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")




def to_jax(x: torch.Tensor):
    return jax.device_put(jnp.asarray(x.detach().cpu().numpy()))

def flatten_pytree(tree):
    return traverse_util.flatten_dict(tree, sep="/")

def unflatten_pytree(flat):
    return traverse_util.unflatten_dict({tuple(k.split("/")): v for k, v in flat.items()})

def assign(out_flat, key, arr):
    out_flat[key] = arr

def stack_per_layer(objs, axis0_len, fn):
    """fn(i) -> np/torch array; returns stacked array with first dim=axis0_len."""
    xs = [fn(i) for i in range(axis0_len)]
    if isinstance(xs[0], torch.Tensor):
        return torch.stack(xs, dim=0)
    return np.stack(xs, axis=0)

def put_with_shape(out_flat, key, tensor, nnx_flat):
    # convert first
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    tensor = np.asarray(tensor)  # ok for shape ops below

    # match target shape (swap last 2 dims if necessary)
    if key not in nnx_flat:
        return # Skip if key not in target
        
    tgt = nnx_flat[key]
    try:
        target_shape = tuple(tgt.shape)
    except Exception:
        target_shape = tuple(getattr(tgt, "value").shape)
    arr = tensor
    if arr.shape != target_shape:
        if arr.shape[:-2] == target_shape[:-2] and arr.shape[-2:] == target_shape[-1:-3:-1]:
            arr = np.swapaxes(arr, -1, -2)
        else:
            # Try to see if it's just a mismatch that we can ignore or is it critical?
            # For now raise error as in original code
            raise ValueError(f"Shape mismatch for {key}: got {arr.shape}, want {target_shape}")

    # finally store as JAX array
    out_flat[key] = jnp.asarray(arr)

def map_siglip_and_projector(hf_model, nnx_flat):
    out = {}

    # Embedding conv: [out,in,kh,kw] -> [kh,kw,in,out]
    w = hf_model.vision_tower.vision_model.embeddings.patch_embedding.weight  # [1152,3,14,14]
    b = hf_model.vision_tower.vision_model.embeddings.patch_embedding.bias    # [1152]
    assign(out, "PaliGemma/img/embedding/kernel", to_jax(w.permute(2,3,1,0)))  # [14,14,3,1152]
    assign(out, "PaliGemma/img/embedding/bias",   to_jax(b))

    # Positional embedding: [256,1152] -> [1,256,1152]
    pe = hf_model.vision_tower.vision_model.embeddings.position_embedding.weight
    assign(out, "PaliGemma/img/pos_embedding", to_jax(pe.unsqueeze(0)))

    # Encoder (27 layers)
    enc = hf_model.vision_tower.vision_model.encoder.layers
    L = 27
    assert len(enc) == L, f"Expected 27 layers, got {len(enc)}"

    # LayerNorms
    ln1_scale = stack_per_layer(enc, L, lambda i: enc[i].layer_norm1.weight)   # [27,1152]
    ln1_bias  = stack_per_layer(enc, L, lambda i: enc[i].layer_norm1.bias)     # [27,1152]
    ln2_scale = stack_per_layer(enc, L, lambda i: enc[i].layer_norm2.weight)
    ln2_bias  = stack_per_layer(enc, L, lambda i: enc[i].layer_norm2.bias)
    assign(out, "PaliGemma/img/Transformer/encoderblock/LayerNorm_0/scale", to_jax(ln1_scale))
    assign(out, "PaliGemma/img/Transformer/encoderblock/LayerNorm_0/bias",  to_jax(ln1_bias))
    assign(out, "PaliGemma/img/Transformer/encoderblock/LayerNorm_1/scale", to_jax(ln2_scale))
    assign(out, "PaliGemma/img/Transformer/encoderblock/LayerNorm_1/bias",  to_jax(ln2_bias))

    # MLPs
    # fc1: [4304,1152] -> kernel [27,1152,4304], bias [27,4304]
    fc1_w = stack_per_layer(enc, L, lambda i: enc[i].mlp.fc1.weight)  # [27,4304,1152]
    fc1_b = stack_per_layer(enc, L, lambda i: enc[i].mlp.fc1.bias)    # [27,4304]
    assign(out, "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel", to_jax(fc1_w.permute(0,2,1)))
    assign(out, "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias",   to_jax(fc1_b))

    # fc2: [1152,4304] -> kernel [27,4304,1152], bias [27,1152]
    fc2_w = stack_per_layer(enc, L, lambda i: enc[i].mlp.fc2.weight)  # [27,1152,4304]
    fc2_b = stack_per_layer(enc, L, lambda i: enc[i].mlp.fc2.bias)    # [27,1152]
    assign(out, "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel", to_jax(fc2_w.permute(0,2,1)))
    assign(out, "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias",   to_jax(fc2_b))

    # Self-attention: each proj is Linear(1152->1152) with bias
    # NNX wants heads split: 16 heads * 72 = 1152
    def split_heads_weight(W):  # [1152,1152] -> [1152,16,72]
        return W.view(1152, 16, 72)

    def split_heads_bias(b):    # [1152] -> [16,72]
        return b.view(16, 72)

    qW = stack_per_layer(enc, L, lambda i: split_heads_weight(enc[i].self_attn.q_proj.weight.T))  # layer stacks already applied
    kW = stack_per_layer(enc, L, lambda i: split_heads_weight(enc[i].self_attn.k_proj.weight.T))
    vW = stack_per_layer(enc, L, lambda i: split_heads_weight(enc[i].self_attn.v_proj.weight.T))
    oW = stack_per_layer(enc, L, lambda i: split_heads_weight(enc[i].self_attn.out_proj.weight))  # note: out wants [16,72,1152]

    qB = stack_per_layer(enc, L, lambda i: split_heads_bias(enc[i].self_attn.q_proj.bias))
    kB = stack_per_layer(enc, L, lambda i: split_heads_bias(enc[i].self_attn.k_proj.bias))
    vB = stack_per_layer(enc, L, lambda i: split_heads_bias(enc[i].self_attn.v_proj.bias))
    oB = stack_per_layer(enc, L, lambda i: enc[i].self_attn.out_proj.bias)  # [27,1152]

    assign(out, "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel", to_jax(qW))  # [27,1152,16,72]
    assign(out, "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel",   to_jax(kW))
    assign(out, "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel", to_jax(vW))
    # out kernel wants [27,16,72,1152]; oW is [27,1152,16,72] right now – transpose last dims:
    assign(out, "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel",
           to_jax(oW.permute(0,2,3,1)))
    assign(out, "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias", to_jax(qB))
    assign(out, "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias",   to_jax(kB))
    assign(out, "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias", to_jax(vB))
    assign(out, "PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias",   to_jax(oB))

    # Post encoder norm
    post = hf_model.vision_tower.vision_model.post_layernorm
    assign(out, "PaliGemma/img/Transformer/encoder_norm/scale", to_jax(post.weight))
    assign(out, "PaliGemma/img/Transformer/encoder_norm/bias",  to_jax(post.bias))

    # Multimodal projector: Linear(1152->2048). HF weight [2048,1152] -> NNX [1152,2048]
    pw = hf_model.multi_modal_projector.linear.weight
    pb = hf_model.multi_modal_projector.linear.bias
    assign(out, "PaliGemma/head/kernel", to_jax(pw.T))
    assign(out, "PaliGemma/head/bias",   to_jax(pb))

    return out

def map_gemma(hf_model, nnx_flat, *, copy_embeddings=True, tolerate_vocab_mismatch=True):
    out = {}

    mdl = hf_model.language_model.model
    L = 18
    assert len(mdl.layers) == L, f"Expected 18 Gemma layers, got {len(mdl.layers)}"

    # Embeddings (optional due to vocab mismatch)
    if copy_embeddings and "PaliGemma/llm/embedder/input_embedding" in nnx_flat:
        tgt = nnx_flat["PaliGemma/llm/embedder/input_embedding"]
        try:
            V_tgt, D = tgt.shape
        except AttributeError:
            V_tgt, D = tgt.value.shape
            
        emb = mdl.embed_tokens.weight  # [V_hf, D]
        V_hf = emb.shape[0]
        if V_hf != V_tgt:
            if not tolerate_vocab_mismatch:
                raise ValueError(f"Vocab size mismatch: HF {V_hf} vs target {V_tgt}")
            V = min(V_hf, V_tgt)
            emb = emb[:V]
            # leave rows [V:V_tgt] in target as random
        assign(out, "PaliGemma/llm/embedder/input_embedding", to_jax(emb))

    # Final norm
    assign(out, "PaliGemma/llm/final_norm/scale", to_jax(mdl.norm.weight))

    # Per-layer
    def split_qkv_o(layer):
        attn = layer.self_attn
        # Q: [2048,2048] -> [8,2048,256]
        q = attn.q_proj.weight.view(2048, 8, 256).permute(1,0,2)
        # K,V: [2048,256] -> [1,2048,256] each; stack -> [2,1,2048,256]
        k = attn.k_proj.weight[None, ...]  # [1,2048,256]
        v = attn.v_proj.weight[None, ...]  # [1,2048,256]
        kv = torch.stack([k, v], dim=0)    # [2,1,2048,256]
        # O: [2048,2048] -> split input 2048 as (8,256) then permute -> [8,256,2048]
        o = attn.o_proj.weight.view(8, 256, 2048)
        return q, kv, o

    qs, kvs, os = [], [], []
    pre_attn_norm, post_attn_norm = [], []
    pre_ffw_norm = []

    for i in range(L):
        layer = mdl.layers[i]
        q, kv, o = split_qkv_o(layer)
        qs.append(q)
        kvs.append(kv)
        os.append(o)
        pre_attn_norm.append(layer.input_layernorm.weight)         # -> pre_attention_norm/scale
        post_attn_norm.append(layer.post_attention_layernorm.weight)  # sometimes used; your tree uses pre_ffw_norm
        pre_ffw_norm.append(layer.post_attention_layernorm.weight) # your tree shows pre_ffw_norm == post_attention_layernorm

    # Stack: add leading layer dim
    q_all  = torch.stack(qs, dim=0)     # [18,8,2048,256]
    kv_all = torch.stack(kvs, dim=0)    # [18,2,1,2048,256]
    o_all  = torch.stack(os, dim=0)     # [18,8,256,2048]

    assign(out, "PaliGemma/llm/layers/attn/q_einsum/w",        to_jax(q_all))
    # assign(out, "PaliGemma/llm/layers/attn/kv_einsum/w",       to_jax(kv_all))
    assign(
        out,
        "PaliGemma/llm/layers/attn/kv_einsum/w",
        to_jax(kv_all.permute(0, 1, 2, 4, 3))  # [18, 2, 1, 256, 2048]
    )
    assign(out, "PaliGemma/llm/layers/attn/attn_vec_einsum/w", to_jax(o_all))

    # Norms
    assign(out, "PaliGemma/llm/layers/pre_attention_norm/scale",
           to_jax(torch.stack(pre_attn_norm, dim=0)))  # [18,2048]
    assign(out, "PaliGemma/llm/layers/pre_ffw_norm/scale",
           to_jax(torch.stack(pre_ffw_norm, dim=0)))   # [18,2048]

    gate = torch.stack([mdl.layers[i].mlp.gate_proj.weight for i in range(L)], dim=0)  # [L, 16384, 2048]
    up   = torch.stack([mdl.layers[i].mlp.up_proj.weight   for i in range(L)], dim=0)  # [L, 16384, 2048]
    gate_t = gate.permute(0, 2, 1)  # [L, 2048, 16384]
    up_t   = up.permute(0, 2, 1)    # [L, 2048, 16384]
    gating = torch.stack([gate_t, up_t], dim=1)  # [L, 2, 2048, 16384]
    put_with_shape(out, "PaliGemma/llm/layers/mlp/gating_einsum", gating, nnx_flat)

    down = torch.stack([mdl.layers[i].mlp.down_proj.weight for i in range(L)], dim=0)  # [L, 16384, 2048]
    put_with_shape(out, "PaliGemma/llm/layers/mlp/linear", down, nnx_flat)

    return out

@dataclasses.dataclass(frozen=True)
class HuggingFaceWeightLoader(WeightLoader):
    """Loads weights from a Hugging Face model.
    
    Args:
        repo_id: The Hugging Face repository ID (e.g., "google/paligemma-3b-pt-224").
        include_llm: Whether to load the LLM weights. Defaults to True.
    """
    repo_id: str
    include_llm: bool = True
    
    def load(self, params: at.Params) -> at.Params:
        logger.info(f"Loading weights from Hugging Face: {self.repo_id}")
        
        # 1. Download/Load HF model (using PyTorch to allow easy conversion)
        # Note: We use float32 to match safe default, but conversion to JAX will handle dtypes later if needed.
        hf_model = PaliGemmaForConditionalGeneration.from_pretrained(self.repo_id, torch_dtype=torch.float32)

        # 2. Get target shapes from params
        # We need to flatten params to map keys easily
        nnx_flat = flatten_pytree(params)

        # 3. Map weights
        out_flat = {}
        out_flat.update(map_siglip_and_projector(hf_model, nnx_flat))
        
        if self.include_llm:
            out_flat.update(map_gemma(hf_model, nnx_flat, copy_embeddings=True, tolerate_vocab_mismatch=True))
            
        # 4. Only keep keys that exist in target (defensive)
        out_flat = {k: v for k, v in out_flat.items() if k in nnx_flat}
        
        loaded_params = unflatten_pytree(out_flat)
        
        # 5. Merge with original params to fill in missing parts (like action heads)
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = traverse_util.flatten_dict(params, sep="/")
    flat_loaded = traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return traverse_util.unflatten_dict(result, sep="/")
