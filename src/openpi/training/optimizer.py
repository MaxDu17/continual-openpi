import dataclasses
from typing import Protocol, runtime_checkable, Optional, Union

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> Union[optax.Schedule, dict[str, optax.Schedule]]: ...


@dataclasses.dataclass(frozen=True)
class WarmupConstantSchedule(LRScheduleConfig):
    """Linear warmup then constant LR forever."""

    warmup_steps: int = 1_000
    lr: float = 2.5e-5

    def create(self) -> optax.Schedule:
        warmup = optax.linear_schedule(
            init_value=self.lr / (self.warmup_steps + 1),
            end_value=self.lr,
            transition_steps=self.warmup_steps,
        )
        constant = optax.constant_schedule(self.lr)

        # At step == warmup_steps, switch to constant schedule.
        return optax.join_schedules(
            schedules=[warmup, constant],
            boundaries=[self.warmup_steps],
        )

@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class CosineSmall(LRScheduleConfig):
    warmup_steps: int = 800
    peak_lr: float = 8e-4
    decay_steps: int = 19_200   # e.g., for a 30k-step run
    decay_lr: float = 1.6e-4     # 20% of peak

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )
        

@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@dataclasses.dataclass(frozen=True)
class PerComponentLRSchedule(LRScheduleConfig):
    """Different learning rates for different components."""
    
    # try these for now
    # vision: LRScheduleConfig = CosineDecaySchedule(
    #     warmup_steps=1_000,
    #     peak_lr=2.5e-6,
    #     decay_steps=30_000,
    #     decay_lr=2.5e-7,
    # )
    vision: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-5,
        decay_steps=30_000,
        decay_lr=1e-6,
    )
    # language: LRScheduleConfig = CosineDecaySchedule(
    #     warmup_steps=1_000,
    #     peak_lr=2.5e-6,
    #     decay_steps=30_000,
    #     decay_lr=2.5e-7,
    # )
    language: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-5,
        decay_steps=30_000,
        decay_lr=1e-6,
    )
    action: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    )
    default: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    )

    def create(self) -> dict[str, optax.Schedule]:
        return {
            "vision": self.vision.create(),
            "language": self.language.create(),
            "action": self.action.create(),
            "default": self.default.create(),
        }

@dataclasses.dataclass(frozen=True)
class PerComponentLRSchedule1(PerComponentLRSchedule):
    """Different learning rates for different components."""
    
    # try these for now
    # vision: LRScheduleConfig = CosineDecaySchedule(
    #     warmup_steps=1_000,
    #     peak_lr=2.5e-6,
    #     decay_steps=30_000,
    #     decay_lr=2.5e-7,
    # )
    vision: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-5,
        decay_steps=30_000,
        decay_lr=1e-6,
    )
    # language: LRScheduleConfig = CosineDecaySchedule(
    #     warmup_steps=1_000,
    #     peak_lr=2.5e-6,
    #     decay_steps=30_000,
    #     decay_lr=2.5e-7,
    # )
    language: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-5,
        decay_steps=30_000,
        decay_lr=1e-6,
    )
    action: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    )
    default: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    )

    def create(self) -> dict[str, optax.Schedule]:
        return {
            "vision": self.vision.create(),
            "language": self.language.create(),
            "action": self.action.create(),
            "default": self.default.create(),
        }


@dataclasses.dataclass(frozen=True)
class PerComponentLRSchedule2(PerComponentLRSchedule):
    """Different learning rates for different components."""
    
    # try these for now
    vision: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-6,
        decay_steps=30_000,
        decay_lr=2.5e-7,
    )
    # vision: LRScheduleConfig = CosineDecaySchedule(
    #     warmup_steps=1_000,
    #     peak_lr=1e-5,
    #     decay_steps=30_000,
    #     decay_lr=1e-6,
    # )
    language: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-6,
        decay_steps=30_000,
        decay_lr=2.5e-7,
    )
    # language: LRScheduleConfig = CosineDecaySchedule(
    #     warmup_steps=1_000,
    #     peak_lr=1e-5,
    #     decay_steps=30_000,
    #     decay_lr=1e-6,
    # )
    action: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    )
    default: LRScheduleConfig = CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    )

    def create(self) -> dict[str, optax.Schedule]:
        return {
            "vision": self.vision.create(),
            "language": self.language.create(),
            "action": self.action.create(),
            "default": self.default.create(),
        }

@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


# def create_optimizer(
#     optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
# ) -> optax.GradientTransformation:
#     lr = lr_schedule.create()
#     return optimizer.create(lr, weight_decay_mask=weight_decay_mask)


def create_optimizer(
    optimizer: OptimizerConfig,
    lr_schedule: LRScheduleConfig,
    weight_decay_mask: at.PyTree | None = None,
    *,
    param_labels: Optional[at.PyTree] = None,   # NEW
) -> optax.GradientTransformation:
    lr = lr_schedule.create()

    # Old behavior: single schedule/scalar
    if not isinstance(lr, dict):
        return optimizer.create(lr, weight_decay_mask=weight_decay_mask)

    # New behavior: dict of schedules => multi_transform
    if param_labels is None:
        raise ValueError(
            "Per-component LR schedule provided (dict), but param_labels is None. "
            "Build a param_labels pytree with labels like 'vision', 'language', 'action', 'default'."
        )

    # Build one transform per label, each with its own LR schedule
    transforms: dict[str, optax.GradientTransformation] = {
        name: optimizer.create(lr[name], weight_decay_mask=weight_decay_mask)
        for name in lr.keys()
    }

    return optax.multi_transform(transforms, param_labels)

# def create_optimizer(optimizer, lr_schedule, weight_decay_mask=None, *, param_labels=None):
#     lr = lr_schedule.create()

#     # old behavior
#     if not isinstance(lr, dict):
#         return optimizer.create(lr, weight_decay_mask=weight_decay_mask)

#     assert param_labels is not None

#     # IMPORTANT:
#     # build AdamW tx *without* internal clipping
#     if isinstance(optimizer, AdamW):
#         def make_adamw(schedule):
#             return optax.adamw(
#                 schedule,
#                 b1=optimizer.b1,
#                 b2=optimizer.b2,
#                 eps=optimizer.eps,
#                 weight_decay=optimizer.weight_decay,
#                 mask=weight_decay_mask,
#             )

#         transforms = {k: make_adamw(lr[k]) for k in lr.keys()}
#         multi_tx = optax.multi_transform(transforms, param_labels)

#         # apply GLOBAL clipping once
#         return optax.chain(optax.clip_by_global_norm(optimizer.clip_gradient_norm), multi_tx)

#     # fallback: keep old behavior (but note: clipping changes if optimizer internally clips)
#     transforms = {k: optimizer.create(lr[k], weight_decay_mask=weight_decay_mask) for k in lr.keys()}
#     return optax.multi_transform(transforms, param_labels)