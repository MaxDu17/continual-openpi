"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_l2 as pi0_l2
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_L2:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, asset_id=None) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        
        if asset_id is None:
            asset_id = self.assets.asset_id or repo_id
        else:
            asset_id = asset_id
            
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoSeqDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    repo_ids: tuple[str, ...] = dataclasses.field(default_factory=tuple)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, task_id: int | None = None) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        
        if task_id is None:
            raise ValueError(
                "`task_id` must be provided when using LeRobotLiberoSeqDataConfig "
                "(this factory is only meant for sequential training)."
            )
        repo_id = self.repo_ids[task_id]
        
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # base = self.create_base_config(assets_dirs)
        
        # task_assets = dataclasses.replace(
        #     self.assets,            
        #     asset_id = repo_id      
        # )
        
        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            # base,
            # assets=task_assets,
            self.create_base_config(assets_dirs, asset_id=repo_id),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            repo_id=repo_id,
        )
        
@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[robocasa_policy.RobocasaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        # delta_action_mask = _transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[_transforms.DeltaActions(delta_action_mask)],
        #     outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        # )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        


@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaSeqDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    repo_ids: tuple[str, ...] = dataclasses.field(default_factory=tuple)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, task_id: int | None = None) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        
        if task_id is None:
            raise ValueError(
                "`task_id` must be provided when using LeRobotLiberoSeqDataConfig "
                "(this factory is only meant for sequential training)."
            )
        repo_id = self.repo_ids[task_id]
        
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[robocasa_policy.RobocasaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        # delta_action_mask = _transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[_transforms.DeltaActions(delta_action_mask)],
        #     outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        # )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # base = self.create_base_config(assets_dirs)
        
        # task_assets = dataclasses.replace(
        #     self.assets,            
        #     asset_id = repo_id      
        # )
        
        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            # base,
            # assets=task_assets,
            self.create_base_config(assets_dirs, asset_id=repo_id),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            repo_id=repo_id,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
        )


@dataclasses.dataclass(frozen=True)
class PackNetConfig:
    prune_perc: float = 0.75          # same meaning
    post_prune_epochs: int = 50       # convert to steps; see below
    post_eval_every: int = 5          # convert to steps; see below
    log_pruning: bool = True            # prints layer‑wise stats on host if True
    
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.NoOpWeightLoader | weight_loaders.CheckpointWeightLoader | weight_loaders.PaliGemmaWeightLoader | weight_loaders.HuggingFaceWeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    lr_schedule: _optimizer.LRScheduleConfig | _optimizer.PerComponentLRSchedule = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000
    steps_per_task: int = 10

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 4

    multitask_to_i: int = -1

    cl: str = ""
    n_tasks: int = 10
    
    # ER
    n_memories: int = 1000 
    double_batch_start: bool = False
    starting_from_task_idx: int = -1 # for debugging
    no_resampling: bool = False 
    cl_order: str = None
    
    ################################
    ####### Physics of Lifelong VLA 
    
    # Lang description
    override_task: str = None # choice: "task_index" --> override task description for ablating language
    
    # Ablation pretrained model
    train_from_scratch: str = None # choice: "from_paligemma", "from_scratch", "from_action_expert"
    
    learn_one_task_idx: int = -1  # only learn one task
   
    # Linear Probing
    probe_step: int = -1 # needs to be specified
    probe_task_id: int = -1 # needs to be specified

    # CKA analysis
    cka: bool = False
    action_features: bool = False 
    
    backbone_checkpoint_dir: str = None

    # --- Merge Configs ---
    # Steps for checkpoint merging
    merge_base_step: int = 9999
    merge_action_step: int = 19999
    
    # Flags for what to take from the second checkpoint (Ckpt2)
    merge_vision: int = 0
    merge_language: int = 0
    merge_action: int = 1
    merge_extra_projections: int = 1
    
    # Optional: Allow overriding the paths via CLI if you don't want them hardcoded
    ckpt1_path: str = "checkpoints/pi0_libero_low_mem_finetune-libero_spatial_sequential/er_bs8_sample10"
    ckpt2_path: str = "checkpoints/pi0_libero_low_mem_finetune-libero_spatial_sequential/er_bs8_sample10"

    ################################
    
    # EWC
    ewc_lambda: float = 50000       # importance of the penalty
    ewc_gamma:  float = 0.9       # Fisher decay across tasks
    ewc_max_batches: int = 50
    
    packnet = PackNetConfig(prune_perc=0)
    
    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_FREEZE_SUFFIX = {
    None: "",           # default LoRA freeze filter from .get_freeze_filter()
    "none": "-full-ft",
    "llm": "-freeze-llm",
    "img": "-freeze-img",
    "both": "-freeze-llm-img",
    "img_llm_only_lora_open": "-img_llm_only_lora_open",
    "img_llm_lora_action_full": "-img_llm_lora_action_full",
    "img_llm_full_action_lora": "-img_llm_full_action_lora",
    "img_freeze_llm_freeze_action_full": "-img_freeze_llm_freeze_action_full",
    "img_freeze_llm_freeze_action_lora": "-img_freeze_llm_freeze_action_lora",
    "img_freeze_llm_full_action_freeze": "-img_freeze_llm_full_action_freeze",
}

def _make_seq_train_config(
    base_name: str,
    repo_ids: tuple[str, ...],
    freeze_mode: str | None,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    siglip_variant="So400m/14",
    loss_type=None,
    img_module="siglip",
    robocasa=False,
) -> TrainConfig:
    
    if robocasa:
        name = f"pi0_robocasa_low_mem_finetune-sequential-{base_name}{_FREEZE_SUFFIX[freeze_mode]}"
    else: # libero
        """Builds a sequential Libero TrainConfig for a given freeze_mode."""
        name = f"pi0_libero_low_mem_finetune-{base_name}{_FREEZE_SUFFIX[freeze_mode]}"
    
    if action_expert_variant != "gemma_300m_lora":
        name += f"-{paligemma_variant}-{action_expert_variant}"
    
    if siglip_variant != "So400m/14":
        name += f"-{siglip_variant}"
    
    if loss_type == "l2":
        name += f"-l2loss"
        
    if img_module != "siglip":
        name += f"-{img_module}"
    
    asset_name = f"pi0_libero_low_mem_finetune-{base_name}"
    
    model_cfg = pi0.Pi0Config(
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        freeze_mode=freeze_mode,
        siglip_variant=siglip_variant,
        loss_type=loss_type,
        img_module=img_module,
    )
    
    if img_module == "resnet":
        
        return TrainConfig(
            name=name,
            model=model_cfg,
            data=LeRobotLiberoSeqDataConfig(
                # repo_id isn't used by LeRobotLiberoSeqDataConfig; it takes repo_ids + task_id at runtime
                repo_id=f"huihanl/{base_name.replace('_sequential','').replace('-','_')}",  # just a placeholder
                repo_ids=repo_ids,
                base_config=DataConfig(prompt_from_task=True),
                assets=AssetsConfig(
                    assets_dir=f"./assets/{asset_name}",
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=100000,
            steps_per_task=10000,
            save_interval=2000,
            freeze_filter=model_cfg.get_freeze_filter(),
            # Turn off EMA for LoRA finetuning.
            ema_decay=None,
            lr_schedule=_optimizer.CosineSmall()
        )
    
    if "nano" in paligemma_variant or "nano" in action_expert_variant \
        or "micro" in paligemma_variant or "micro" in action_expert_variant \
            or "tiny" in paligemma_variant or "tiny" in action_expert_variant \
                or "pico" in paligemma_variant or "pico" in action_expert_variant:
                    return TrainConfig(
                        name=name,
                        model=model_cfg,
                        data=LeRobotLiberoSeqDataConfig(
                            # repo_id isn't used by LeRobotLiberoSeqDataConfig; it takes repo_ids + task_id at runtime
                            repo_id=f"huihanl/{base_name.replace('_sequential','').replace('-','_')}",  # just a placeholder
                            repo_ids=repo_ids,
                            base_config=DataConfig(prompt_from_task=True),
                            assets=AssetsConfig(
                                assets_dir=f"./assets/{asset_name}",
                            ),
                        ),
                        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
                        num_train_steps=100000,
                        steps_per_task=10000,
                        save_interval=2000,
                        freeze_filter=model_cfg.get_freeze_filter(),
                        # Turn off EMA for LoRA finetuning.
                        ema_decay=None,
                        lr_schedule=_optimizer.CosineSmall()
                    )
    
    return TrainConfig(
        name=name,
        model=model_cfg,
        data=LeRobotLiberoSeqDataConfig(
            # repo_id isn't used by LeRobotLiberoSeqDataConfig; it takes repo_ids + task_id at runtime
            repo_id=f"huihanl/{base_name.replace('_sequential','').replace('-','_')}",  # just a placeholder
            repo_ids=repo_ids,
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir=f"./assets/{asset_name}",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        freeze_filter=model_cfg.get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    )

def _seq_variants_for_benchmark(base_name: str, repo_ids: tuple[str, ...]) -> list[TrainConfig]:
    """Produce all freeze-mode variants for a given sequential benchmark."""
    freeze_modes = ["none", "llm", "img", "both", 
                    "img_llm_only_lora_open", 
                    "img_llm_lora_action_full", 
                    "img_llm_full_action_lora",
                    "img_freeze_llm_freeze_action_full",
                    "img_freeze_llm_freeze_action_lora",
                    "img_freeze_llm_full_action_freeze",
                    ]
    variants = [_make_seq_train_config(base_name, repo_ids, fm) for fm in freeze_modes]
    variants.extend(
        [_make_seq_train_config(base_name, repo_ids, fm, siglip_variant="B/16") for fm in freeze_modes]
    )
    
    paligemma_variants = [
        "gemma_llm_small_220m",
        "gemma_2b_lora",
        "gemma_llm_small_120m",
        "gemma_tiny_8m",
        "gemma_nano_3m",
        "gemma_micro_800k",
        "gemma_pico_300k",
    ]

    action_expert_variants = [
        "gemma_300m",
        "gemma_100m",
        "gemma_30m",
        "gemma_15m",
        "gemma_tiny_8m",
        "gemma_nano_3m",
        "gemma_micro_800k",
        "gemma_pico_300k",
    ]

    for paligemma_variant in paligemma_variants:
        for action_expert_variant in action_expert_variants:
            freeze_modes = ["none", "img_freeze_llm_freeze_action_full", "img"]
            for fm in freeze_modes:
                variants.append(
                    _make_seq_train_config(
                        base_name, repo_ids, fm, 
                        paligemma_variant=paligemma_variant,
                        action_expert_variant=action_expert_variant,
                    )
                )
                
                variants.append(
                    _make_seq_train_config(
                        base_name, repo_ids, fm, 
                        paligemma_variant=paligemma_variant,
                        action_expert_variant=action_expert_variant,
                        siglip_variant="B/16",
                    )
                )
                
                variants.append(
                    _make_seq_train_config(
                        base_name, repo_ids, fm, 
                        paligemma_variant=paligemma_variant,
                        action_expert_variant=action_expert_variant,
                        img_module="resnet",
                    )
                )
    
    return variants

def _make_multitask_to_i_config(
    base_name: str,
    repo_ids: tuple[str, ...],
    freeze_mode: str | None,
    multitask_to_i: int = 10,
) -> TrainConfig:
    """Builds a multitask-to-i Libero TrainConfig with a given freeze_mode."""
    suffix = _FREEZE_SUFFIX[freeze_mode]
    name = f"pi0_libero_low_mem_finetune-{base_name}-multi_task-to_i{suffix}"
    asset_name = f"pi0_libero_low_mem_finetune-{base_name}_sequential"
    model_cfg = pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        freeze_mode=freeze_mode,
    )
    return TrainConfig(
        name=name,
        model=model_cfg,
        data=LeRobotLiberoSeqDataConfig(
            repo_id=f"huihanl/{base_name.replace('-','_')}_no_noops",  # placeholder, not used
            repo_ids=repo_ids,
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir=f"./assets/{asset_name}",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=model_cfg.get_freeze_filter(),
        ema_decay=None,
        multitask_to_i=multitask_to_i,
    )

def _multitask_to_i_variants(base_name: str, repo_ids: tuple[str, ...]) -> list[TrainConfig]:
    """Produce all freeze-mode variants for a given multitask-to-i benchmark."""
    freeze_modes = ["none", "llm", "img", "both", 
                    "img_llm_only_lora_open", 
                    "img_llm_lora_action_full", 
                    "img_llm_full_action_lora",
                    "img_freeze_llm_freeze_action_full",
                    "img_freeze_llm_freeze_action_lora",
                    "img_freeze_llm_full_action_freeze",
                    ]
    return [_make_multitask_to_i_config(base_name, repo_ids, fm) for fm in freeze_modes]


def _make_robocasa_seq_train_config(
    base_name: str,
    repo_ids: tuple[str, ...],
    freeze_mode: str | None,
    paligemma_variant: str = "gemma_2b_lora",
    action_expert_variant: str = "gemma_300m_lora",
    siglip_variant: str = "So400m/14",
    loss_type: str | None = None,
    img_module: str = "siglip",
) -> TrainConfig:
    """
    Builds a sequential RoboCasa TrainConfig with the same variant logic as Libero.
    Asset directory follows: .../assets/pi0_robocasa_low_mem_finetune-sequential-{base_name}
    Checkpoint path uses gs:// (match your Libero defaults).
    """
    suffix = _FREEZE_SUFFIX[freeze_mode]
    # e.g., pi0_robocasa_low_mem_finetune-sequential-all[-extras]
    name = f"pi0_robocasa_low_mem_finetune-sequential-{base_name}{suffix}"

    if action_expert_variant != "gemma_300m_lora":
        name += f"-{paligemma_variant}-{action_expert_variant}"
    if siglip_variant != "So400m/14":
        name += f"-{siglip_variant}"
    if loss_type == "l2":
        name += f"-l2loss"
    if img_module != "siglip":
        name += f"-{img_module}"

    asset_name = f"pi0_robocasa_low_mem_finetune-sequential-all"

    model_cfg = pi0.Pi0Config(
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        freeze_mode=freeze_mode,
        siglip_variant=siglip_variant,
        loss_type=loss_type,
        img_module=img_module,
    )

    data_cfg = LeRobotRobocasaSeqDataConfig(
        # repo_id not used; LeRobotRobocasaSeqDataConfig consumes repo_ids + task_id at runtime
        repo_id=f"robocasa/{base_name}",  # dummy placeholder
        repo_ids=repo_ids,
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir=f"./assets/{asset_name}",
        ),
    )

    # Shared defaults with your Libero sequential configs
    common_kwargs = dict(
        name=name,
        model=model_cfg,
        data=data_cfg,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        freeze_filter=model_cfg.get_freeze_filter(),
        ema_decay=None,
        n_tasks=len(repo_ids),
    )

    # Special-case tiny variants to mirror your Libero behavior
    small_keywords = ("nano", "micro", "tiny", "pico")
    if any(k in paligemma_variant for k in small_keywords) or any(k in action_expert_variant for k in small_keywords):
        return TrainConfig(lr_schedule=_optimizer.CosineSmall(), **common_kwargs)

    if img_module == "resnet":
        return TrainConfig(lr_schedule=_optimizer.CosineSmall(), **common_kwargs)

    return TrainConfig(**common_kwargs)

# -------------------------------
# RoboCasa: generate full variant grid
# -------------------------------
def _robocasa_seq_variants_for_benchmark(
    base_name: str,
    repo_ids: tuple[str, ...],
) -> list[TrainConfig]:
    """Produce all freeze-mode and model-size variants for a given RoboCasa sequential benchmark."""
    freeze_modes = [
        "none", "llm", "img", "both",
        "img_llm_only_lora_open",
        "img_llm_lora_action_full",
        "img_llm_full_action_lora",
        "img_freeze_llm_freeze_action_full",
        "img_freeze_llm_freeze_action_lora",
        "img_freeze_llm_full_action_freeze",
    ]

    variants: list[TrainConfig] = []
    # Base sweep for default paligemma/action_expert + two vision backbones
    variants.extend(
        [_make_robocasa_seq_train_config(base_name, repo_ids, fm) for fm in freeze_modes]
    )
    variants.extend(
        [_make_robocasa_seq_train_config(base_name, repo_ids, fm, siglip_variant="B/16") for fm in freeze_modes]
    )

    paligemma_variants = [
        "gemma_llm_small_220m",
        "gemma_2b_lora",
        "gemma_llm_small_120m",
        "gemma_tiny_8m",
        "gemma_nano_3m",
        "gemma_micro_800k",
        "gemma_pico_300k",
    ]
    action_expert_variants = [
        "gemma_300m",
        "gemma_100m",
        "gemma_30m",
        "gemma_15m",
        "gemma_tiny_8m",
        "gemma_nano_3m",
        "gemma_micro_800k",
        "gemma_pico_300k",
    ]

    for paligemma_variant in paligemma_variants:
        for action_expert_variant in action_expert_variants:
            for fm in ["none", "img_freeze_llm_freeze_action_full", "img"]:
                variants.append(
                    _make_robocasa_seq_train_config(
                        base_name, repo_ids, fm,
                        paligemma_variant=paligemma_variant,
                        action_expert_variant=action_expert_variant,
                    )
                )
                variants.append(
                    _make_robocasa_seq_train_config(
                        base_name, repo_ids, fm,
                        paligemma_variant=paligemma_variant,
                        action_expert_variant=action_expert_variant,
                        siglip_variant="B/16",
                    )
                )
                variants.append(
                    _make_robocasa_seq_train_config(
                        base_name, repo_ids, fm,
                        paligemma_variant=paligemma_variant,
                        action_expert_variant=action_expert_variant,
                        img_module="resnet",
                    )
                )

    return variants

# ========= Libero sequential task lists (from your existing configs) =========

_LIBERO_GOAL_SEQ = (
    "huihanl/libero-libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet",
    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_the_stove",
    "huihanl/libero-libero_goal_no_noops/put_the_wine_bottle_on_top_of_the_cabinet",
    "huihanl/libero-libero_goal_no_noops/open_the_top_drawer_and_put_the_bowl_inside",
    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_top_of_the_cabinet",
    "huihanl/libero-libero_goal_no_noops/push_the_plate_to_the_front_of_the_stove",
    "huihanl/libero-libero_goal_no_noops/put_the_cream_cheese_in_the_bowl",
    "huihanl/libero-libero_goal_no_noops/turn_on_the_stove",
    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_the_plate",
    "huihanl/libero-libero_goal_no_noops/put_the_wine_bottle_on_the_rack",
)

_LIBERO_SPATIAL_SEQ = (
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
)

_LIBERO_OBJECT_SEQ = (
    "huihanl/libero-libero_object_no_noops/pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "huihanl/libero-libero_object_no_noops/pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "huihanl/libero-libero_object_no_noops/pick_up_the_butter_and_place_it_in_the_basket",
    "huihanl/libero-libero_object_no_noops/pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "huihanl/libero-libero_object_no_noops/pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "huihanl/libero-libero_object_no_noops/pick_up_the_ketchup_and_place_it_in_the_basket",
    "huihanl/libero-libero_object_no_noops/pick_up_the_milk_and_place_it_in_the_basket",
    "huihanl/libero-libero_object_no_noops/pick_up_the_orange_juice_and_place_it_in_the_basket",
    "huihanl/libero-libero_object_no_noops/pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "huihanl/libero-libero_object_no_noops/pick_up_the_tomato_sauce_and_place_it_in_the_basket",
)

_LIBERO_10_SEQ = (
    "huihanl/libero-libero_10_no_noops/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "huihanl/libero-libero_10_no_noops/put_both_moka_pots_on_the_stove",
    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "huihanl/libero-libero_10_no_noops/put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "huihanl/libero-libero_10_no_noops/put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "huihanl/libero-libero_10_no_noops/put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "huihanl/libero-libero_10_no_noops/turn_on_the_stove_and_put_the_moka_pot_on_it",
)


_ROBOCASA_ALL_SEQ: tuple[str, ...] = (
    "robocasa/single_task/CloseDoubleDoor",
    "robocasa/single_task/CloseDrawer",
    "robocasa/single_task/CloseSingleDoor",
    "robocasa/single_task/CoffeePressButton",
    "robocasa/single_task/CoffeeServeMug",
    "robocasa/single_task/CoffeeSetupMug",
    "robocasa/single_task/OpenDoubleDoor",
    "robocasa/single_task/OpenDrawer",
    "robocasa/single_task/OpenSingleDoor",
    "robocasa/single_task/PnPCabToCounter",
    "robocasa/single_task/PnPCounterToCab",
    "robocasa/single_task/PnPCounterToMicrowave",
    "robocasa/single_task/PnPCounterToSink",
    "robocasa/single_task/PnPCounterToStove",
    "robocasa/single_task/PnPMicrowaveToCounter",
    "robocasa/single_task/PnPSinkToCounter",
    "robocasa/single_task/PnPStoveToCounter",
    "robocasa/single_task/TurnOffMicrowave",
    "robocasa/single_task/TurnOffSinkFaucet",
    "robocasa/single_task/TurnOffStove",
    "robocasa/single_task/TurnOnMicrowave",
    "robocasa/single_task/TurnOnSinkFaucet",
    "robocasa/single_task/TurnOnStove",
    "robocasa/single_task/TurnSinkSpout",
)

_ROBOCASA_DOOR_SEQ: tuple[str, ...] = (
    "robocasa/single_task/CloseDoubleDoor",
    "robocasa/single_task/CloseDrawer",
    "robocasa/single_task/CloseSingleDoor",
    "robocasa/single_task/OpenDoubleDoor",
    "robocasa/single_task/OpenDrawer",
    "robocasa/single_task/OpenSingleDoor",
)

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero",
                asset_id="huihanl/libero",
                ),
            ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero",
                asset_id="huihanl/libero",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_10",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero_low_mem_finetune-libero_10",
                asset_id="huihanl/libero-libero_10_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
   

    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_10_from_libero_object",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero_low_mem_finetune-libero_10",
                asset_id="huihanl/libero-libero_10_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_fast_libero_low_mem_finetune-libero_object/my_experiment/8000/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),

    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_goal",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero-libero_goal_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero_low_mem_finetune-libero_goal",
                asset_id="huihanl/libero-libero_goal_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_object",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero-libero_object_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero_low_mem_finetune-libero_object",
                asset_id="huihanl/libero-libero_object_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_object-single_task",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero-libero_object_no_noops-pick_up_the_alphabet_soup_and_place_it_in_the_basket",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero_low_mem_finetune-libero_object-single_task",
                asset_id="huihanl/libero-libero_object_no_noops-pick_up_the_alphabet_soup_and_place_it_in_the_basket",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_object-multi_task",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/mt/libero-libero_object_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_object-multi_task",
                asset_id="huihanl/libero-libero_object_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),

   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_10-multi_task",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/mt/libero-libero_10_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_10-multi_task",
                asset_id="huihanl/libero-libero_10_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
   
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_spatial-multi_task",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/mt/libero-libero_spatial_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_spatial-multi_task",
                asset_id="huihanl/libero-libero_spatial_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
   
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_goal-multi_task",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/mt/libero-libero_goal_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_goal-multi_task",
                asset_id="huihanl/libero-libero_goal_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
   
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_object-multi_task-to_i",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_object_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_object_no_noops/pick_up_the_alphabet_soup_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_bbq_sauce_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_butter_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_cream_cheese_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_ketchup_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_milk_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_orange_juice_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_salad_dressing_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_tomato_sauce_and_place_it_in_the_basket",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_object_sequential",
                ),
        ),
        
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        multitask_to_i=10,
        
    ),

   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_10-multi_task-to_i",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_10_no_noops/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
                    "huihanl/libero-libero_10_no_noops/put_both_moka_pots_on_the_stove",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
                    "huihanl/libero-libero_10_no_noops/turn_on_the_stove_and_put_the_moka_pot_on_it",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_10_sequential",
                ),
        ),
        
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        multitask_to_i=10,
        
    ),
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_10-multi_task-to_i-l2_loss",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_l2.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", loss_type="l2"),
        
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_10_no_noops/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
                    "huihanl/libero-libero_10_no_noops/put_both_moka_pots_on_the_stove",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
                    "huihanl/libero-libero_10_no_noops/turn_on_the_stove_and_put_the_moka_pot_on_it",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_10_sequential",
                ),
        ),
        
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_l2.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", loss_type="l2"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        multitask_to_i=10,
    ),
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_spatial-multi_task-to_i",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_spatial_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_spatial_sequential",
                ),
        ),
        
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        multitask_to_i=10,
        
    ),
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_spatial-multi_task-to_i-l2_loss",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_l2.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", loss_type="l2"),
        
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_spatial_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_spatial_sequential",
                ),
        ),
        
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", loss_type="l2"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        multitask_to_i=10,
        
    ),
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_spatial-multi_task-to_i-resnet",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_micro_800k", action_expert_variant="gemma_pico_300k", img_module="resnet"),
        
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_spatial_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_spatial_sequential",
                ),
        ),
        
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_micro_800k", action_expert_variant="gemma_pico_300k", img_module="resnet"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        multitask_to_i=10,
        lr_schedule=_optimizer.CosineSmall()
    ),
   
   
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_goal-multi_task-to_i",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_goal_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet",
                    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_the_stove",
                    "huihanl/libero-libero_goal_no_noops/put_the_wine_bottle_on_top_of_the_cabinet",
                    "huihanl/libero-libero_goal_no_noops/open_the_top_drawer_and_put_the_bowl_inside",
                    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_top_of_the_cabinet",
                    "huihanl/libero-libero_goal_no_noops/push_the_plate_to_the_front_of_the_stove",
                    "huihanl/libero-libero_goal_no_noops/put_the_cream_cheese_in_the_bowl",
                    "huihanl/libero-libero_goal_no_noops/turn_on_the_stove",
                    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_the_plate",
                    "huihanl/libero-libero_goal_no_noops/put_the_wine_bottle_on_the_rack",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_goal_sequential",
                ),
        ),
        
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        multitask_to_i=10,
        
    ),
    
   
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_object-single_task",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero-libero_object_no_noops-pick_up_the_alphabet_soup_and_place_it_in_the_basket",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_object-single_task",
                asset_id="huihanl/libero-libero_object_no_noops-pick_up_the_alphabet_soup_and_place_it_in_the_basket",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),

    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_10-single_task",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops-put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero_low_mem_finetune-libero_10-single_task",
                asset_id="huihanl/libero-libero_10_no_noops-put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),

    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_object_from_libero_10",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero-libero_object_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero_low_mem_finetune-libero_object",
                asset_id="huihanl/libero-libero_object_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_fast_libero_low_mem_finetune-libero_10/my_experiment/8000/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),

    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_spatial",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="huihanl/libero-libero_spatial_no_noops",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero_low_mem_finetune-libero_spatial",
                asset_id="huihanl/libero-libero_spatial_no_noops",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        name="pi0_fast_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
   


    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_object_sequential",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_object_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_object_no_noops/pick_up_the_alphabet_soup_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_bbq_sauce_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_butter_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_cream_cheese_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_ketchup_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_milk_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_orange_juice_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_salad_dressing_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_tomato_sauce_and_place_it_in_the_basket",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_object_sequential",
                ),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
       
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_10_sequential",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_10_no_noops/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
                    "huihanl/libero-libero_10_no_noops/put_both_moka_pots_on_the_stove",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
                    "huihanl/libero-libero_10_no_noops/turn_on_the_stove_and_put_the_moka_pot_on_it",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_10_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),

    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_goal_sequential",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_goal_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet",
                    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_the_stove",
                    "huihanl/libero-libero_goal_no_noops/put_the_wine_bottle_on_top_of_the_cabinet",
                    "huihanl/libero-libero_goal_no_noops/open_the_top_drawer_and_put_the_bowl_inside",
                    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_top_of_the_cabinet",
                    "huihanl/libero-libero_goal_no_noops/push_the_plate_to_the_front_of_the_stove",
                    "huihanl/libero-libero_goal_no_noops/put_the_cream_cheese_in_the_bowl",
                    "huihanl/libero-libero_goal_no_noops/turn_on_the_stove",
                    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_the_plate",
                    "huihanl/libero-libero_goal_no_noops/put_the_wine_bottle_on_the_rack",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_goal_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_spatial_sequential",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_spatial_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_spatial_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/hf_models/retain-berkeley-stanford-droid/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    

    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_spatial_sequential-all",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_spatial_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_spatial_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/hf_models/retain-all-droid/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),    


    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_90_sequential",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_90_no_noops",# not used
            repo_ids=(
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
                
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE3_turn_on_the_stove",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE6_close_the_microwave",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE7_open_the_microwave",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE8_turn_off_the_stove",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE9_turn_on_the_stove",
                "huihanl/libero-libero_90_no_noops/KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",

                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
                #"huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
                "huihanl/libero-libero_90_no_noops/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
                
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
                "huihanl/libero-libero_90_no_noops/STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_90_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        n_tasks=89,
        
    ),

 
 
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_goal_sequential-lr",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_goal_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet",
                    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_the_stove",
                    "huihanl/libero-libero_goal_no_noops/put_the_wine_bottle_on_top_of_the_cabinet",
                    "huihanl/libero-libero_goal_no_noops/open_the_top_drawer_and_put_the_bowl_inside",
                    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_top_of_the_cabinet",
                    "huihanl/libero-libero_goal_no_noops/push_the_plate_to_the_front_of_the_stove",
                    "huihanl/libero-libero_goal_no_noops/put_the_cream_cheese_in_the_bowl",
                    "huihanl/libero-libero_goal_no_noops/turn_on_the_stove",
                    "huihanl/libero-libero_goal_no_noops/put_the_bowl_on_the_plate",
                    "huihanl/libero-libero_goal_no_noops/put_the_wine_bottle_on_the_rack",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_goal_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
                
        lr_schedule=_optimizer.PerComponentLRSchedule()
    ),
 
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_object_sequential-lr",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_object_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_object_no_noops/pick_up_the_alphabet_soup_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_bbq_sauce_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_butter_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_cream_cheese_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_ketchup_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_milk_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_orange_juice_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_salad_dressing_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_tomato_sauce_and_place_it_in_the_basket",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_object_sequential",
                ),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
       
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        lr_schedule=_optimizer.PerComponentLRSchedule()  
    ),
    
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_10_sequential-lr",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_10_no_noops/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
                    "huihanl/libero-libero_10_no_noops/put_both_moka_pots_on_the_stove",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
                    "huihanl/libero-libero_10_no_noops/turn_on_the_stove_and_put_the_moka_pot_on_it",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_10_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        lr_schedule=_optimizer.PerComponentLRSchedule()  
    ),
    
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_10_sequential-lr1",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_10_no_noops/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
                    "huihanl/libero-libero_10_no_noops/put_both_moka_pots_on_the_stove",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
                    "huihanl/libero-libero_10_no_noops/turn_on_the_stove_and_put_the_moka_pot_on_it",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_10_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        lr_schedule=_optimizer.PerComponentLRSchedule1()  
    ),
    
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_10_sequential-lr2",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_10_no_noops/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
                    "huihanl/libero-libero_10_no_noops/put_both_moka_pots_on_the_stove",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
                    "huihanl/libero-libero_10_no_noops/turn_on_the_stove_and_put_the_moka_pot_on_it",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_10_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        lr_schedule=_optimizer.PerComponentLRSchedule2()  
    ),
 
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_spatial_sequential-lr",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_spatial_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_spatial_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        
        lr_schedule=_optimizer.PerComponentLRSchedule()  
    ),
 
    TrainConfig(
        name="pi0_libero_low_mem_finetune-libero_spatial_sequential-l2_loss",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", loss_type="l2"),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_spatial_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                    "huihanl/libero-libero_spatial_no_noops/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_spatial_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", loss_type="l2"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
 
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_10_sequential",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_10_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_10_no_noops/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
                    "huihanl/libero-libero_10_no_noops/put_both_moka_pots_on_the_stove",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
                    "huihanl/libero-libero_10_no_noops/put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
                    "huihanl/libero-libero_10_no_noops/put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
                    "huihanl/libero-libero_10_no_noops/turn_on_the_stove_and_put_the_moka_pot_on_it",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_libero_low_mem_finetune-libero_10_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/hf_models/retain-berkeley-stanford-droid/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=200000,
        steps_per_task=20000,
        save_interval=500,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune-libero_object_sequential",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoSeqDataConfig(
            repo_id="huihanl/libero-libero_object_no_noops",# not used
            repo_ids=(
                    "huihanl/libero-libero_object_no_noops/pick_up_the_alphabet_soup_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_bbq_sauce_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_butter_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_cream_cheese_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_ketchup_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_milk_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_orange_juice_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_salad_dressing_and_place_it_in_the_basket",
                    "huihanl/libero-libero_object_no_noops/pick_up_the_tomato_sauce_and_place_it_in_the_basket",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_libero_low_mem_finetune-libero_object_sequential",
                ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=200000,
        steps_per_task=20000,
        save_interval=500,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    
    TrainConfig(
        name="pi0_robocasa_CoffeePressButton_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotRobocasaDataConfig(
            repo_id="robocasa/CoffeePressButton",
            base_config=DataConfig(
                # local_files_only=True,  
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_robocasa_CoffeePressButton_low_mem_finetune",
                asset_id="robocasa/CoffeePressButton",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    TrainConfig(
        name="pi0_robocasa_multitask_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotRobocasaDataConfig(
            repo_id="robocasa/multitask",
            base_config=DataConfig(
                # local_files_only=True,  
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_robocasa_multitask_low_mem_finetune",
                asset_id="robocasa/multitask",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    TrainConfig(
        name="pi0_robocasa_multitask_low_mem_finetune-door",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotRobocasaDataConfig(
            repo_id="robocasa/multitask-door",
            base_config=DataConfig(
                # local_files_only=True,  
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_robocasa_multitask_low_mem_finetune-door",
                asset_id="robocasa/multitask-door",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    
    TrainConfig(
        name="pi0_robocasa_multitask_low_mem_finetune-pnp",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotRobocasaDataConfig(
            repo_id="robocasa/multitask-pnp",
            base_config=DataConfig(
                # local_files_only=True,  
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_robocasa_multitask_low_mem_finetune-pnp",
                asset_id="robocasa/multitask-pnp",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    TrainConfig(
        name="pi0_robocasa_low_mem_finetune-sequential-all",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        n_tasks=24,
        
        data=LeRobotRobocasaSeqDataConfig(
            repo_id="robocasa/single_task",# not used
            repo_ids=(
                        "robocasa/single_task/CloseDoubleDoor",
                        "robocasa/single_task/CloseDrawer",
                        "robocasa/single_task/CloseSingleDoor",
                        "robocasa/single_task/CoffeePressButton",
                        "robocasa/single_task/CoffeeServeMug",
                        "robocasa/single_task/CoffeeSetupMug",
                        "robocasa/single_task/OpenDoubleDoor",
                        "robocasa/single_task/OpenDrawer",
                        "robocasa/single_task/OpenSingleDoor",
                        "robocasa/single_task/PnPCabToCounter",
                        "robocasa/single_task/PnPCounterToCab",
                        "robocasa/single_task/PnPCounterToMicrowave",
                        "robocasa/single_task/PnPCounterToSink",
                        "robocasa/single_task/PnPCounterToStove",
                        "robocasa/single_task/PnPMicrowaveToCounter",
                        "robocasa/single_task/PnPSinkToCounter",
                        "robocasa/single_task/PnPStoveToCounter",
                        "robocasa/single_task/TurnOffMicrowave",
                        "robocasa/single_task/TurnOffSinkFaucet",
                        "robocasa/single_task/TurnOffStove",
                        "robocasa/single_task/TurnOnMicrowave",
                        "robocasa/single_task/TurnOnSinkFaucet",
                        "robocasa/single_task/TurnOnStove",
                        "robocasa/single_task/TurnSinkSpout",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_robocasa_low_mem_finetune-sequential-all",
                ),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
       
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    
    TrainConfig(
        name="pi0_robocasa_low_mem_finetune-sequential-all-randomized",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        n_tasks=24,
        
        data=LeRobotRobocasaSeqDataConfig(
            repo_id="robocasa/single_task",# not used
            repo_ids=(
                        "robocasa/single_task/PnPSinkToCounter",
                        "robocasa/single_task/CoffeeSetupMug",
                        "robocasa/single_task/TurnOffSinkFaucet",
                        "robocasa/single_task/TurnSinkSpout",
                        "robocasa/single_task/PnPCounterToStove",
                        "robocasa/single_task/PnPCounterToCab",
                        "robocasa/single_task/TurnOffMicrowave",
                        "robocasa/single_task/PnPCounterToMicrowave",
                        "robocasa/single_task/TurnOffStove",
                        "robocasa/single_task/PnPStoveToCounter",
                        "robocasa/single_task/CloseDrawer",
                        "robocasa/single_task/PnPCounterToSink",
                        "robocasa/single_task/TurnOnSinkFaucet",
                        "robocasa/single_task/OpenDoubleDoor",
                        "robocasa/single_task/PnPCabToCounter",
                        "robocasa/single_task/CloseSingleDoor",
                        "robocasa/single_task/TurnOnStove",
                        "robocasa/single_task/CoffeeServeMug",
                        "robocasa/single_task/PnPMicrowaveToCounter",
                        "robocasa/single_task/OpenDrawer",
                        "robocasa/single_task/OpenSingleDoor",
                        "robocasa/single_task/CloseDoubleDoor",
                        "robocasa/single_task/CoffeePressButton",
                        "robocasa/single_task/TurnOnMicrowave",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_robocasa_low_mem_finetune-sequential-all",
                ),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
       
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    
    TrainConfig(
        name="pi0_robocasa_low_mem_finetune-sequential-door",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        n_tasks=6,
        
        data=LeRobotRobocasaSeqDataConfig(
            repo_id="robocasa/single_task",# not used
            repo_ids=(
                        "robocasa/single_task/CloseDoubleDoor",
                        "robocasa/single_task/CloseDrawer",
                        "robocasa/single_task/CloseSingleDoor",
                        "robocasa/single_task/OpenDoubleDoor",
                        "robocasa/single_task/OpenDrawer",
                        "robocasa/single_task/OpenSingleDoor",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_robocasa_low_mem_finetune-sequential-all",
                ),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
       
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    
    
    TrainConfig(
        name="pi0_robocasa_low_mem_finetune-sequential-pnp",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        
        n_tasks=8,
        
        data=LeRobotRobocasaSeqDataConfig(
            repo_id="robocasa/single_task",# not used
            repo_ids=(
                        "robocasa/single_task/PnPSinkToCounter",
                        "robocasa/single_task/PnPCounterToMicrowave",
                        "robocasa/single_task/PnPMicrowaveToCounter",
                        "robocasa/single_task/PnPCounterToCab",
                        "robocasa/single_task/PnPStoveToCounter",
                        "robocasa/single_task/PnPCounterToSink",
                        "robocasa/single_task/PnPCabToCounter",
                        "robocasa/single_task/PnPCounterToStove",
                    ),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0_robocasa_low_mem_finetune-sequential-all",
                ),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
       
        num_train_steps=100000,
        steps_per_task=10000,
        save_interval=2000,
        
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
]

_CONFIGS += (
    _seq_variants_for_benchmark("libero_goal_sequential", _LIBERO_GOAL_SEQ)
    + _seq_variants_for_benchmark("libero_spatial_sequential", _LIBERO_SPATIAL_SEQ)
    + _seq_variants_for_benchmark("libero_object_sequential", _LIBERO_OBJECT_SEQ)
    + _seq_variants_for_benchmark("libero_10_sequential", _LIBERO_10_SEQ)
)

_CONFIGS += (
    _multitask_to_i_variants("libero_goal", _LIBERO_GOAL_SEQ)
    + _multitask_to_i_variants("libero_spatial", _LIBERO_SPATIAL_SEQ)
    + _multitask_to_i_variants("libero_object", _LIBERO_OBJECT_SEQ)
    + _multitask_to_i_variants("libero_10", _LIBERO_10_SEQ)
)

_CONFIGS += (
    _robocasa_seq_variants_for_benchmark("all", _ROBOCASA_ALL_SEQ)
    + _robocasa_seq_variants_for_benchmark("door", _ROBOCASA_DOOR_SEQ)
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
