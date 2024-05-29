#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import copy

import torch
import torch.nn as nn

import deeplabcut.pose_estimation_pytorch.modelzoo.utils as modelzoo_utils
from deeplabcut.pose_estimation_pytorch.models.backbones import BaseBackbone, BACKBONES
from deeplabcut.pose_estimation_pytorch.models.criterions import (
    CRITERIONS,
    LOSS_AGGREGATORS,
)
from deeplabcut.pose_estimation_pytorch.models.heads import BaseHead, HEADS
from deeplabcut.pose_estimation_pytorch.models.necks import BaseNeck, NECKS
from deeplabcut.pose_estimation_pytorch.models.predictors import PREDICTORS
from deeplabcut.pose_estimation_pytorch.models.target_generators import (
    TARGET_GENERATORS,
)
from deeplabcut.core.weight_init import WeightInitialization


class PoseModel(nn.Module):
    """A pose estimation model

    A pose estimation model is composed of a backbone, optionally a neck, and an
    arbitrary number of heads. Outputs are computed as follows:
    """

    def __init__(
        self,
        cfg: dict,
        backbone: BaseBackbone,
        heads: dict[str, BaseHead],
        neck: BaseNeck | None = None,
    ) -> None:
        """
        Args:
            cfg: configuration dictionary for the model.
            backbone: backbone network architecture.
            heads: the heads for the model
            neck: neck network architecture (default is None). Defaults to None.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.neck = neck

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass of the PoseModel.

        Args:
            x: input images

        Returns:
            Outputs of head groups
        """
        if x.dim() == 3:
            x = x[None, :]
        features = self.backbone(x)
        if self.neck:
            features = self.neck(features)

        outputs = {}
        for head_name, head in self.heads.items():
            outputs[head_name] = head(features)
        return outputs

    def get_loss(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        total_losses = []
        losses: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            head_losses = head.get_loss(outputs[name], targets[name])
            total_losses.append(head_losses["total_loss"])
            for k, v in head_losses.items():
                losses[f"{name}_{k}"] = v

        # TODO: Different aggregation for multi-head loss?
        losses["total_loss"] = torch.mean(torch.stack(total_losses))
        return losses

    def get_target(
        self,
        inputs: torch.Tensor,
        outputs: dict[str, dict[str, torch.Tensor]],
        labels: dict,
    ) -> dict[str, dict]:
        """Summary:
        Get targets for model training.

        Args:
            inputs: the input images given to the model, of shape (b, c, w, h)
            outputs: output of each head group
            labels: dictionary of labels

        Returns:
            targets: dict of the targets for each model head group
        """
        return {
            name: head.target_generator(inputs, outputs[name], labels)
            for name, head in self.heads.items()
        }

    def get_predictions(
        self, inputs: torch.Tensor, outputs: dict[str, dict[str, torch.Tensor]]
    ) -> dict:
        """Abstract method for the forward pass of the Predictor.

        Args:
            inputs: the input images given to the model, of shape (b, c, w, h)
            outputs: outputs of the model heads

        Returns:
            A dictionary containing the predictions of each head group
        """
        return {
            head_name: head.predictor(inputs, outputs[head_name])
            for head_name, head in self.heads.items()
        }

    @staticmethod
    def build(
        cfg: dict,
        weight_init: None | WeightInitialization = None,
    ) -> "PoseModel":
        """
        Args:
            cfg: The configuration of the model to build.
            weight_init: How model weights should be initialized. If None, ImageNet
                pre-trained backbone weights are loaded from Timm.

        Returns:
            the built pose model
        """
        if weight_init is None:  # Transfer learning from ImageNet
            cfg["backbone"]["pretrained"] = True

        backbone = BACKBONES.build(dict(cfg["backbone"]))

        neck = None
        if cfg.get("neck"):
            neck = NECKS.build(dict(cfg["neck"]))

        heads = {}
        for name, head_cfg in cfg["heads"].items():
            head_cfg = copy.deepcopy(head_cfg)
            if "type" in head_cfg["criterion"]:
                head_cfg["criterion"] = CRITERIONS.build(head_cfg["criterion"])
            else:
                weights = {}
                criterions = {}
                for loss_name, criterion_cfg in head_cfg["criterion"].items():
                    weights[loss_name] = criterion_cfg.get("weight", 1.0)
                    criterion_cfg = {
                        k: v for k, v in criterion_cfg.items() if k != "weight"
                    }
                    criterions[loss_name] = CRITERIONS.build(criterion_cfg)

                aggregator_cfg = {"type": "WeightedLossAggregator", "weights": weights}
                head_cfg["aggregator"] = LOSS_AGGREGATORS.build(aggregator_cfg)
                head_cfg["criterion"] = criterions

            head_cfg["target_generator"] = TARGET_GENERATORS.build(
                head_cfg["target_generator"]
            )
            head_cfg["predictor"] = PREDICTORS.build(head_cfg["predictor"])
            heads[name] = HEADS.build(head_cfg)

        model = PoseModel(cfg=cfg, backbone=backbone, neck=neck, heads=heads)

        if weight_init is not None:
            print(f"Loading pretrained model weights: {weight_init}")

            # TODO: Should we specify the pose_model_type in WeightInitialization?
            backbone_name = cfg["backbone"]["model_name"]
            pose_model_type = modelzoo_utils.get_pose_model_type(backbone_name)

            # load pretrained weights
            _, _, snapshot_path, _ = modelzoo_utils.get_config_model_paths(
                project_name=weight_init.dataset,
                pose_model_type=pose_model_type,
            )
            snapshot = torch.load(snapshot_path, map_location="cpu")
            state_dict = snapshot["model"]

            # load backbone state dict
            model.backbone.load_state_dict(filter_state_dict(state_dict, "backbone"))

            # if there's a neck, load state dict
            if model.neck is not None:
                model.neck.load_state_dict(filter_state_dict(state_dict, "neck"))

            # load head state dicts
            if weight_init.with_decoder:
                heads_state_dict = filter_state_dict(state_dict, "heads")
                conversion_tensor = torch.from_numpy(weight_init.conversion_array)
                for name, head in model.heads.items():
                    # requires WeightConversionMixin
                    head.load_state_dict(
                        head.convert_weights(
                            state_dict=filter_state_dict(heads_state_dict, name),
                            module_prefix="",
                            conversion=conversion_tensor,
                        )
                    )

        return model


def filter_state_dict(state_dict: dict, module: str) -> dict[str, torch.Tensor]:
    """
    Filters keys in the state dict for a module to only keep a given prefix. Removes
    the module from the keys (e.g. for module="backbone", "backbone.stage1.weight" will
    be converted to "stage1.weight" so the state dict can be loaded into the backbone
    directly).

    Args:
        state_dict: the state dict
        module: the module to keep, e.g. "backbone"

    Returns:
        the filtered state dict, with the module removed from the keys

    Examples:
        state_dict = {"backbone.conv.weight": t1, "head.conv.weight": t2}
        filtered = filter_state_dict(state_dict, "backbone")
        # filtered = {"conv.weight": t1}
        model.backbone.load_state_dict(filtered)
    """
    return {
        ".".join(k.split(".")[1:]): v  # remove 'backbone.' from the keys
        for k, v in state_dict.items()
        if k.startswith(module)
    }
