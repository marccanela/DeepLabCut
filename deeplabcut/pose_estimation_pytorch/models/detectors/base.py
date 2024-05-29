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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import deeplabcut.pose_estimation_pytorch.modelzoo.utils as modelzoo_utils
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg


def _build_detector(
    cfg: dict, weight_init: WeightInitialization | None = None, **kwargs,
) -> BaseDetector:
    """Builds a detector using its configuration file

    Args:
        cfg: The detector configuration.
        weight_init: The weight initialization to use.
        **kwargs: Other parameters given by the Registry.

    Returns:
        the built detector
    """
    if weight_init is not None:
        cfg["pretrained"] = False

    detector: BaseDetector = build_from_cfg(cfg, **kwargs)

    if weight_init is not None:
        _, _, _, snapshot_path = modelzoo_utils.get_config_model_paths(
            project_name=weight_init.dataset,
            pose_model_type="hrnetw32",  # pose model does not matter here
            detector_type="fasterrcnn",  # TODO: include variant
        )
        snapshot = torch.load(snapshot_path, map_location="cpu")
        detector.load_state_dict(snapshot["model"])

    return detector


DETECTORS = Registry("detectors", build_func=_build_detector)


class BaseDetector(ABC, nn.Module):
    """
    Definition of the class BaseDetector object.
    This is an abstract class defining the common structure and inference for detectors.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self._pretrained = pretrained

    @abstractmethod
    def forward(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """
        Forward pass of the detector

        Args:
            x: images to be processed
            targets: ground-truth boxes present in each images

        Returns:
            losses: {'loss_name': loss_value}
            detections: for each of the b images, {"boxes": bounding_boxes}
        """
        pass

    @abstractmethod
    def get_target(self, labels: dict) -> list[dict]:
        """
        Get the target for training the detector

        Args:
            labels: annotations containing keypoints, bounding boxes, etc.

        Returns:
            list of dictionaries, each representing target information for a single annotation.
        """
        pass
