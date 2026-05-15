"""RVT wrapped as a LeRobot-compatible policy.

This stub gives the *shape* of a LeRobot integration. The two integration
edges are clearly marked with TODO so the contribution path to LeRobot is
unambiguous:

  1. Config: subclass `PreTrainedConfig` and register under the name "rvt".
  2. Policy: implement `forward()` (training loss) and `select_action()` (inference)
     by delegating to RVT's `RVTAgent` from external/RVT.

To avoid a hard dep on LeRobot at import time (useful for the data-pipeline-only
flow), the LeRobot subclassing happens behind a try/except — the same module
can be imported without LeRobot installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class RVTConfigStub:
    """Minimal config mirroring what a LeRobot `PreTrainedConfig` subclass would hold.

    To make this a real LeRobot config, do:
        from lerobot.configs.policies import PreTrainedConfig
        @PreTrainedConfig.register_subclass("rvt")
        class RVTConfig(PreTrainedConfig): ...
    and add `input_features`, `output_features`, and processor wiring.
    """
    image_size: int = 128
    cameras: tuple[str, ...] = ("front", "left_shoulder", "right_shoulder", "wrist")
    num_virtual_views: int = 5
    scene_bounds: tuple[float, float, float, float, float, float] = (
        -0.35, -0.55, 0.05, 0.35, 0.05, 0.55,
    )
    voxel_size: int = 100
    rotation_resolution: int = 5
    n_low_dim: int = 4
    use_clip: bool = True
    clip_model_name: str = "RN50"
    learning_rate: float = 1e-4

    @property
    def observation_delta_indices(self) -> list[int] | None: return None

    @property
    def action_delta_indices(self) -> list[int] | None: return None

    @property
    def reward_delta_indices(self) -> list[int] | None: return None


class RVTPolicyStub(nn.Module):
    """LeRobot-style wrapper around RVT's `RVTAgent`.

    The intended drop-in path:
        class RVTPolicy(PreTrainedPolicy):
            config_class = RVTConfig
            name = "rvt"
            def __init__(self, config, dataset_stats=None):
                super().__init__(config)
                self.agent = build_rvt_agent(config)   # from external/RVT
            def forward(self, batch): ...
            def select_action(self, batch): ...
    """
    name = "rvt"
    config_class = RVTConfigStub

    def __init__(self, config: RVTConfigStub | None = None):
        super().__init__()
        self.config = config or RVTConfigStub()
        # TODO(lerobot-integration): swap in the real RVT agent. Import path:
        #   from rvt.models.rvt_agent import RVTAgent
        #   from rvt.config import get_default_cfg
        # Then build with mvt = MultiViewTransformer(...) and RVTAgent(mvt, ...).
        self._agent = None
        self._action_queue: list[Tensor] = []

    # -- LeRobot PreTrainedPolicy surface -------------------------------------

    def reset(self) -> None:
        self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict the next end-effector keypose given a multi-camera RGBD batch.

        Expected batch keys (per camera in self.config.cameras):
            "observation.images.{cam}":  (B, 3, H, W) float in [0,1]
            "observation.depths.{cam}":  (B, H, W)    float (meters)
            "observation.cam_extrinsics.{cam}": (B, 4, 4)
            "observation.cam_intrinsics.{cam}": (B, 3, 3)
        Plus:
            "observation.low_dim_state": (B, n_low_dim)
            "task":                      list[str] of length B (language goal)

        Returns:
            action: (B, A) where A encodes [xyz_target, rot6d, gripper_open].
        """
        if self._agent is None:
            B = next(iter(batch.values())).shape[0]
            # Stub: zero action; replace with `self._agent.act(...)`.
            return torch.zeros(B, 8, device=next(iter(batch.values())).device)
        raise NotImplementedError("Wire `self._agent.act(observation, ...)` here.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any] | None]:
        """Training step: returns (loss, info)."""
        if self._agent is None:
            zeros = torch.zeros((), device=next(iter(batch.values())).device, requires_grad=True)
            return zeros, {"stub": True}
        raise NotImplementedError("Wire `self._agent.update(...)` here.")


def try_register_lerobot_policy() -> bool:
    """Best-effort registration with LeRobot's policy factory.

    Returns True if registered, False if LeRobot isn't importable.
    """
    try:
        from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig    # noqa: F401
    except ImportError:
        return False
    # TODO: implement once LeRobot is the runtime target. See class docstring.
    return False
