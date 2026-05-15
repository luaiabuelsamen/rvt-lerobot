"""MuJoCo SO-ARM100 env with 4 RLBench-style RGBD cameras.

Produces per-step observations in the same per-camera dict layout RVT/PerAct
consume after going through `peract_colab.rlbench.utils.get_stored_demo`:

    obs = {
        "front_rgb":      (H, W, 3) uint8,
        "front_depth":    (H, W)    float32 (meters),
        "front_extrinsics": (4, 4)  float32 (cam-to-world),
        "front_intrinsics": (3, 3)  float32,
        "front_near":     float, "front_far": float,
        ... same for left_shoulder, right_shoulder, wrist ...
        "joint_positions": (n_dof,) float32,
        "gripper_open":   float in {0., 1.},
        "gripper_pose":   (7,)  float32 [x,y,z, qw,qx,qy,qz],
        "low_dim_state":  (4,)  float32 [l_finger, r_finger, gripper_open, t/T]
    }

The fused PCD is *not* materialized here — RVT does it on the fly from depth +
extrinsics. We only emit raw RGBD + calibration.
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Optional

import numpy as np
import gymnasium as gym
import mujoco

CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
IMAGE_SIZE = 128
NEAR = 0.01
FAR = 3.0


def _mj_quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo stores quaternions as (w, x, y, z). Return 3x3 rotation."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class SoArmRVTEnv(gym.Env):
    """SO-ARM100 with 4 RGBD cameras for RVT/PerAct-style data collection."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: str | os.PathLike = None,
        image_size: int = IMAGE_SIZE,
        frame_skip: int = 10,
        max_episode_steps: int = 200,
        cameras: tuple = CAMERAS,
        near: float = NEAR,
        far: float = FAR,
    ):
        super().__init__()
        if model_path is None:
            here = pathlib.Path(__file__).resolve().parent.parent.parent
            model_path = here / "assets" / "so_arm_scene" / "scene_rvt.xml"
        model_path = pathlib.Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.image_size = int(image_size)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.cameras = tuple(cameras)
        self.near = float(near)
        self.far = float(far)
        self._step = 0

        self._cam_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in self.cameras
        }
        missing = [n for n, i in self._cam_ids.items() if i < 0]
        if missing:
            raise ValueError(f"Camera(s) not found in MJCF: {missing}")

        self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._depth_renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._depth_renderer.enable_depth_rendering()

        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_space = gym.spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict({
            **{
                f"{c}_rgb": gym.spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                for c in self.cameras
            },
            **{
                f"{c}_depth": gym.spaces.Box(0.0, far, (image_size, image_size), dtype=np.float32)
                for c in self.cameras
            },
            "low_dim_state": gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        })

        self._home_qpos = np.zeros(self.model.nq, dtype=np.float32)
        self._home_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for k in range(self.model.nkey):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, k)
            if name == "home":
                self._home_qpos = self.model.key_qpos[k].copy()
                if self.model.nu:
                    self._home_ctrl = self.model.key_ctrl[k].copy()
                break

    # -- gym API --------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._home_ctrl
        self._randomize_block()
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step += 1

        obs = self._get_obs()
        terminated = bool(self._block_in_box())
        truncated = self._step >= self.max_episode_steps
        info = {
            "success": terminated,
            "block_pos": self._block_pos(),
            "box_pos": self._box_pos(),
            "gripper_pos": self._gripper_pos(),
        }
        return obs, 0.0, terminated, truncated, info

    # -- RVT-flavored observation --------------------------------------------

    def _get_obs(self) -> dict:
        obs: dict[str, Any] = {}
        for cam in self.cameras:
            rgb, depth = self._render_camera(cam)
            K = self._intrinsics_matrix(cam)
            T = self._extrinsics_matrix(cam)
            obs[f"{cam}_rgb"] = rgb
            obs[f"{cam}_depth"] = depth
            obs[f"{cam}_intrinsics"] = K.astype(np.float32)
            obs[f"{cam}_extrinsics"] = T.astype(np.float32)
            obs[f"{cam}_near"] = float(self.near)
            obs[f"{cam}_far"] = float(self.far)
        gripper_open = self._gripper_open()
        obs["joint_positions"] = self.data.qpos[: self.model.nu].copy().astype(np.float32)
        obs["gripper_open"] = float(gripper_open)
        obs["gripper_pose"] = self._gripper_pose().astype(np.float32)
        obs["low_dim_state"] = np.array(
            [0.0, 0.0, gripper_open, self._step / max(1, self.max_episode_steps)],
            dtype=np.float32,
        )
        return obs

    def _render_camera(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        cam_id = self._cam_ids[name]
        self._renderer.update_scene(self.data, camera=cam_id)
        rgb = self._renderer.render().copy()
        self._depth_renderer.update_scene(self.data, camera=cam_id)
        depth = self._depth_renderer.render().copy().astype(np.float32)
        depth = np.clip(depth, self.near, self.far)
        return rgb, depth

    def _intrinsics_matrix(self, name: str) -> np.ndarray:
        """Pinhole intrinsics from MuJoCo's fovy."""
        cam_id = self._cam_ids[name]
        fovy_deg = float(self.model.cam_fovy[cam_id])
        H = W = self.image_size
        f = 0.5 * H / np.tan(np.deg2rad(fovy_deg) * 0.5)
        return np.array([[f, 0, W / 2 - 0.5],
                         [0, f, H / 2 - 0.5],
                         [0, 0, 1.0]], dtype=np.float64)

    def _extrinsics_matrix(self, name: str) -> np.ndarray:
        """Camera-to-world 4x4 in OpenCV convention (z forward, y down)."""
        cam_id = self._cam_ids[name]
        pos = self.data.cam_xpos[cam_id].copy()
        R_mj = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        # MuJoCo camera: -z forward, +y up. Convert to OpenCV: +z forward, -y up.
        flip = np.diag([1.0, -1.0, -1.0])
        R_cv = R_mj @ flip
        T = np.eye(4)
        T[:3, :3] = R_cv
        T[:3, 3] = pos
        return T

    # -- helpers --------------------------------------------------------------

    def _randomize_block(self):
        block_x = float(self.np_random.uniform(-0.12, 0.12))
        block_y = float(self.np_random.uniform(-0.32, -0.20))
        block_z = 0.115
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block")
        if jid < 0:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
            jadr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
        else:
            jadr = self.model.jnt_qposadr[jid]
        self.data.qpos[jadr:jadr + 3] = [block_x, block_y, block_z]
        self.data.qpos[jadr + 3:jadr + 7] = [1, 0, 0, 0]

    def _body_xpos(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy() if bid >= 0 else np.zeros(3)

    def _block_pos(self):
        return self._body_xpos("block")

    def _box_pos(self):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "box_center")
        return self.data.site_xpos[sid].copy() if sid >= 0 else np.array([-0.1, -0.35, 0.125])

    def _gripper_pos(self):
        return self._body_xpos("Fixed_Jaw")

    def _gripper_pose(self) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        pos = self.data.xpos[bid].copy()
        quat = self.data.xquat[bid].copy()  # (w, x, y, z)
        return np.concatenate([pos, quat])

    def _gripper_open(self) -> float:
        """Heuristic 0/1 from last actuator (jaw)."""
        if self.model.nu == 0:
            return 1.0
        ctrl = float(self.data.ctrl[-1])
        lo, hi = float(self.model.actuator_ctrlrange[-1, 0]), float(self.model.actuator_ctrlrange[-1, 1])
        return float((ctrl - lo) / max(1e-6, hi - lo) > 0.5)

    def _block_in_box(self) -> bool:
        bp = self._block_pos()
        xp = self._box_pos()
        return (abs(bp[0] - xp[0]) < 0.05
                and abs(bp[1] - xp[1]) < 0.05
                and abs(bp[2] - 0.115) < 0.03)

    def close(self):
        pass
