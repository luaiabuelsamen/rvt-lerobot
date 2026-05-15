"""Interactive 3D viewer for the RVT pipeline using `rerun`.

Logs:
  - Fused point cloud in world frame (from 4 real RGBD cams).
  - 4 real camera frusta with their RGB images attached as image planes.
  - 5 virtual orthographic cameras placed around the workspace center, with
    each reprojected view attached as the camera's image plane.
  - Gripper pose as an axis-aligned transform.

Run:
    PYTHONPATH=. python -m rvt_lerobot.visualize.rerun_viewer \
        --episode data/demos/.../episode0 --frame 0

This will spawn the rerun viewer in a window. Add `--save out.rrd` to write
a recording you can open later with `rerun out.rrd`.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image

from rvt_lerobot.visualize.virtual_views import (
    decode_depth_png,
    unproject,
    ortho_render,
    CAMERAS_DEFAULT,
)


VIRTUAL_VIEWS = ("front", "top", "left", "right", "back")


def virtual_camera_pose(view: str, center: np.ndarray, dist: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (cam_pos_world, R_cam2world) for the 5 ortho virtual views.

    Camera convention: OpenCV (+z forward, +x right, +y down looking through the lens).
    """
    if view == "front":   forward = np.array([0, 1, 0]);  up = np.array([0, 0, 1])
    elif view == "back":  forward = np.array([0, -1, 0]); up = np.array([0, 0, 1])
    elif view == "left":  forward = np.array([1, 0, 0]);  up = np.array([0, 0, 1])
    elif view == "right": forward = np.array([-1, 0, 0]); up = np.array([0, 0, 1])
    elif view == "top":   forward = np.array([0, 0, -1]); up = np.array([0, 1, 0])
    else: raise ValueError(view)
    forward = forward.astype(np.float32)
    up = up.astype(np.float32)
    pos = center - forward * dist
    z = forward
    x = np.cross(up, z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, -y, z], axis=1).astype(np.float32)  # cam2world; OpenCV: +y is down
    return pos.astype(np.float32), R


def log_real_camera(name: str, rgb: np.ndarray, K: np.ndarray, T_cam2world: np.ndarray) -> None:
    rr.log(
        f"world/cams/real/{name}",
        rr.Transform3D(translation=T_cam2world[:3, 3], mat3x3=T_cam2world[:3, :3]),
    )
    H, W = rgb.shape[:2]
    rr.log(
        f"world/cams/real/{name}/image",
        rr.Pinhole(image_from_camera=K, width=W, height=H),
    )
    rr.log(f"world/cams/real/{name}/image", rr.Image(rgb))


def log_virtual_camera(name: str, img: np.ndarray, pos: np.ndarray, R: np.ndarray,
                        extent: float, img_size: int) -> None:
    f = img_size / extent  # makes the image plane at z=1 match the workspace extent
    K = np.array([[f, 0, img_size / 2],
                  [0, f, img_size / 2],
                  [0, 0, 1.0]], dtype=np.float32)
    rr.log(
        f"world/cams/virtual/{name}",
        rr.Transform3D(translation=pos, mat3x3=R),
    )
    rr.log(
        f"world/cams/virtual/{name}/image",
        rr.Pinhole(image_from_camera=K, width=img_size, height=img_size),
    )
    rr.log(f"world/cams/virtual/{name}/image", rr.Image(img))


def view_episode(
    episode_dir: str | Path,
    frame_idx: int = 0,
    cameras: tuple[str, ...] = CAMERAS_DEFAULT,
    workspace_center: tuple[float, float, float] = (0.0, -0.25, 0.15),
    workspace_extent: float = 0.6,
    virtual_image_size: int = 220,
    application_id: str = "rvt_lerobot",
    save_to: str | None = None,
) -> None:
    ep = Path(episode_dir)
    with open(ep / "low_dim_obs.pkl", "rb") as f:
        obs_list = pickle.load(f)
    obs = obs_list[frame_idx]

    if save_to is None:
        rr.init(application_id, spawn=True)
    else:
        rr.init(application_id)
        rr.save(save_to)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Real cameras: log image planes + accumulate fused PCD.
    pts_all, cols_all = [], []
    for cam in cameras:
        rgb = np.array(Image.open(ep / f"{cam}_rgb" / f"{frame_idx}.png"))
        near = float(obs.misc[f"{cam}_camera_near"])
        far = float(obs.misc[f"{cam}_camera_far"])
        depth = decode_depth_png(ep / f"{cam}_depth" / f"{frame_idx}.png", near, far)
        K = obs.misc[f"{cam}_camera_intrinsics"]
        T = obs.misc[f"{cam}_camera_extrinsics"]
        log_real_camera(cam, rgb, K, T)
        p, c = unproject(rgb, depth, K, T)
        pts_all.append(p); cols_all.append(c)
    points = np.concatenate(pts_all, axis=0)
    colors = np.concatenate(cols_all, axis=0)

    rr.log("world/fused_pcd", rr.Points3D(points, colors=colors, radii=0.002))
    print(f"fused PCD: {points.shape[0]:,} points")

    # Virtual cameras: re-render the PCD from each, log image plane + frustum.
    center = np.array(workspace_center, dtype=np.float32)
    dist = workspace_extent  # camera placed `dist` from center along view direction
    for v in VIRTUAL_VIEWS:
        img = ortho_render(points, colors, v, center=center,
                           extent=workspace_extent, img_size=virtual_image_size)
        pos, R = virtual_camera_pose(v, center, dist)
        log_virtual_camera(v, img, pos, R, workspace_extent, virtual_image_size)

    # Gripper as an axis frame.
    gp = obs.gripper_pose
    pos = gp[:3]
    qw, qx, qy, qz = gp[3], gp[4], gp[5], gp[6]
    R_g = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)
    rr.log("world/gripper", rr.Transform3D(translation=pos, mat3x3=R_g, axis_length=0.05))

    # Helpful annotations
    rr.log("world/workspace_center", rr.Points3D([center], colors=[[255, 255, 0]], radii=0.01))


def view_episode_animated(
    episode_dir: str | Path,
    cameras: tuple[str, ...] = CAMERAS_DEFAULT,
    workspace_center: tuple[float, float, float] = (0.0, -0.25, 0.15),
    workspace_extent: float = 0.6,
    virtual_image_size: int = 220,
    stride: int = 5,
    application_id: str = "rvt_lerobot",
    save_to: str | None = None,
) -> None:
    """Step through every Nth frame of the episode, with rerun's timeline."""
    ep = Path(episode_dir)
    with open(ep / "low_dim_obs.pkl", "rb") as f:
        obs_list = pickle.load(f)

    if save_to is None:
        rr.init(application_id, spawn=True)
    else:
        rr.init(application_id)
        rr.save(save_to)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for t in range(0, len(obs_list), stride):
        rr.set_time_sequence("frame", t)
        obs = obs_list[t]
        pts_all, cols_all = [], []
        for cam in cameras:
            rgb = np.array(Image.open(ep / f"{cam}_rgb" / f"{t}.png"))
            near = float(obs.misc[f"{cam}_camera_near"])
            far = float(obs.misc[f"{cam}_camera_far"])
            depth = decode_depth_png(ep / f"{cam}_depth" / f"{t}.png", near, far)
            K = obs.misc[f"{cam}_camera_intrinsics"]
            T = obs.misc[f"{cam}_camera_extrinsics"]
            log_real_camera(cam, rgb, K, T)
            p, c = unproject(rgb, depth, K, T)
            pts_all.append(p); cols_all.append(c)
        points = np.concatenate(pts_all, axis=0)
        colors = np.concatenate(cols_all, axis=0)
        rr.log("world/fused_pcd", rr.Points3D(points, colors=colors, radii=0.002))

        center = np.array(workspace_center, dtype=np.float32)
        for v in VIRTUAL_VIEWS:
            img = ortho_render(points, colors, v, center=center,
                               extent=workspace_extent, img_size=virtual_image_size)
            pos, R = virtual_camera_pose(v, center, workspace_extent)
            log_virtual_camera(v, img, pos, R, workspace_extent, virtual_image_size)
    print(f"logged {len(obs_list) // stride} frames")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--animate", action="store_true", help="step through whole episode")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--save", default=None, help="write .rrd recording instead of spawning viewer")
    args = p.parse_args()
    if args.animate:
        view_episode_animated(args.episode, stride=args.stride, save_to=args.save)
    else:
        view_episode(args.episode, frame_idx=args.frame, save_to=args.save)


if __name__ == "__main__":
    main()
