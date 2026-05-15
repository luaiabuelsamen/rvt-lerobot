"""Build the RVT-style virtual orthographic views from a fused point cloud.

Pipeline:
  1) Load one stored frame: 4x (rgb, depth, K, T_cam_to_world).
  2) Unproject each cam to a 3D point cloud, carrying RGB.
  3) Concatenate -> one workspace point cloud.
  4) Re-render from K canonical orthographic cameras (front, top, left, right, back)
     with a simple z-buffered nearest-point splat (no PyTorch3D needed).

This is exactly the trick the multi-view transformer in RVT operates on: the
projection is geometric and *not* learned. The transformer just attends across
these always-the-same views.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from rvt_lerobot.data.rlbench_format import DEPTH_SCALE

CAMERAS_DEFAULT = ("front", "left_shoulder", "right_shoulder", "wrist")


def decode_depth_png(path: Path, near: float, far: float) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint32)
    coded = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
    norm = coded.astype(np.float32) / DEPTH_SCALE
    return norm * (far - near) + near


def unproject(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, T_cam2world: np.ndarray,
              z_clip: tuple[float, float] = (0.05, 2.5)) -> tuple[np.ndarray, np.ndarray]:
    """Return (Nx3 points in world, Nx3 colors uint8)."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    valid = (z > z_clip[0]) & (z < z_clip[1])
    x_c = (us - cx) * z / fx
    y_c = (vs - cy) * z / fy
    z_c = z
    pts_cam = np.stack([x_c, y_c, z_c], axis=-1)[valid]
    cols = rgb[valid]
    ones = np.ones((pts_cam.shape[0], 1), dtype=pts_cam.dtype)
    pts_world = (T_cam2world @ np.concatenate([pts_cam, ones], axis=1).T).T[:, :3]
    return pts_world.astype(np.float32), cols.astype(np.uint8)


def ortho_render(
    points: np.ndarray,
    colors: np.ndarray,
    view: str,
    center: np.ndarray = np.array([0.0, -0.25, 0.15]),
    extent: float = 0.5,
    img_size: int = 220,
    bg: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    """Z-buffered orthographic splat of a colored point cloud.

    `view` in {front, back, top, left, right}. Defines a workspace-aligned
    orthographic camera pointed at `center` with a `extent` x `extent` window.
    """
    p = points - center

    def proj(axis_u, axis_v, depth_axis, flip_depth=False):
        u = p[:, axis_u]
        v = p[:, axis_v]
        d = p[:, depth_axis] * (-1.0 if flip_depth else 1.0)
        return u, v, d

    if view == "front":   u, v, d = proj(0, 2, 1, flip_depth=False)
    elif view == "back":  u, v, d = proj(0, 2, 1, flip_depth=True);  u = -u
    elif view == "top":   u, v, d = proj(0, 1, 2, flip_depth=True)
    elif view == "left":  u, v, d = proj(1, 2, 0, flip_depth=True);  u = -u
    elif view == "right": u, v, d = proj(1, 2, 0, flip_depth=False)
    else: raise ValueError(view)

    half = extent / 2.0
    keep = (np.abs(u) <= half) & (np.abs(v) <= half)
    u, v, d, col = u[keep], v[keep], d[keep], colors[keep]

    px = ((u + half) / extent * (img_size - 1)).astype(np.int32)
    py = ((half - v) / extent * (img_size - 1)).astype(np.int32)  # flip y so +z is up

    img = np.full((img_size, img_size, 3), bg, dtype=np.uint8)
    zbuf = np.full((img_size, img_size), np.inf, dtype=np.float32)

    order = np.argsort(d)
    px, py, d, col = px[order], py[order], d[order], col[order]
    mask = d < zbuf[py, px]
    zbuf[py[mask], px[mask]] = d[mask]
    img[py[mask], px[mask]] = col[mask]

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        py2 = np.clip(py + dy, 0, img_size - 1)
        px2 = np.clip(px + dx, 0, img_size - 1)
        m2 = d < zbuf[py2, px2]
        zbuf[py2[m2], px2[m2]] = d[m2]
        img[py2[m2], px2[m2]] = col[m2]
    return img


def render_episode_frame(
    episode_dir: str | Path,
    frame_idx: int = 0,
    cameras: tuple[str, ...] = CAMERAS_DEFAULT,
    out_path: str | Path = "virtual_views.png",
    workspace_center: tuple[float, float, float] = (0.0, -0.25, 0.15),
    workspace_extent: float = 0.6,
) -> None:
    ep = Path(episode_dir)
    with open(ep / "low_dim_obs.pkl", "rb") as f:
        obs_list = pickle.load(f)
    obs = obs_list[frame_idx]

    pts_all, cols_all = [], []
    for cam in cameras:
        rgb = np.array(Image.open(ep / f"{cam}_rgb" / f"{frame_idx}.png"))
        near = obs.misc[f"{cam}_camera_near"]
        far = obs.misc[f"{cam}_camera_far"]
        depth = decode_depth_png(ep / f"{cam}_depth" / f"{frame_idx}.png", near, far)
        K = obs.misc[f"{cam}_camera_intrinsics"]
        T = obs.misc[f"{cam}_camera_extrinsics"]
        p, c = unproject(rgb, depth, K, T)
        pts_all.append(p); cols_all.append(c)
    points = np.concatenate(pts_all, axis=0)
    colors = np.concatenate(cols_all, axis=0)
    print(f"fused PCD: {points.shape[0]} points from {len(cameras)} cams")

    virtual_views = ["front", "top", "left", "right", "back"]
    imgs = {
        v: ortho_render(points, colors, v,
                        center=np.array(workspace_center, dtype=np.float32),
                        extent=workspace_extent)
        for v in virtual_views
    }

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    for i, cam in enumerate(cameras):
        rgb = np.array(Image.open(ep / f"{cam}_rgb" / f"{frame_idx}.png"))
        axes[0, i].imshow(rgb); axes[0, i].set_title(f"real: {cam}", fontsize=10)
        axes[0, i].axis("off")
    axes[0, 4].axis("off")
    axes[0, 4].text(0.5, 0.5, "fused PCD\n↓\nreproject", ha="center", va="center",
                    fontsize=13, transform=axes[0, 4].transAxes)

    for i, v in enumerate(virtual_views):
        axes[1, i].imshow(imgs[v]); axes[1, i].set_title(f"virtual: {v}", fontsize=10)
        axes[1, i].axis("off")
    plt.suptitle("RVT multi-view transformer input: 4 real RGBD cams → fused PCD → 5 orthographic virtual views",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True, help="path to episodeN dir")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", default="virtual_views.png")
    args = p.parse_args()
    render_episode_frame(args.episode, args.frame, out_path=args.out)


if __name__ == "__main__":
    main()
