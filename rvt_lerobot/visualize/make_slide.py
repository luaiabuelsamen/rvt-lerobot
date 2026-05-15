"""Combined 16:9 slide: 4 real RGBD cams (top) -> fused PCD (arrow) -> 5 virtual views (bottom).

The point: show the *whole* RVT input pipeline on one slide.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from PIL import Image

from rvt_lerobot.visualize.virtual_views import (
    decode_depth_png, unproject, ortho_render, CAMERAS_DEFAULT,
)

VIRTUAL_VIEWS = ("front", "top", "left", "right", "back")


def build_slide(
    episode_dir: str | Path,
    frame_idx: int = 0,
    out: str | Path = "rvt_pipeline_slide.png",
    workspace_center=(0.0, -0.25, 0.15),
    workspace_extent: float = 0.6,
    virtual_image_size: int = 360,
) -> None:
    ep = Path(episode_dir)
    with open(ep / "low_dim_obs.pkl", "rb") as f:
        obs_list = pickle.load(f)
    obs = obs_list[frame_idx]

    real_imgs, real_depths = {}, {}
    pts_all, cols_all = [], []
    for cam in CAMERAS_DEFAULT:
        rgb = np.array(Image.open(ep / f"{cam}_rgb" / f"{frame_idx}.png"))
        near = float(obs.misc[f"{cam}_camera_near"])
        far = float(obs.misc[f"{cam}_camera_far"])
        depth = decode_depth_png(ep / f"{cam}_depth" / f"{frame_idx}.png", near, far)
        K = obs.misc[f"{cam}_camera_intrinsics"]
        T = obs.misc[f"{cam}_camera_extrinsics"]
        real_imgs[cam] = rgb
        real_depths[cam] = depth
        p, c = unproject(rgb, depth, K, T)
        pts_all.append(p); cols_all.append(c)
    points = np.concatenate(pts_all, axis=0)
    colors = np.concatenate(cols_all, axis=0)

    virtual_imgs = {
        v: ortho_render(points, colors, v,
                        center=np.array(workspace_center, dtype=np.float32),
                        extent=workspace_extent, img_size=virtual_image_size)
        for v in VIRTUAL_VIEWS
    }

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.suptitle(
        "RVT pipeline:  4 real RGBD cameras  →  fused 3D point cloud  →  5 orthographic virtual views",
        fontsize=15, fontweight="bold", y=0.965,
    )
    sub = fig.text(0.5, 0.93, f"frame {frame_idx} · {points.shape[0]:,} fused points · "
                              "projection is *not* learned — the transformer always sees the same canonical viewpoints",
                   ha="center", fontsize=10, style="italic", color="#444")

    gs = GridSpec(
        nrows=3, ncols=5,
        height_ratios=[1.0, 0.18, 1.0],
        hspace=0.20, wspace=0.10,
        left=0.03, right=0.97, top=0.88, bottom=0.04,
        figure=fig,
    )

    # Top row: 4 real cameras, last col reserved for the pipeline arrow
    real_titles = {"front": "front", "left_shoulder": "left shoulder",
                   "right_shoulder": "right shoulder", "wrist": "wrist"}
    for i, cam in enumerate(CAMERAS_DEFAULT):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(real_imgs[cam])
        ax.set_title(real_titles[cam], fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor("#3a7"); s.set_linewidth(1.5)

    ax_pcd = fig.add_subplot(gs[0, 4])
    ax_pcd.axis("off")
    ax_pcd.text(0.5, 0.85, "REAL RGBD", ha="center", fontsize=11, fontweight="bold", color="#3a7",
                transform=ax_pcd.transAxes)
    ax_pcd.text(0.5, 0.55, "unproject\n+ fuse", ha="center", fontsize=12,
                transform=ax_pcd.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", fc="#eef6ff", ec="#3978c4", lw=1.2))
    ax_pcd.annotate("", xy=(0.5, 0.10), xytext=(0.5, 0.40),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=2.4, color="#3978c4"))
    ax_pcd.text(0.5, 0.02, f"{points.shape[0]:,} pts", ha="center", fontsize=9, color="#3978c4",
                transform=ax_pcd.transAxes)

    # Middle band: full-width pipeline arrow / connector
    ax_arrow = fig.add_subplot(gs[1, :])
    ax_arrow.axis("off")
    ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
    for x in np.linspace(0.06, 0.94, 5):
        ax_arrow.annotate("", xy=(x, 0.2), xytext=(x, 0.85),
                          arrowprops=dict(arrowstyle="->", lw=1.6, color="#3978c4", alpha=0.55))
    ax_arrow.text(0.5, 0.55, "re-render PCD from 5 fixed orthographic virtual cameras",
                  ha="center", fontsize=11, color="#1b4f8a", fontweight="bold")

    # Bottom row: 5 virtual orthographic views
    for i, v in enumerate(VIRTUAL_VIEWS):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(virtual_imgs[v])
        ax.set_title(f"virtual: {v}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor("#c43"); s.set_linewidth(1.5)

    fig.text(0.025, 0.475, "VIRTUAL  ORTHOGRAPHIC", fontsize=11, fontweight="bold",
             color="#c43", rotation=90, va="center")
    fig.text(0.025, 0.71, "REAL  RGBD", fontsize=11, fontweight="bold",
             color="#3a7", rotation=90, va="center")

    fig.text(0.5, 0.005,
             "the multi-view transformer attends across the bottom row only — the projection step is geometric, not learned",
             ha="center", fontsize=9.5, color="#555", style="italic")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out.resolve()}  ({out.stat().st_size // 1024} KB)")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", default="assets/rvt_pipeline_slide.png")
    args = p.parse_args()
    build_slide(args.episode, args.frame, args.out)


if __name__ == "__main__":
    main()
