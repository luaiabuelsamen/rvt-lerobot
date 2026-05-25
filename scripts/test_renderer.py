"""Validate the torch renderer against the NumPy oracle (visualize/virtual_views.py).

Run:
    PYTHONPATH=. python scripts/test_renderer.py \
        --episode data/demos/train/put_block_in_box/all_variations/episodes/episode0

Checks:
  1. Foreground agreement + color match vs the NumPy ortho_render oracle on a real frame.
  2. gradcheck on soft mode (differentiable w.r.t. per-point features).
  3. CUDA parity (if available) — Jetson sm_87.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import torch
from PIL import Image

from rvt_lerobot.visualize.virtual_views import unproject, decode_depth_png, ortho_render
from rvt_lerobot.policy.renderer import render_view, VIRTUAL_VIEWS, DEFAULT_CENTER

CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
IMG_SIZE = 220
EXTENT = 0.6
CENTER = np.array(DEFAULT_CENTER, dtype=np.float32)


def fuse_frame(ep: Path, frame: int):
    pts_all, cols_all = [], []
    with open(ep / "low_dim_obs.pkl", "rb") as f:
        obs = pickle.load(f)[frame]
    for cam in CAMERAS:
        rgb = np.array(Image.open(ep / f"{cam}_rgb" / f"{frame}.png"))
        near, far = obs.misc[f"{cam}_camera_near"], obs.misc[f"{cam}_camera_far"]
        depth = decode_depth_png(ep / f"{cam}_depth" / f"{frame}.png", near, far)
        K, T = obs.misc[f"{cam}_camera_intrinsics"], obs.misc[f"{cam}_camera_extrinsics"]
        p, c = unproject(rgb, depth, K, T)
        pts_all.append(p)
        cols_all.append(c)
    return np.concatenate(pts_all), np.concatenate(cols_all)


def compare_oracle(points: np.ndarray, colors: np.ndarray) -> None:
    print(f"\n[1] oracle comparison — {points.shape[0]} fused points, img {IMG_SIZE}, extent {EXTENT}")
    pts_t = torch.from_numpy(points).float()
    feats_t = torch.from_numpy(colors).float() / 255.0
    center_t = torch.from_numpy(CENTER)
    for view in VIRTUAL_VIEWS:
        np_img = ortho_render(points, colors, view, center=CENTER, extent=EXTENT, img_size=IMG_SIZE)
        np_fg = np.any(np_img != 245, axis=-1)  # oracle bg is (245,245,245), incl. dilation ring
        t_img, t_mask = render_view(pts_t, feats_t, view, center=center_t,
                                    extent=EXTENT, img_size=IMG_SIZE, mode="hard")
        t_fg = t_mask.numpy()
        # IoU of foreground (torch has no dilation -> subset of oracle fg).
        inter = (np_fg & t_fg).sum()
        iou = inter / max(1, (np_fg | t_fg).sum())
        # Color agreement on common foreground (z-fight pixels may differ).
        common = np_fg & t_fg
        t_rgb = (t_img.numpy() * 255.0)
        cdiff = np.abs(t_rgb[common] - np_img[common]).mean() if common.any() else float("nan")
        print(f"  {view:6s}  fg(oracle/torch)={np_fg.sum():6d}/{t_fg.sum():6d}  "
              f"IoU={iou:.3f}  mean|Δcolor| on common={cdiff:5.1f}/255")


def check_grad() -> None:
    print("\n[2] gradcheck — soft mode differentiable w.r.t. feats")
    torch.manual_seed(0)
    pts = torch.rand(40, 3, dtype=torch.float64) * 0.3 + torch.tensor(DEFAULT_CENTER, dtype=torch.float64) - 0.15
    feats = torch.rand(40, 2, dtype=torch.float64, requires_grad=True)
    center = torch.tensor(DEFAULT_CENTER, dtype=torch.float64)

    def f(ft):
        img, _ = render_view(pts, ft, "front", center=center, extent=EXTENT,
                             img_size=24, mode="soft", tau=0.05)
        return img

    ok = torch.autograd.gradcheck(f, (feats,), eps=1e-6, atol=1e-4)
    print(f"  gradcheck passed: {ok}")


def check_cuda(points: np.ndarray, colors: np.ndarray) -> None:
    if not torch.cuda.is_available():
        print("\n[3] CUDA: not available, skipping")
        return
    print(f"\n[3] CUDA parity — {torch.cuda.get_device_name(0)}")
    pts_c = torch.from_numpy(points).float().cuda()
    feats_c = torch.from_numpy(colors).float().cuda() / 255.0
    center_c = torch.from_numpy(CENTER).cuda()
    pts_cpu, feats_cpu, center_cpu = pts_c.cpu(), feats_c.cpu(), center_c.cpu()
    for view in VIRTUAL_VIEWS:
        ic, _ = render_view(pts_c, feats_c, view, center=center_c, extent=EXTENT, img_size=IMG_SIZE)
        ih, _ = render_view(pts_cpu, feats_cpu, view, center=center_cpu, extent=EXTENT, img_size=IMG_SIZE)
        maxdiff = (ic.cpu() - ih).abs().max().item()
        print(f"  {view:6s}  max|cuda-cpu|={maxdiff:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()
    points, colors = fuse_frame(Path(args.episode), args.frame)
    compare_oracle(points, colors)
    check_grad()
    check_cuda(points, colors)


if __name__ == "__main__":
    main()
