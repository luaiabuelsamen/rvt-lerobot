"""Torch-native differentiable point-cloud renderer for RVT-style virtual views.

This is the dep-light replacement for RVT-2's custom CUDA `point-renderer`
(`external/RVT/rvt/libs/point-renderer/.../render_feature_pointcloud.cu`) and
RVT-1's PyTorch3D path — neither of which builds cleanly on Jetson aarch64.

The geometry mirrors the NumPy oracle in `rvt_lerobot/visualize/virtual_views.py`
(`ortho_render`): K fixed orthographic virtual cameras looking at the workspace,
each projecting the fused world point cloud onto a square image. The projection
is *geometric, not learned* — that is the load-bearing RVT idea.

Two render modes:
  * hard (default): nearest-point-wins z-buffer. Use for rendering the fixed
    RGB+XYZ input channels. Correct nearest-wins semantics (the NumPy oracle has
    a latent z-fight bug where the *farthest* point wins a contested pixel; we
    do not replicate that).
  * soft: depth-softmax blend over points per pixel. Differentiable w.r.t. the
    per-point feature channels (`feats`) — needed only if we ever render learned
    per-point features. Not differentiable w.r.t. point *positions* (pixel
    assignment is discrete); we don't need position gradients since points come
    from depth, geometrically.

All functions are device-agnostic (CPU or CUDA / Jetson sm_87).
"""
from __future__ import annotations

import torch
from torch import Tensor

VIRTUAL_VIEWS: tuple[str, ...] = ("front", "top", "left", "right", "back")

# view -> (axis_u, axis_v, axis_depth, flip_depth, flip_u)
# Matches the proj(...) mapping in visualize/virtual_views.py:ortho_render.
_VIEW_SPEC: dict[str, tuple[int, int, int, bool, bool]] = {
    "front": (0, 2, 1, False, False),
    "back":  (0, 2, 1, True, True),
    "top":   (0, 1, 2, True, False),
    "left":  (1, 2, 0, True, True),
    "right": (1, 2, 0, False, False),
}

# Default workspace box (meters), aligned with virtual_views.py defaults.
DEFAULT_CENTER: tuple[float, float, float] = (0.0, -0.25, 0.15)
DEFAULT_EXTENT: float = 0.6


def _project(points: Tensor, view: str, center: Tensor, extent: float, img_size: int):
    """World points (N,3) -> (u_pix, v_pix, depth, in_window mask) for one view."""
    au, av, ad, flip_d, flip_u = _VIEW_SPEC[view]
    p = points - center
    u = p[:, au]
    v = p[:, av]
    d = p[:, ad] * (-1.0 if flip_d else 1.0)
    if flip_u:
        u = -u
    half = extent / 2.0
    in_win = (u.abs() <= half) & (v.abs() <= half)
    # +z (v) maps up -> flip the row axis, identical to the NumPy oracle.
    px = ((u + half) / extent * (img_size - 1)).round().long()
    py = ((half - v) / extent * (img_size - 1)).round().long()
    px = px.clamp(0, img_size - 1)
    py = py.clamp(0, img_size - 1)
    return px, py, d, in_win


def render_view(
    points: Tensor,
    feats: Tensor,
    view: str,
    *,
    center: Tensor | None = None,
    extent: float = DEFAULT_EXTENT,
    img_size: int = 220,
    mode: str = "hard",
    tau: float = 0.01,
    bg: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Render one orthographic virtual view of a colored/featured point cloud.

    Args:
        points: (N, 3) world-frame points.
        feats:  (N, C) per-point features (e.g. RGB in [0,1], or XYZ, or learned).
        view:   one of VIRTUAL_VIEWS.
        mode:   "hard" (nearest wins) or "soft" (depth-softmax blend, differentiable in feats).
        tau:    softmax temperature (meters) for soft mode; smaller -> sharper -> closer to hard.

    Returns:
        img:  (img_size, img_size, C) rendered features.
        mask: (img_size, img_size) bool, True where any point landed.
    """
    if center is None:
        center = points.new_tensor(DEFAULT_CENTER)
    S, C = img_size, feats.shape[1]
    px, py, d, in_win = _project(points, view, center, extent, img_size)

    img = points.new_full((S * S, C), bg)
    mask = torch.zeros(S * S, dtype=torch.bool, device=points.device)
    if in_win.any():
        px, py, d, feats_w = px[in_win], py[in_win], d[in_win], feats[in_win]
        lin = py * S + px

        if mode == "hard":
            # Sort by depth DESCENDING so the nearest point is written last;
            # index_put last-write-wins => nearest point colors the pixel.
            order = torch.argsort(d, descending=True)
            lin_o, feats_o = lin[order], feats_w[order]
            img.index_copy_(0, lin_o, feats_o)
            mask.index_fill_(0, lin, True)
        elif mode == "soft":
            # Per-pixel min depth for numerical stability, then softmax(-d/tau).
            zmin = img.new_full((S * S,), float("inf"))
            zmin = zmin.scatter_reduce(0, lin, d, reduce="amin", include_self=True)
            w = torch.exp(-(d - zmin[lin]) / tau)                      # (M,)
            num = img.new_zeros(S * S, C).index_add(0, lin, feats_w * w[:, None])
            den = img.new_zeros(S * S).index_add(0, lin, w)
            hit = den > 0
            img[hit] = num[hit] / den[hit, None]
            mask[hit] = True
        else:
            raise ValueError(f"unknown mode {mode!r}")

    return img.view(S, S, C), mask.view(S, S)


def render_all_views(
    points: Tensor,
    feats: Tensor,
    *,
    views: tuple[str, ...] = VIRTUAL_VIEWS,
    center: Tensor | None = None,
    extent: float = DEFAULT_EXTENT,
    img_size: int = 220,
    mode: str = "hard",
    tau: float = 0.01,
    bg: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Render all virtual views. Returns (V,S,S,C) images and (V,S,S) masks."""
    imgs, masks = [], []
    for v in views:
        img, m = render_view(points, feats, v, center=center, extent=extent,
                             img_size=img_size, mode=mode, tau=tau, bg=bg)
        imgs.append(img)
        masks.append(m)
    return torch.stack(imgs, 0), torch.stack(masks, 0)
