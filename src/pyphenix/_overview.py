"""Plate overview image generator.

Produces one diagnostic PNG per channel combo plus a JSON provenance
sidecar for an Opera Phenix experiment. Importable without napari.
"""

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from PIL import Image as PILImage
from tqdm import tqdm

from ._colormaps import channel_color
from ._reader import OperaPhenixReader

try:
    from ._version import version as __pyphenix_version__
except ImportError:
    __pyphenix_version__ = "unknown"


# napari color names → (R, G, B) for additive merge compositing.
# Single-color colormaps go from black to the named tip.
_COLOR_RGB = {
    "gray": (1.0, 1.0, 1.0),
    "grey": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def _named_cmap(color_name: str) -> LinearSegmentedColormap:
    """Black → named-color linear colormap for matplotlib display."""
    rgb = _COLOR_RGB.get(color_name.lower(), (1.0, 1.0, 1.0))
    return LinearSegmentedColormap.from_list(
        f"pyphenix_{color_name}", [(0, 0, 0), rgb]
    )


def _downsample(arr: np.ndarray, target_long: int) -> np.ndarray:
    """Block-mean downsample so the longest side ≤ ``target_long``.

    Parameters
    ----------
    arr : np.ndarray
        Shape ``(C, H, W)``, any numeric dtype.
    target_long : int
        Target pixel count on the longest side.

    Returns
    -------
    np.ndarray
        Shape ``(C, h, w)`` float32 with ``max(h, w) ≈ target_long``.
    """
    C, H, W = arr.shape
    long_side = max(H, W)
    if long_side <= target_long:
        return arr.astype(np.float32)
    scale = target_long / long_side
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    out = np.zeros((C, new_h, new_w), dtype=np.float32)
    for c in range(C):
        img = PILImage.fromarray(arr[c].astype(np.float32), mode="F")
        out[c] = np.asarray(img.resize((new_w, new_h), PILImage.BOX))
    return out


def _compute_plate_contrast(
    well_thumbs: Dict[Tuple[int, int], np.ndarray],
    channels: List[int],
) -> Dict[int, Tuple[float, float]]:
    """Per-channel ``[0, p99.5]`` over non-zero downsampled pixels.

    See ``docs/adr/0001-plate-wide-contrast-on-downsampled-pixels.md``.
    """
    limits: Dict[int, Tuple[float, float]] = {}
    for ch_idx, ch_id in enumerate(channels):
        if not well_thumbs:
            limits[ch_id] = (0.0, 1.0)
            continue
        pooled = np.concatenate(
            [t[ch_idx].ravel() for t in well_thumbs.values()]
        )
        nonzero = pooled[pooled > 0]
        if nonzero.size:
            limits[ch_id] = (0.0, float(np.percentile(nonzero, 99.5)))
        else:
            limits[ch_id] = (0.0, 1.0)
    return limits


def _normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly map ``[lo, hi]`` to ``[0, 1]`` with clipping."""
    span = hi - lo
    if span <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / span, 0.0, 1.0)


def _render_cell(
    thumb: np.ndarray,
    combo_channel_indices: List[int],
    contrast: List[Tuple[float, float]],
    combo_colors: List[str],
    is_singleton: bool,
) -> np.ndarray:
    """Render a single well's thumbnail for one channel combo.

    Returns an ``(h, w, 3)`` RGB float32 array in ``[0, 1]``.
    """
    h, w = thumb.shape[1:]
    if is_singleton:
        ch_idx = combo_channel_indices[0]
        lo, hi = contrast[0]
        norm = _normalize(thumb[ch_idx], lo, hi)
        viridis = matplotlib.colormaps["viridis"]
        return viridis(norm)[..., :3].astype(np.float32)

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for ch_idx, (lo, hi), color in zip(
        combo_channel_indices, contrast, combo_colors
    ):
        norm = _normalize(thumb[ch_idx], lo, hi)
        r, g, b = _COLOR_RGB.get(color.lower(), (1.0, 1.0, 1.0))
        rgb[..., 0] += norm * r
        rgb[..., 1] += norm * g
        rgb[..., 2] += norm * b
    return np.clip(rgb, 0.0, 1.0)


def _nice_scalebar_length_um(visible_um: float) -> float:
    """Pick a 1/2/5×10^k value covering ~15-25 % of ``visible_um``."""
    if visible_um <= 0:
        return 1.0
    target = visible_um * 0.2
    exp = np.floor(np.log10(target))
    base = target / (10**exp)
    if base < 1.5:
        nice = 1.0
    elif base < 3.5:
        nice = 2.0
    elif base < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10**exp)


def _row_letter(row: int) -> str:
    """1-based row index → ``'A'``, ``'B'``, …, ``'P'`` (with 'I' kept)."""
    if row < 1 or row > 26:
        return str(row)
    return chr(ord("A") + row - 1)


def _render_combo_png(
    *,
    combo: Tuple[int, ...],
    combo_channel_indices: List[int],
    well_thumbs: Dict[Tuple[int, int], np.ndarray],
    contrast_for_combo: List[Tuple[float, float]],
    combo_colors: List[str],
    channel_names: Dict[int, str],
    plate_rows: int,
    plate_cols: int,
    cell_h: int,
    cell_w: int,
    plate_id: str,
    objective_mag: Optional[str],
    um_per_thumb_pixel: float,
    outpath: Path,
) -> None:
    """Render and save one PNG for one channel combo."""
    is_singleton = len(combo) == 1

    # Stitch all wells into one big RGB image.
    H = plate_rows * cell_h
    W = plate_cols * cell_w
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for (row, col), thumb in well_thumbs.items():
        if row < 1 or row > plate_rows or col < 1 or col > plate_cols:
            continue
        rgb = _render_cell(
            thumb,
            combo_channel_indices,
            contrast_for_combo,
            combo_colors,
            is_singleton,
        )
        h, w = rgb.shape[:2]
        h = min(h, cell_h)
        w = min(w, cell_w)
        y0 = (row - 1) * cell_h
        x0 = (col - 1) * cell_w
        canvas[y0 : y0 + h, x0 : x0 + w] = rgb[:h, :w]

    n_combo = len(combo)
    fig_w = max(8.0, plate_cols * 0.55 + 2.5)
    fig_h = max(6.0, plate_rows * 0.55 + 2.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[20, 1],
        height_ratios=[20, 1],
        hspace=0.05,
        wspace=0.05,
        left=0.07,
        right=0.93,
        top=0.88,
        bottom=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(canvas, interpolation="nearest", origin="upper")

    # Column ticks on top: 1..plate_cols at cell centers.
    col_positions = [(c - 0.5) * cell_w for c in range(1, plate_cols + 1)]
    ax.set_xticks(col_positions)
    ax.set_xticklabels([str(c) for c in range(1, plate_cols + 1)], fontsize=7)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Row ticks on left: A..letter(plate_rows).
    row_positions = [(r - 0.5) * cell_h for r in range(1, plate_rows + 1)]
    ax.set_yticks(row_positions)
    ax.set_yticklabels(
        [_row_letter(r) for r in range(1, plate_rows + 1)], fontsize=7
    )
    ax.tick_params(axis="both", length=0)

    # Subtle gridlines between cells.
    for c in range(plate_cols + 1):
        ax.axvline(c * cell_w - 0.5, color="white", linewidth=0.3, alpha=0.3)
    for r in range(plate_rows + 1):
        ax.axhline(r * cell_h - 0.5, color="white", linewidth=0.3, alpha=0.3)
    ax.set_xlim(-0.5, plate_cols * cell_w - 0.5)
    ax.set_ylim(plate_rows * cell_h - 0.5, -0.5)
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title + subtitle.
    if is_singleton:
        ch_id = combo[0]
        title_combo = f"Ch{ch_id}: {channel_names.get(ch_id, '?')}"
    elif n_combo == len({*channel_names}):
        title_combo = "Merge: all channels"
    else:
        title_combo = "Merge: " + " + ".join(
            f"Ch{c}" for c in combo
        )
    fig.suptitle(
        f"{plate_id} — {title_combo}",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )
    if objective_mag:
        fig.text(
            0.5,
            0.925,
            f"Objective: {objective_mag}×",
            ha="center",
            fontsize=9,
            color="gray",
        )

    # Colorbar column on the right: stack one per channel in combo.
    cb_gs = gs[0, 1].subgridspec(nrows=n_combo, ncols=1, hspace=0.6)
    for i, ch_id in enumerate(combo):
        cax = fig.add_subplot(cb_gs[i, 0])
        lo, hi = contrast_for_combo[i]
        if is_singleton:
            cmap = matplotlib.colormaps["viridis"]
        else:
            cmap = _named_cmap(combo_colors[i])
        norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)
        cb = matplotlib.colorbar.ColorbarBase(
            cax, cmap=cmap, norm=norm, orientation="vertical"
        )
        cb.ax.tick_params(labelsize=6)
        cb.set_label(
            f"Ch{ch_id}: {channel_names.get(ch_id, '?')}", fontsize=7
        )

    # Scale bar across the bottom (in µm). Length: nice 1/2/5×10^k value
    # covering ~20 % of one well's visible width.
    sb_ax = fig.add_subplot(gs[1, 0])
    sb_ax.set_xlim(0, cell_w)
    sb_ax.set_ylim(0, 1)
    sb_ax.axis("off")
    well_visible_um = cell_w * um_per_thumb_pixel
    bar_um = _nice_scalebar_length_um(well_visible_um)
    bar_thumb_px = bar_um / um_per_thumb_pixel if um_per_thumb_pixel > 0 else 0
    sb_ax.add_patch(
        plt.Rectangle((0, 0.55), bar_thumb_px, 0.15, color="black")
    )
    sb_ax.text(
        bar_thumb_px / 2,
        0.35,
        f"{bar_um:g} µm  ·  one well ≈ {well_visible_um:.0f} µm",
        ha="center",
        va="top",
        fontsize=7,
    )

    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_plate_overview(
    experiment_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    field: Optional[Union[int, str]] = None,
    channels: Optional[List[int]] = None,
    timepoint: Optional[int] = None,
    z_slices: Optional[Union[int, List[int]]] = None,
    well_px: int = 300,
    contrast_limits: Optional[Dict[int, Tuple[float, float]]] = None,
    apply_ffc: bool = True,
    verbose: bool = True,
) -> List[Path]:
    """Generate plate overview PNGs and a JSON sidecar.

    Produces one PNG per non-empty channel combo (``2**N - 1`` for N
    selected channels) plus a single JSON provenance sidecar. Every PNG
    shares one set of plate-wide per-channel contrast limits so wells
    are visually comparable.

    Parameters
    ----------
    experiment_path : str or Path
        Path to an Opera Phenix Harmony export directory.
    output_dir : str or Path
        Directory to write PNGs and JSON sidecar into (created if absent).
    field : int, ``'stitched'``, or ``None``, optional
        Per-well field-choice rule applied uniformly to every well.
        ``None`` (default) uses each well's first available field.
        ``'stitched'`` stitches all fields. An integer selects that field
        from each well.
    channels : list of int, optional
        Channel IDs to consider (the full combo set comes from this
        subset). ``None`` uses every acquired channel.
    timepoint : int, optional
        Single timepoint to render. ``None`` uses the first timepoint.
    z_slices : int, list of int, or None, optional
        Z planes to load before max-projecting. ``None`` loads all.
    well_px : int, default 300
        Per-well render size on the longest side, in pixels.
    contrast_limits : dict, optional
        Optional ``{channel_id: (lo, hi)}`` override. Channels not listed
        fall back to the computed plate-wide value. Passing this skips
        nothing computationally — limits are still computed for the
        sidecar — but the override is applied for rendering.
    apply_ffc : bool, default True
        Apply flat-field correction if profiles are present.
    verbose : bool, default True
        Print progress and show a ``tqdm`` progress bar.

    Returns
    -------
    list of Path
        Every file written (PNGs + the JSON sidecar).
    """
    experiment_path = Path(experiment_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = OperaPhenixReader(str(experiment_path), verbose=verbose)
    meta = reader.metadata

    # Resolve selection.
    all_channels = list(meta.channel_ids)
    if channels is None:
        sel_channels = all_channels
    else:
        sel_channels = [c for c in channels if c in all_channels]
    if not sel_channels:
        raise ValueError("No selected channels overlap with acquired channels.")

    if timepoint is None:
        timepoint = meta.timepoints[0]
    elif timepoint not in meta.timepoints:
        raise ValueError(
            f"timepoint={timepoint} not in acquired timepoints {meta.timepoints}"
        )

    stitch_fields = field == "stitched"
    fixed_field: Optional[int] = None
    if not stitch_fields and field is not None:
        fixed_field = int(field)

    if isinstance(z_slices, int):
        z_arg: Optional[List[int]] = [z_slices]
    else:
        z_arg = z_slices

    # Single pass over wells: read → max-project Z → downsample → store.
    available_wells = reader.get_available_wells()
    well_thumbs: Dict[Tuple[int, int], np.ndarray] = {}

    iterator = available_wells
    if verbose:
        iterator = tqdm(available_wells, desc="Reading wells")

    for (row, col) in iterator:
        if stitch_fields:
            use_field = None
            use_stitch = True
        else:
            wfields = reader.well_field_map.get((row, col), meta.fields)
            if fixed_field is not None:
                if fixed_field not in wfields:
                    continue
                use_field = fixed_field
            else:
                use_field = wfields[0] if wfields else meta.fields[0]
            use_stitch = False

        data, _ = reader.read_data(
            row=row,
            column=col,
            field=use_field,
            stitch_fields=use_stitch,
            timepoints=timepoint,
            channels=sel_channels,
            z_slices=z_arg,
            apply_ffc=apply_ffc,
            verbose=False,
        )
        # data shape: (T, C, Z, Y, X); we asked for single tp, so T == 1.
        arr = data[0]  # (C, Z, Y, X)
        arr = arr.max(axis=1)  # max-project Z → (C, Y, X)
        thumb = _downsample(arr, well_px)
        well_thumbs[(row, col)] = thumb

    # Cell size = largest thumb we got (cells get zero-padded to this).
    if well_thumbs:
        cell_h = max(t.shape[1] for t in well_thumbs.values())
        cell_w = max(t.shape[2] for t in well_thumbs.values())
    else:
        cell_h = cell_w = well_px

    # Re-pad shorter thumbs to (cell_h, cell_w) for uniform grid placement.
    for key, thumb in list(well_thumbs.items()):
        C, h, w = thumb.shape
        if h == cell_h and w == cell_w:
            continue
        padded = np.zeros((C, cell_h, cell_w), dtype=thumb.dtype)
        padded[:, :h, :w] = thumb
        well_thumbs[key] = padded

    # Plate-wide contrast limits — always computed for the sidecar.
    computed_contrast = _compute_plate_contrast(well_thumbs, sel_channels)
    rendering_contrast: Dict[int, Tuple[float, float]] = dict(computed_contrast)
    if contrast_limits:
        for ch_id, lh in contrast_limits.items():
            if ch_id in rendering_contrast:
                rendering_contrast[ch_id] = (float(lh[0]), float(lh[1]))

    # Channel colormaps for merges; singletons always use viridis.
    channel_names = {
        ch_id: meta.channels[ch_id]["name"] for ch_id in sel_channels
    }
    ch_colors: Dict[int, str] = {}
    for idx, ch_id in enumerate(sel_channels):
        ch_colors[ch_id] = channel_color(channel_names[ch_id], idx)

    # Effective µm per thumb pixel — derived from one well's downsample.
    if reader.metadata.image_size and well_thumbs:
        # image_size is stored as (X, Y) in the reader.
        full_x = reader.metadata.image_size[0]
        full_y = reader.metadata.image_size[1]
        any_thumb = next(iter(well_thumbs.values()))
        # In first-field mode the thumb matches the single-field downsample;
        # in stitched mode this underestimates effective µm/pixel slightly
        # (still adequate for a diagnostic scale bar).
        thumb_long = max(any_thumb.shape[1], any_thumb.shape[2])
        full_long = max(full_x, full_y)
        downsample_factor = full_long / thumb_long if thumb_long else 1.0
    else:
        downsample_factor = 1.0
    um_per_full_px = reader.metadata.pixel_size[1] * 1e6
    um_per_thumb_pixel = um_per_full_px * downsample_factor

    objective_mag = None
    for ch in meta.channels.values():
        if ch.get("objective_mag"):
            objective_mag = ch["objective_mag"]
            break

    plate_id = meta.plate_id
    written: List[Path] = []
    n = len(sel_channels)

    combos_iter = []
    for k in range(1, n + 1):
        for combo in combinations(sel_channels, k):
            combos_iter.append(combo)

    if verbose:
        combos_iter_display = tqdm(combos_iter, desc="Rendering combos")
    else:
        combos_iter_display = combos_iter

    for combo in combos_iter_display:
        is_all = len(combo) == n
        if len(combo) == 1:
            fname = f"{plate_id}_overview_ch{combo[0]}.png"
        elif is_all:
            fname = f"{plate_id}_overview_merge_all.png"
        else:
            fname = (
                f"{plate_id}_overview_"
                + "+".join(f"ch{c}" for c in combo)
                + ".png"
            )
        outpath = output_dir / fname
        combo_channel_indices = [sel_channels.index(c) for c in combo]
        contrast_for_combo = [rendering_contrast[c] for c in combo]
        combo_colors = [ch_colors[c] for c in combo]
        _render_combo_png(
            combo=combo,
            combo_channel_indices=combo_channel_indices,
            well_thumbs=well_thumbs,
            contrast_for_combo=contrast_for_combo,
            combo_colors=combo_colors,
            channel_names=channel_names,
            plate_rows=meta.plate_rows,
            plate_cols=meta.plate_columns,
            cell_h=cell_h,
            cell_w=cell_w,
            plate_id=plate_id,
            objective_mag=objective_mag,
            um_per_thumb_pixel=um_per_thumb_pixel,
            outpath=outpath,
        )
        written.append(outpath)

    # JSON sidecar.
    sidecar = {
        "pyphenix_version": __pyphenix_version__,
        "plate_id": plate_id,
        "experiment_path": str(experiment_path),
        "parameters": {
            "field": field,
            "channels": sel_channels,
            "timepoint": timepoint,
            "z_slices": z_slices,
            "well_px": well_px,
            "apply_ffc": apply_ffc,
            "contrast_limits_override": (
                {str(k): list(v) for k, v in contrast_limits.items()}
                if contrast_limits
                else None
            ),
        },
        "plate_layout": {
            "rows": meta.plate_rows,
            "columns": meta.plate_columns,
        },
        "plate_contrast_limits": {
            str(k): [float(v[0]), float(v[1])]
            for k, v in computed_contrast.items()
        },
        "rendering_contrast_limits": {
            str(k): [float(v[0]), float(v[1])]
            for k, v in rendering_contrast.items()
        },
        "channel_colormaps": {str(k): v for k, v in ch_colors.items()},
        "channel_names": {str(k): v for k, v in channel_names.items()},
        "pixel_size_m": [
            float(reader.metadata.pixel_size[0]),
            float(reader.metadata.pixel_size[1]),
        ],
        "um_per_thumb_pixel": float(um_per_thumb_pixel),
        "objective_magnification": objective_mag,
        "output_files": [str(p.name) for p in written],
    }
    json_path = output_dir / f"{plate_id}_overview.json"
    json_path.write_text(json.dumps(sidecar, indent=2))
    written.append(json_path)

    return written
