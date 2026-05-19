# pyphenix

A reader and napari widget for high-content microscopy data acquired on the
PerkinElmer/Revvity **Opera Phenix** and processed by its **Harmony** software.
This file captures the shared language used across the codebase, the napari
widget UI, and the wishlist features.

## Language

### Acquisition layout

**Plate**:
A multi-well container the acquisition was performed on, typically 96-well
(8 rows × 12 columns) or 384-well (16 × 24).
_Avoid_: dish, sample holder.

**Well**:
A single addressable position on the **Plate**, identified as `r{row:02d}c{col:02d}`
(e.g. `r02c03`). Rows are surfaced to humans as letters (A–H / A–P).
_Avoid_: site, position (these mean **Field**).

**Field**:
A single rectangular field-of-view captured at one X,Y position inside a
**Well**. Opera Phenix typically captures a grid of **Fields** per **Well**
(e.g. 25 fields = 5×5) which can be stitched.
_Avoid_: tile (overloaded), FOV (only fine in passing).

**Stitching**:
Combining all **Fields** of one **Well** into a single larger image using their
recorded stage coordinates. The opposite of "first-field" mode, which reads
only one **Field** per **Well**.

**Channel**:
A fluorescence acquisition (excitation/emission/exposure) — e.g. `DAPI`,
`Alexa 488`. Each **Channel** has a name, an integer ID, and a suggested
display colormap. The channel→colormap mapping is the single source of truth
shared by the napari widget and the plate overview.

**Z-plane** / **Z-slice**:
One depth slice in a Z-stack. Used interchangeably; **Z-slice** wins in API
parameter names (`z_slices=`).

**Timepoint**:
One time index in a time-series acquisition.

### Source data structure

**Export**:
A directory produced by Harmony's "Export" operation. Image files are TIFFs
under `Images/`; metadata is in `Index.xml`. The primary input shape pyphenix
reads.

**Archive**:
An alternate Harmony output (`.kw.txt` + per-image files). Supported by the
reader; less common in practice.

**FFC (flat-field correction)**:
Per-**Channel** illumination-correction profile shipped alongside the acquisition.
When present and enabled, divides out the vignetting/gain pattern. On by default
in `read_data` and in the plate overview.

### Plate overview (new)

**Plate overview**:
A single diagnostic PNG showing a grid of downsampled **Well** thumbnails in
the layout of the **Plate**, used to spot acquisition or biology issues at a
glance. Generated as one PNG per **Channel combo**, written to a flat output
directory alongside a JSON provenance sidecar.

**Channel combo**:
A non-empty subset of the acquired **Channels** rendered into one **Plate
overview** PNG. **Singletons** (one **Channel**) are colored with `viridis`;
**Merges** (two or more) use each **Channel**'s suggested colormap and additive
blending, matching what the napari widget shows.

**Plate-wide contrast limits**:
A per-**Channel** intensity display range `[0, p99.5]` computed once across
the downsampled, max-projected pixels of every acquired **Well**. Every
**Well** in every **Plate overview** is rendered with these identical limits,
so signal differences between wells are visually comparable.

**Field choice**:
The per-**Well** rule for what to render in the overview: a specific **Field**
index, `'stitched'`, or the default first **Field**. Applied uniformly to
every **Well** in the run.

## Relationships

- A **Plate** contains many **Wells**; a **Well** contains many **Fields**.
- A **Field** has many **Z-planes** × **Channels** × **Timepoints** (5-D `(T,C,Z,Y,X)`).
- A **Plate overview** is one PNG per **Channel combo** (so `2^N − 1` PNGs for
  N selected **Channels**) plus one JSON sidecar.
- **Plate-wide contrast limits** are computed once per overview run and shared
  by every **Channel combo** PNG in that run.
- The napari widget and the plate overview both consume the channel→colormap
  mapping from the same shared module — they must not duplicate it.

## Example dialogue

> **Dev:** "If I run the overview on a 6-channel plate with `channels=[1,4]`, how many PNGs do I get?"
> **Domain expert:** "Three — Ch1 alone in viridis, Ch4 alone in viridis, and a Ch1+Ch4 merge in their assigned colormaps. Plate-wide contrast is still computed per channel from the full set, then applied identically to every well."

> **Dev:** "Should the overview let users gray out wells filtered by the experimental metadata CSV?"
> **Domain expert:** "Not in v1. The point of the overview is to *see what was acquired*, not what you're going to analyze. If you've filtered wells, run the overview before the filter."

## Flagged ambiguities

- "Field" vs "tile" — both were used in the wishlist for what the napari
  widget calls a "tile" in side-by-side mode. Resolved: **Field** is always
  the acquisition unit; "tile" is only used for napari side-by-side viewer
  positioning.
- "Stitched" applies to **Field**-level combination only. The **Plate overview**
  grid is not "stitched" — it's a `matplotlib` grid of independent thumbnails.