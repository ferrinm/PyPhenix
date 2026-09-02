# Changelog

## 0.6.0 — 2026-09-01

### Added

- **Harmony V6 acquisitions are now readable.** The reader detects the OME-XML
  namespace from the file instead of assuming HarmonyV7, and accepts both
  `HarmonyV6` and `HarmonyV7`. An unrecognised version now raises the new
  `pyphenix.UnsupportedHarmonyVersionError` naming the version it found,
  rather than failing obscurely further along. ([#21])
- OME-TIFF **Saves** are now genuinely OME-TIFF. Previously the `ome-tiff`
  format wrote a plain multi-page TIFF — no OME-XML, `is_ome=False`, and
  unlabelled `QQQYX` axes. Saves now declare `TCZYX` axes and embed
  `PhysicalSizeX/Y/Z`, `TimeIncrement`, and per-channel `Name`,
  `ExcitationWavelength` and `EmissionWavelength`. Phenix provenance that OME
  has no standard field for (plate, well, fields, stitched, timepoint offsets)
  goes into the OME-XML image `Description`. ([#19])
- `tifffile` is now an explicit dependency. It was already arriving
  transitively via napari and scikit-image, so installs without the `[napari]`
  extra were relying on luck. ([#19])

### Changed

- **Breaking:** an OME-TIFF **Save** no longer writes a JSON sidecar beside
  itself. Every field it used to carry now lives inside the file. A numpy Save
  still writes one, because a bare `.npy` has nowhere to hold it — the
  asymmetry is deliberate and recorded in
  [ADR 0002](docs/adr/0002-ome-tiff-saves-are-self-describing.md). ([#19])
- **Breaking:** the napari reader no longer claims arbitrary files. Its
  `filename_patterns` was `['*']`, so pyphenix offered to open anything dropped
  onto the viewer; it is now empty, and the plugin only accepts directories —
  which is the only thing it could ever actually read. ([#23])
- The save file dialog filters on the format selected in the combo box instead
  of always offering `*.npy`. ([#19])
- Windows paths are handled correctly when constructing URLs on POSIX systems,
  via `PureWindowsPath`. ([#20])
- The plate overview scale bar label now says "well" rather than "field"; a
  stitched cell in the overview is a whole well.

### Removed

- The unused dummy writer inherited from the napari plugin template
  (`_writer.py`). It was never registered in `napari.yaml` and never exported,
  so nothing could have called it. ([#22])

### Fixed

- Saves are written by a single shared implementation (`pyphenix._save`) that
  both the reader and the napari widget delegate to, so the two can no longer
  drift apart. A third stale copy of the save logic (`_widget_backup.py`) was
  deleted. ([#19])

[#19]: https://github.com/ferrinm/PyPhenix/issues/19
[#20]: https://github.com/ferrinm/PyPhenix/pull/20
[#21]: https://github.com/ferrinm/PyPhenix/pull/21
[#22]: https://github.com/ferrinm/PyPhenix/pull/22
[#23]: https://github.com/ferrinm/PyPhenix/pull/23

## 0.5.0 — 2026-06-18

### Added

- **Stable public API.** `FFCProfile` and `FFCCoverageWarning` are now importable
  from the top-level `pyphenix` namespace (the latter also from `pyphenix.errors`).
  `OperaPhenixReader.apply_ffc`, `OperaPhenixReader.ffc_profiles`, and the new
  `OperaPhenixReader.ffc_correction_images` method are documented as part of the
  stable surface — see the "Public API" section of the README. ([#15])
- `OperaPhenixReader.ffc_correction_images(shape=None, channel_ids=None)` —
  returns per-channel `(Y, X)` float32 illumination tiles for chunk-wise FFC.
  Designed for amortising polynomial evaluation across many `dask.array`
  chunks. Channels without a real profile are omitted from the returned dict
  (callers should treat absence as "no correction for this channel"). Defaults
  to `metadata.image_size` and `metadata.channel_ids`. ([#15])
- `OperaPhenixReader.apply_ffc` gained a keyword-only `dtype="float32" |
  "uint16"` parameter. The default `"float32"` is bytewise-identical to the
  previous behaviour. `"uint16"` re-scales each corrected channel by
  `profile.mean` before clipping to `[0, 65535]`, rounding, and casting back
  to uint16 — useful when downstream storage must stay bounded. ([#15])

### Changed

- **Behaviour change:** `OperaPhenixReader.apply_ffc` now emits
  `pyphenix.FFCCoverageWarning` (once per call) when at least one requested
  channel lacks a real FFC profile and `ffc_profiles` is non-empty. The
  warning lists which channels are corrected vs. which are returned
  uncorrected and the reason for each gap. The reader stays silent in the
  fully-uncovered case (no FFC XML detected) and the fully-covered case.
  Existing callers — the napari widget and `generate_plate_overview` — will
  start surfacing this warning when an acquisition has partial FFC coverage.
  ([#15])

[#15]: https://github.com/ferrinm/PyPhenix/issues/15
