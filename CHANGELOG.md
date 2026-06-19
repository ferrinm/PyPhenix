# Changelog

## Unreleased

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
