# Plate-wide contrast limits are computed on downsampled, max-projected pixels

The plate overview needs a per-channel intensity range that's identical across
every well, so wells can be visually compared. The obvious approach — pool every
raw 16-bit pixel from every well and take the 99.5th percentile — roughly doubles
I/O (one streaming pass to compute limits, a second pass to render). Instead we
compute the percentile on the same downsampled, Z-max-projected pixels we already
hold for rendering, which keeps the overview single-pass and bounded in memory.

The trade-off is that downsampling (block-mean) attenuates single-pixel hot spots,
so the resulting `[0, p99.5]` will sit slightly lower than the napari widget's
per-well number on the same data. For a *diagnostic* overview this is arguably an
improvement (suppresses isolated saturation), but it does mean overview intensities
are not directly comparable to live napari intensities. If a future use case
demands exact parity, switch to a two-pass implementation rather than changing the
percentile or the downsampling kernel — both of those would silently shift every
existing overview's apparent brightness.