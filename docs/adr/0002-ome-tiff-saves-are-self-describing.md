# An OME-TIFF Save is self-describing; a numpy Save is not

Saving a **Well** as `ome-tiff` used to write a plain multipage TIFF plus an
adjacent JSON **Sidecar**, so the image carried no pixel size, no channel
identity, and not even its own T/C/Z structure — a downstream tool saw an
undifferentiated stack of planes. We now write real OME-TIFF: the standard OME
fields carry pixel size, Z step, time increment and channel names, and the
Phenix provenance that OME has no standard home for (plate, well, fields,
stitched) goes into the OME-XML `Image` description. Having done that, the
`ome-tiff` **Sidecar** is pure duplication and is no longer written. The numpy
format keeps its **Sidecar**, because a bare `.npy` has nowhere else to put any
of this.

The asymmetry is deliberate and is the point worth remembering: metadata lives
inside the artifact whenever the format can hold it, and beside the artifact
only when it cannot. Dropping the `ome-tiff` **Sidecar** is a breaking change —
nothing in this repo read that file, but a user's own script might have — and it
was accepted because a **Save** that stops being self-describing the moment it is
copied off the machine defeats the reason for choosing OME-TIFF at all. If the
sidecar has to return, add it back alongside the embedded metadata rather than
moving provenance back out of the file, and do not "fix" the inconsistency by
stripping numpy's sidecar: that would make the numpy format silently lossy.
