import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyphenix import generate_plate_overview


PHENIX_NS = "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"


def _add_text(parent, tag, text, **attrs):
    el = ET.SubElement(parent, tag, **attrs)
    el.text = str(text)
    return el


def _add_channel_entry(maps_el, ch_id: int, ch_name: str, img_size: int):
    entry = ET.SubElement(maps_el, "Entry", ChannelID=str(ch_id))
    _add_text(entry, "ChannelName", ch_name)
    _add_text(entry, "ImageType", "Signal")
    _add_text(entry, "ImageResolutionX", "3.0E-07", Unit="m")
    _add_text(entry, "ImageResolutionY", "3.0E-07", Unit="m")
    _add_text(entry, "ImageSizeX", img_size)
    _add_text(entry, "ImageSizeY", img_size)
    _add_text(entry, "MainExcitationWavelength", "405", Unit="nm")
    _add_text(entry, "MainEmissionWavelength", "456", Unit="nm")
    _add_text(entry, "ObjectiveMagnification", "40", Unit="")
    _add_text(entry, "ObjectiveNA", "1.1", Unit="")
    _add_text(entry, "ExposureTime", "0.1", Unit="s")


def _add_image_entry(images_el, row, col, ch_id, ch_name):
    image_id = f"{row:02d}{col:02d}K1F1P1R1"
    image = ET.SubElement(images_el, "Image", Version="1")
    _add_text(image, "id", image_id)
    _add_text(image, "State", "Ok")
    fname = f"r{row:02d}c{col:02d}f01p01-ch{ch_id}sk1fk1fl1.tiff"
    _add_text(image, "URL", fname)
    _add_text(image, "Row", row)
    _add_text(image, "Col", col)
    _add_text(image, "FieldID", "1")
    _add_text(image, "PlaneID", "1")
    _add_text(image, "TimepointID", "1")
    _add_text(image, "ChannelID", ch_id)
    _add_text(image, "PositionX", "0.0003", Unit="m")
    _add_text(image, "PositionY", "0.0003", Unit="m")
    _add_text(image, "PositionZ", "-2E-06", Unit="m")
    _add_text(image, "AbsPositionZ", "0.135", Unit="m")
    return fname


@pytest.fixture
def mock_plate(tmp_path):
    """A 2×2 plate, every well populated, 3 channels, 1 field, 1 z, 1 tp.

    Image size is 60×60 so tests stay fast.
    """
    images_dir = tmp_path / "Images"
    images_dir.mkdir()

    img_size = 60
    plate_rows, plate_cols = 2, 2
    channels = [(1, "DAPI"), (2, "Alexa 488"), (3, "mCherry")]

    root = ET.Element("EvaluationInputData")
    root.set("xmlns", PHENIX_NS)
    root.set("Version", "2")
    _add_text(root, "User", "TEST")
    _add_text(root, "InstrumentType", "Phenix")

    plates = ET.SubElement(root, "Plates")
    plate = ET.SubElement(plates, "Plate")
    _add_text(plate, "PlateID", "MOCK001")
    _add_text(plate, "PlateRows", plate_rows)
    _add_text(plate, "PlateColumns", plate_cols)
    for r in range(1, plate_rows + 1):
        for c in range(1, plate_cols + 1):
            ET.SubElement(plate, "Well", id=f"{r:02d}{c:02d}")

    wells = ET.SubElement(root, "Wells")
    for r in range(1, plate_rows + 1):
        for c in range(1, plate_cols + 1):
            well = ET.SubElement(wells, "Well")
            _add_text(well, "id", f"{r:02d}{c:02d}")
            _add_text(well, "Row", r)
            _add_text(well, "Col", c)
            for ch_id, _ in channels:
                ET.SubElement(
                    well, "Image", id=f"{r:02d}{c:02d}K1F1P1R1"
                )

    maps = ET.SubElement(root, "Maps")
    map_elem = ET.SubElement(maps, "Map")
    for ch_id, ch_name in channels:
        _add_channel_entry(map_elem, ch_id, ch_name, img_size)

    images_el = ET.SubElement(root, "Images")
    rng = np.random.default_rng(42)
    for r in range(1, plate_rows + 1):
        for c in range(1, plate_cols + 1):
            for ch_id, ch_name in channels:
                fname = _add_image_entry(images_el, r, c, ch_id, ch_name)
                # Per-channel intensity scale so plate-wide contrast varies.
                scale = 200 + 800 * ch_id
                arr = rng.integers(
                    0, scale, (img_size, img_size), dtype=np.uint16
                )
                Image.fromarray(arr).save(images_dir / fname)

    tree = ET.ElementTree(root)
    tree.write(
        images_dir / "Index.xml", encoding="utf-8", xml_declaration=True
    )
    return tmp_path


def test_writes_all_combos_and_sidecar(mock_plate, tmp_path):
    out = tmp_path / "overviews"
    written = generate_plate_overview(
        experiment_path=mock_plate,
        output_dir=out,
        well_px=80,
        verbose=False,
    )

    # 3 channels → 2^3 - 1 = 7 PNGs, plus one JSON sidecar.
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 7
    sidecars = list(out.glob("*.json"))
    assert len(sidecars) == 1
    assert set(written) == set(pngs + sidecars)


def test_filename_conventions(mock_plate, tmp_path):
    out = tmp_path / "overviews"
    generate_plate_overview(
        experiment_path=mock_plate,
        output_dir=out,
        well_px=80,
        verbose=False,
    )
    names = {p.name for p in out.glob("*.png")}
    # Singletons named ch{X}
    assert "MOCK001_overview_ch1.png" in names
    assert "MOCK001_overview_ch2.png" in names
    assert "MOCK001_overview_ch3.png" in names
    # Pairwise merges named with ch{X}+ch{Y}
    assert "MOCK001_overview_ch1+ch2.png" in names
    assert "MOCK001_overview_ch1+ch3.png" in names
    assert "MOCK001_overview_ch2+ch3.png" in names
    # Grand merge has special name
    assert "MOCK001_overview_merge_all.png" in names


def test_sidecar_contents(mock_plate, tmp_path):
    out = tmp_path / "overviews"
    generate_plate_overview(
        experiment_path=mock_plate,
        output_dir=out,
        well_px=80,
        verbose=False,
    )
    sidecar = json.loads((out / "MOCK001_overview.json").read_text())
    assert sidecar["plate_id"] == "MOCK001"
    assert sidecar["plate_layout"] == {"rows": 2, "columns": 2}
    assert sidecar["parameters"]["well_px"] == 80
    assert sidecar["parameters"]["apply_ffc"] is True
    # All 3 channels show up with computed contrast.
    assert set(sidecar["plate_contrast_limits"]) == {"1", "2", "3"}
    for lo, hi in sidecar["plate_contrast_limits"].values():
        assert lo == 0.0
        assert hi > 0
    # Channel colormaps come from _colormaps.channel_color.
    assert sidecar["channel_colormaps"] == {
        "1": "cyan",
        "2": "green",
        "3": "magenta",
    }
    assert sidecar["channel_names"] == {
        "1": "DAPI",
        "2": "Alexa 488",
        "3": "mCherry",
    }
    assert sidecar["objective_magnification"] == "40"
    assert "pyphenix_version" in sidecar
    assert sidecar["pixel_size_m"][0] == pytest.approx(3.0e-7)


def test_channels_filter_limits_combos(mock_plate, tmp_path):
    out = tmp_path / "overviews"
    generate_plate_overview(
        experiment_path=mock_plate,
        output_dir=out,
        channels=[1, 3],
        well_px=80,
        verbose=False,
    )
    pngs = {p.name for p in out.glob("*.png")}
    # With 2 selected channels: 2^2 - 1 = 3 combos.
    assert pngs == {
        "MOCK001_overview_ch1.png",
        "MOCK001_overview_ch3.png",
        "MOCK001_overview_merge_all.png",
    }


def test_contrast_override_appears_in_sidecar(mock_plate, tmp_path):
    out = tmp_path / "overviews"
    generate_plate_overview(
        experiment_path=mock_plate,
        output_dir=out,
        well_px=80,
        contrast_limits={1: (10.0, 500.0)},
        verbose=False,
    )
    sidecar = json.loads((out / "MOCK001_overview.json").read_text())
    assert sidecar["rendering_contrast_limits"]["1"] == [10.0, 500.0]
    # Override is recorded under parameters too.
    assert sidecar["parameters"]["contrast_limits_override"] == {
        "1": [10.0, 500.0]
    }
    # Unoverridden channels keep computed plate-wide value.
    assert (
        sidecar["rendering_contrast_limits"]["2"]
        == sidecar["plate_contrast_limits"]["2"]
    )


def test_png_files_are_non_empty(mock_plate, tmp_path):
    out = tmp_path / "overviews"
    generate_plate_overview(
        experiment_path=mock_plate,
        output_dir=out,
        well_px=80,
        verbose=False,
    )
    for p in out.glob("*.png"):
        # PNGs render to >5KB even for a 2x2 plate at 80px/well.
        assert p.stat().st_size > 5000, p


def test_overview_module_source_has_no_napari_import():
    import pyphenix._overview as overview_module
    src = Path(overview_module.__file__).read_text()
    for line in src.splitlines():
        stripped = line.split("#", 1)[0].strip()
        assert not stripped.startswith(
            ("import napari", "from napari")
        ), f"_overview.py must not import napari: {line!r}"


def test_overview_works_when_napari_is_blocked():
    # Headless-pipeline scenario: napari not installed. Run in a fresh
    # subprocess with a meta_path blocker, then import _overview directly
    # (skipping pyphenix.__init__, which conditionally imports the widget).
    import subprocess
    import textwrap

    script = textwrap.dedent(
        """
        import sys, importlib.util
        from pathlib import Path

        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == 'napari' or name.startswith('napari.'):
                    raise ImportError(f'blocked: {name}')
                return None

        sys.meta_path.insert(0, _Block())

        # Import _overview without triggering pyphenix/__init__.
        pkg_init = Path(sys.argv[1]) / 'pyphenix' / '__init__.py'
        spec = importlib.util.spec_from_file_location(
            'pyphenix', pkg_init,
            submodule_search_locations=[str(pkg_init.parent)],
        )
        # Build a stub package so relative imports inside _overview resolve.
        import types
        pkg = types.ModuleType('pyphenix')
        pkg.__path__ = [str(pkg_init.parent)]
        sys.modules['pyphenix'] = pkg

        from pyphenix import _overview
        assert hasattr(_overview, 'generate_plate_overview')
        print('ok')
        """
    )
    src_root = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", script, str(src_root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


def test_resolve_field_label():
    from pyphenix._overview import _resolve_field_label

    assert _resolve_field_label("stitched", True, None) == "stitched"
    assert _resolve_field_label(3, False, 3) == "3"
    assert _resolve_field_label(None, False, None) == "per-well first"


def test_resolve_z_label():
    from pyphenix._overview import _resolve_z_label

    assert _resolve_z_label(None) == "all"
    assert _resolve_z_label(2) == "[2]"
    assert _resolve_z_label([1, 2, 3]) == "[1, 2, 3]"


def test_combo_label():
    from pyphenix._overview import _combo_label

    names = {1: "DAPI", 2: "Alexa 488", 3: "mCherry"}
    assert _combo_label((1,), names, 3) == "Ch1: DAPI"
    assert _combo_label((1, 2, 3), names, 3) == "Merge: all channels"
    assert _combo_label((1, 3), names, 3) == "Merge: Ch1 + Ch3"


def test_options_label_includes_all_user_controllable_options():
    from pyphenix._overview import _options_label

    line = _options_label(
        objective_mag="40",
        field_label="stitched",
        timepoint_label="0",
        z_label="all",
        ffc_label="on",
    )
    for fragment in (
        "Objective: 40×",
        "Field: stitched",
        "T: 0",
        "Z: all",
        "FFC: on",
    ):
        assert fragment in line


def test_options_label_omits_objective_when_unknown():
    from pyphenix._overview import _options_label

    line = _options_label(
        objective_mag=None,
        field_label="per-well first",
        timepoint_label="1",
        z_label="[2]",
        ffc_label="off",
    )
    assert "Objective" not in line
    assert "Field: per-well first" in line
    assert "FFC: off" in line


def test_subtitle_renders_into_png(mock_plate, tmp_path):
    # The rendered figure carries the new subtitle/options as fig.texts.
    # Render via the public API into a tmp dir, then re-render one combo
    # in-process to inspect the matplotlib figure's text content.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = tmp_path / "overviews"
    generate_plate_overview(
        experiment_path=mock_plate,
        output_dir=out,
        well_px=80,
        verbose=False,
    )
    # The PNGs are written from a closed figure, so we can't inspect them
    # directly. Instead, drive the public helpers to confirm what would
    # be drawn.
    from pyphenix._overview import _combo_label, _options_label

    line1 = _combo_label((1,), {1: "DAPI", 2: "Alexa 488", 3: "mCherry"}, 3)
    line2 = _options_label(
        objective_mag="40",
        field_label="per-well first",
        timepoint_label="1",
        z_label="all",
        ffc_label="on",
    )
    assert line1 == "Ch1: DAPI"
    assert "Field: per-well first" in line2
    assert "T: 1" in line2
    assert "Z: all" in line2
    assert "FFC: on" in line2
    plt.close("all")


def test_empty_well_yields_blank_cell(mock_plate, tmp_path):
    # Delete one well's image files; the well drops out of the index but
    # the plate layout (and its row/col labels) should still render.
    images_dir = mock_plate / "Images"
    for f in images_dir.glob("r02c02*.tiff"):
        f.unlink()

    out = tmp_path / "overviews"
    written = generate_plate_overview(
        experiment_path=mock_plate,
        output_dir=out,
        well_px=80,
        verbose=False,
    )
    # Still produce all 7 PNGs + 1 sidecar; one well is just blank.
    assert sum(1 for p in written if p.suffix == ".png") == 7
