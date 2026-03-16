# pyphenix

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/pyphenix.svg?color=green)](https://github.com/ferrinm/pyphenix/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyphenix.svg?color=green)](https://pypi.org/project/pyphenix)
[![Python Version](https://img.shields.io/pypi/pyversions/pyphenix.svg?color=green)](https://python.org)
[![tests](https://github.com/ferrinm/pyphenix/workflows/tests/badge.svg)](https://github.com/ferrinm/pyphenix/actions)
[![codecov](https://codecov.io/gh/ferrinm/pyphenix/branch/main/graph/badge.svg)](https://codecov.io/gh/ferrinm/pyphenix)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/pyphenix)](https://napari-hub.org/plugins/pyphenix)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A data loader and widget to visualize high-dimensional data collected on the Opera Phenix

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template] (None).

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

### Using `napari`'s graphical plugin manager
Users can entirely avoid using a command prompt by using the package manager
built in to the bundled napari app.

Follow the [installation
instructions](https://napari.org/stable/tutorials/fundamentals/installation_bundle_conda.html)
to download and install the correct version of the bundled napari app for your
operating systems (MacOS, Windows, or Linux).

Then following the [instructions for installing
plugins](https://napari.org/stable/plugins/start_using_plugins/finding_and_installing_plugins.html#installing-plugins-with-napari),
search for `PyPhenix` in the Plugin Manager search bar and click "Install".

### Using Python package installer (`pip`)
#### Reader only (no GUI, no napari required)

If you only need the `OperaPhenixReader` — for example on a server or in a
headless analysis pipeline — install the base package:

```
pip install pyphenix
```

This installs only `numpy` and `Pillow`. napari, Qt, and pandas are **not**
required and will **not** be installed.

```python
from pyphenix import OperaPhenixReader

reader = OperaPhenixReader("/path/to/experiment")
data, metadata = reader.read_data(row="D", column=4)
```

#### Full GUI install (napari widget)

To use the interactive napari widget, install with the `napari` extra:

```
pip install "pyphenix[napari]"
```

This additionally installs `napari`, `qtpy`, and `pandas`.

```python
from pyphenix import launch_viewer

viewer, widget = launch_viewer("/path/to/experiment")
```

Alternatively, if napari is already installed in your environment, the base
install is sufficient — pyphenix will detect napari at import time and make
the widget available automatically:

```
pip install pyphenix
```

#### Latest development version

```
pip install git+https://github.com/ferrinm/pyphenix.git
```


### Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"pyphenix" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/ferrinm/pyphenix/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
