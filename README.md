# phd_configuration

Shared configuration module for the PhD seismic processing projects.

This repository is intended to be used as a Git submodule named
`phd_configuration` inside the other project repositories, for example
`phd_data_acquisition`, `phd_preprocessing`, and `phd_visualization`.
Those projects import the same configuration and helper functions instead of
keeping separate local copies.

## Repository role

`phd_configuration` provides:

- A central `config.json` with data paths, acquisition thresholds,
  preprocessing parameters, EOD geometry, and study-area geometry.
- A runtime config loader in `config.py`.
- Shared utility functions in `utils.py` for JSON I/O, QC/run folders,
  stage tracking, coordinate conversion, ZNE-to-BOX rotation, and logging.

The module is deliberately small because it is a common dependency. Changes in
this repository can affect all projects that include it as a submodule.

## Files

| File | Purpose |
| --- | --- |
| `config.json` | Base configuration used by all projects. |
| `config.py` | Loads the base config or the active run config. |
| `utils.py` | Shared helpers used by acquisition, preprocessing, and visualization scripts. |
| `README.md` | Documentation for this shared module. |

## Use as a submodule

Add this repository inside another project with:

```bash
git submodule add https://github.com/iarvanitis69/phd_configuration.git phd_configuration
git submodule update --init --recursive
```

After cloning a project that already contains this submodule, initialize it with:

```bash
git submodule update --init --recursive
```

Update the submodule in a parent project with:

```bash
cd phd_configuration
git pull
cd ..
git add phd_configuration
git commit -m "Update phd_configuration submodule"
```

## Imports

Typical usage from a parent repository:

```python
from phd_configuration import config
from phd_configuration.utils import get_qc_folder_path, geo_to_cartesian_wrapper

logs_dir = config.get("LOGS_DIR")
qc_dir = get_qc_folder_path()
x, y, z = geo_to_cartesian_wrapper(lat, lon, depth_km)
```

The parent repository must be run from a location where Python can import the
`phd_configuration` directory. This is normally true when scripts are executed
from the parent repository root and the submodule path is `./phd_configuration`.

## Runtime configuration

`config.py` exposes the following public functions:

| Function | Description |
| --- | --- |
| `config.get(key, default=None)` | Read a config value from the active config. |
| `config.all()` | Return the active config dictionary. |
| `config.base_all()` | Return the base `config.json` dictionary from this repository. |
| `config.source_path()` | Return the path of the config file currently being used. |
| `config.activate(path)` | Manually switch the active config to another JSON file. |

By default, the active config is `phd_configuration/config.json`.

If the parent repository root contains `session_info.txt` and it stores a run
folder name, `config.py` checks:

```text
<LOGS_DIR>/<folder name>/config.json
```

If that file exists, it becomes the active config. This lets later pipeline
stages use the exact config snapshot that was written when the run started.

## Run-folder workflow

The run-folder helpers in `utils.py` coordinate pipeline stages through the
parent repository file:

```text
session_info.txt
```

Important helpers:

| Function | Description |
| --- | --- |
| `start_new_qc_folder()` | Creates a timestamped folder under `LOGS_DIR`, writes a run `config.json`, activates it, and stores the folder name in `session_info.txt`. |
| `start_or_resume_qc_folder()` | Resumes the folder in `session_info.txt`, or creates a new one if the file is empty. |
| `get_qc_folder_name()` | Returns the active run folder name, or raises an error if no run is active. |
| `get_qc_folder_path()` | Returns and creates the active run folder path under `LOGS_DIR`. |
| `resolve_stage_output_dir(output_dir=None)` | Resolves a stage output path and verifies it matches the active run. |
| `latest_qc_folder_name()` | Finds the latest run folder that contains a `config.json`. |
| `clear_session_info()` | Clears the active run and reactivates the base config. |

The intended pipeline behavior is:

1. The first stage starts or resumes a run.
2. Every later stage writes into the active run folder.
3. The final stage clears `session_info.txt` after successful completion.

## Stage tracking helpers

`utils.py` also includes small helpers for JSON tracking files:

| Function | Description |
| --- | --- |
| `load_json(path)` | Read JSON and return `{}` if the file is missing or invalid. |
| `save_json(path, data)` | Write formatted UTF-8 JSON. |
| `get_stage_tracking_path(output_dir, stage_name)` | Build `<output_dir>/<stage_name>.json`. |
| `get_tracking_stage(data, stage_name)` | Extract a stage dictionary from flat or nested tracking data. |
| `stage_tracking_complete(output_dir, stage_name)` | Check whether a stage tracking file reports a complete stage. |

## Geometry helpers

The coordinate helpers use the `EOD` section of `config.json`.

Important conventions:

- Geographic coordinates are `(latitude, longitude, depth_km)`.
- `depth_km` is positive downward.
- Local BOX/EOD `z` increases upward.
- The EOD origin is `bottom_SW`.
- The local X, Y, and Z axes are defined from the EOD reference corners.

Common helpers:

| Function | Description |
| --- | --- |
| `geo_to_cartesian_wrapper(lat, lon, depth)` | Convert geographic coordinates to local Cartesian coordinates using the configured EOD origin. |
| `compute_distance_and_angles_geo(...)` | Compute distance, azimuth, and polar angle between two geographic points. |
| `build_box_basis_from_config()` | Build ECEF-to-BOX and BOX-to-ECEF rotation matrices from `config.json`. |
| `rotate_vector_zne_to_box(...)` | Rotate one ZNE vector into BOX/EOD coordinates. |
| `rotate_vector_series_zne_to_box(...)` | Rotate vector arrays from ZNE into BOX/EOD coordinates. |
| `transform_traces_zne_to_box(...)` | Convert ObsPy Z/N/E traces into X/Y/Z BOX traces. |

## Logging helpers

Use `tee_stdout(logfile_path)` to mirror `print()` output both to the terminal
and to a log file:

```python
from phd_configuration.utils import tee_stdout

with tee_stdout("/path/to/stage.log"):
    print("This goes to stdout and the log file")
```

## Configuration keys

The base `config.json` currently contains:

- Data paths: `BASE_DIR`, `LOGS_DIR`
- Acquisition settings: `min_magnitude_acquisition`,
  `max_event_depth_acquisition`, `preset_on_data_acquisition`,
  `offset_on_data_acquisition`
- Preprocessing/QC settings: `cube_voxel_side_length`,
  `FIX_GLITCHES_WITH_INTERPOLATION`, `minSnr`, `minEventDuration`,
  `maxEventDuration`, `minDepth`, `maxDepth`, `low_frequency`,
  `high_frequency`
- Geometry sections: `EOD`, `STUDY_AREA`

When adding keys, keep the change backward-compatible where possible because
all parent projects may read this file through the same submodule.

## Development notes

- Use the `phd_conda_env_p10` interpreter/environment for this project.
- Keep this repository importable as a plain Python package directory.
- Do not write parent-project-specific code here unless it is genuinely shared.
- After changing `config.py`, `utils.py`, or `config.json`, update each parent
  repository's submodule pointer.
- To generate a PDF copy of this README:

```bash
pandoc README.md -o README.pdf \
  --pdf-engine=xelatex \
  --toc --toc-depth=3 \
  --number-sections \
  --highlight-style=tango \
  -V geometry:margin=0.8in \
  -V mainfont="DejaVu Serif" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue
```
