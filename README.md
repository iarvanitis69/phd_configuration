# phd_configuration

Shared configuration module for the PhD seismic processing projects.

This repository is intended to be used as a Git submodule named
`phd_configuration` inside the other project repositories, for example
`phd_data_acquisition`, `phd_preprocessing`, and `phd_visualization`.
Those projects import the same configuration and helper functions instead of
keeping separate local copies.

## Repository role

`phd_configuration` provides:

- A shared `config.yaml` containing base settings for the consuming projects.
- A generic config loader in `config_loader.py`.
- Shared utility functions in `utils.py` for plain JSON I/O, stage tracking,
  coordinate conversion, ZNE-to-BOX rotation, FFT file formats and logging.

This package is independent of the preprocessing project. `utils.load_json`
and `utils.save_json` use Python's standard `json` library and preserve the
supplied structure: they do not compact station paths, expand channel records
or replace legacy identifiers. Preprocessing-specific manifest handling lives
in that project's runtime and is explicitly selected by its callers.

The FFT bundle helpers validate their own file-format labels and schemas locally.
They support full or partial native NZE, legacy XYZ and LQT bundles, reject mixed coordinate frames,
and retain per-channel timing and amplitude metadata without importing a
preprocessing runtime. No preprocessing module is required to import or use
these shared JSON/FFT helpers.

The module is deliberately small because it is a common dependency. Changes in
this repository can affect all projects that include it as a submodule.

## Files

| File | Purpose |
| --- | --- |
| `config.yaml` | Shared base settings for acquisition, preprocessing and visualization. |
| `config_loader.py` | Loads the shared base YAML configuration or an explicitly selected YAML/legacy JSON file. |
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
from phd_configuration import config_loader as config
from phd_configuration.utils import geo_to_cartesian_wrapper

logs_dir = config.get("LOGS_DIR")
x, y, z = geo_to_cartesian_wrapper(lat, lon, depth_km)
```

The parent repository must be run from a location where Python can import the
`phd_configuration` directory. This is normally true when scripts are executed
from the parent repository root and the submodule path is `./phd_configuration`.

## Runtime configuration

`config_loader.py` exposes the following public functions:

| Function | Description |
| --- | --- |
| `config.get(key, default=None)` | Read a config value from the active config. |
| `config.all()` | Return the active config dictionary. |
| `config.base_all()` | Return the shared base `config.yaml` dictionary. |
| `config.source_path()` | Return the path of the config file currently being used. |
| `config.activate(path)` | Manually switch the active config to another YAML file or a legacy JSON snapshot. |

The base configuration is the shared `config.yaml` beside `config_loader.py`.
`config.CONFIG_PATH` resolves that file relative to the loader itself, so the
caller project and process working directory do not select a different base
file. An unrelated `config.yaml` in a parent project or working directory is
ignored. A missing shared configuration is an explicit file error.

The loader reads the base file until a caller explicitly selects another YAML
file with `config.activate(path)`. It does not discover sessions, create run
folders or inspect preprocessing state. A project can explicitly activate a
shared configuration file used by other projects; `base_all()` continues to
read the shared `CONFIG_PATH`.

YAML is read using PyYAML's `safe_load`; consuming environments need `pyyaml`.
All configuration files must contain a mapping at the root. The migration from
`config.json` to `config.yaml` preserves all existing keys, values and types.
Comments beginning with `#` are supported. The supplied YAML documents every
parameter with a short Greek description, with units where applicable, including
velocity-model layers and nested study-area coordinates. Explicitly activated `.json` files
still use the standard JSON parser so existing run snapshots, including values
such as `1e-6`, remain compatible. New preprocessing run snapshots are YAML;
preprocessing alone owns the choice of run snapshot and session handling.

Preprocessing owns its session/run lifecycle in its own `stage_runtime.py`.
That runtime selects saved run configurations, creates snapshots and clears
session state. The run-folder and session helpers have been removed from this
shared package; preprocessing callers import them directly from the runtime.
Other projects using the loader do not inherit preprocessing run selection.

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

The coordinate helpers use the `STUDY_AREA` section of `config.yaml`.

Important conventions:

- Geographic coordinates are `(latitude, longitude, depth_km)`.
- `depth_km` is positive downward.
- Local BOX/`STUDY_AREA` `z` increases upward.
- The `STUDY_AREA` origin is `bottom_SW`.
- The local X, Y, and Z axes are defined from the `STUDY_AREA` reference corners.

Common helpers:

| Function | Description |
| --- | --- |
| `geo_to_cartesian_wrapper(lat, lon, depth)` | Convert geographic coordinates to local Cartesian coordinates using the configured `STUDY_AREA` origin. |
| `compute_distance_and_angles_geo(...)` | Compute distance, azimuth, and polar angle between two geographic points. |
| `build_box_basis_from_config()` | Build ECEF-to-BOX and BOX-to-ECEF rotation matrices from `config.yaml`. |
| `rotate_vector_zne_to_box(...)` | Rotate one ZNE vector into BOX/`STUDY_AREA` coordinates. |
| `rotate_vector_series_zne_to_box(...)` | Rotate vector arrays from ZNE into BOX/`STUDY_AREA` coordinates. |
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


`velocity_model` supplies the preprocessing depth-layer velocities:

```json
"velocity_model": "velocity_model_valianatos",
"velocity_model_valianatos": {
  "depth_bin_edges_km": [0.0, 1.0, 3.0, 8.0, 12.0, 16.0, 18.0, 22.0, 26.0],
  "Vp": [4.0, 4.3, 4.8, 5.2, 5.5, 5.8, 6.1, 6.6],
  "Vp_per_Vs": 1.777
}
```

Depth edges are in km and must be finite and strictly increasing. There must
be one finite, positive P-wave velocity (km/s) per interval in `Vp`.
Preprocessing also accepts the older speed-array key in existing snapshots,
while new output model metadata use `Vp`. Preprocessing reads the
active run’s selection and resolves the named table from the snapshot or
shared config. The active selection is `velocity_model_valianatos`; the
`velocity_model_papadimitriou_2015` table remains available for explicit selection.
Legacy inline model objects remain supported. If an older snapshot lacks the
selection key, it uses the shared selection. Optional `Vp_per_Vs` must be finite
and greater than one and is retained in output model metadata. An invalid explicit model produces an error. Validation and numerical
lookup remain in preprocessing; the shared loader simply reads the JSON.

The shared `config.yaml` contains settings used by the consuming projects,
including data paths, acquisition thresholds, preprocessing SNR/filter/polarization
settings and study-area geometry. Each project reads the keys it needs; the
loader does not hardcode those keys. Shared helpers read the keys required by
their documented operations, such as `LOGS_DIR` for paths and `STUDY_AREA` for
geometry.

The `station_s_*` settings define preprocessing's S-arrival plausibility gates:
minimum/maximum apparent velocity (km/s), minimum/maximum S-to-P travel-time
ratio, minimum S–P gap and travel-time margin (seconds). Bounds and input
validation are applied in `boundaries.py`. Only a triggered/AIC S candidate that
passes the checks is stored; otherwise its channel has `s_arrival="NOT_AVAILABLE"`
and an explanatory reason. Station P is the earliest native-channel `p_arrival`
and is recorded as `station_p_arrival`. These additions introduce no pipeline step.

`station_p_min_rms_snr` has been removed: preprocessing's initial P picker records
RMS ratios only as diagnostics and ignores that obsolete key in older snapshots.
The native NZE SNR stage uses `minSnr` to flag each input component, while retaining
all three waveforms for covariance and LQT. `lqt_snr_threshold_components` is
obsolete in the reordered pipeline and is ignored in older snapshots.

`cut_pre_arrival_seconds=0.2` selects the preprocessing cut margin before the
earliest native N/E/Z arrival; all output components use one common UTC window.
`filter_padding_seconds=3.0` sets the minimum reflection duration at each end
of preprocessing's zero-phase ObsPy band-pass. The filter also requires at
least three periods at `low_frequency`, then removes the reflected samples.
The signal length and timing are preserved.
The current filter band is `low_frequency=0.5` to `high_frequency=30` Hz.
Preprocessing’s existing verification stage reads saved Fourier coefficients.
Its `fourier_phase_*` settings control amplitude floors, the spectral SNR
threshold/metric, optional unit phasors and `fourier_phase_fnn_require_snr`.
Without a reliable noise spectrum, passing bins are labelled
`AMPLITUDE_ONLY_NO_SNR`; this is distinct from a measured SNR or a phase
reliability claim. Requiring spectral SNR for FNN can later be enabled by config.
Shared FFT helpers retain scalar `spectral_window_energy`, `spectral_units` and
`spectral_power_rule` metadata alongside existing timing/amplitude metadata.
These fields support consistent one-sided PSD comparisons without another FFT.
 Existing run snapshots retain their
explicit settings, including an older cut margin, until updated.


Preprocessing snapshots the shared configuration for new runs. Existing runs
retain their own saved settings until explicitly changed or a new run is started.
Session discovery and snapshot selection belong exclusively to the preprocessing
runtime, not this shared package.

## Development notes

- Use the `phd_conda_env_p10` interpreter/environment for this project.
- Keep this repository importable as a plain Python package directory.
- Do not write parent-project-specific code here unless it is genuinely shared.
- After changing `config.yaml`, `config_loader.py` or `utils.py`, update each parent
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
