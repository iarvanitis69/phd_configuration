import os
import json
import sys
import inspect
import re
from contextlib import contextmanager
from typing import Any, Mapping
from datetime import datetime
from time import perf_counter

import numpy as np
from obspy import Trace

from phd_configuration import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_INFO_FILE = "session_info.txt"
_QC_FOLDER_NAME = None
_LINE_PREFIX_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+\S+(?:\.py)?(?:\s|$)")
_EMBEDDED_LINE_PREFIX_RE = re.compile(r"\[(\d+/\d+)\]\s+(\S+(?:\.py)?):?\s*")


def sanitize_event_name(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").replace(":", "_")


def event_stem(event_name: str) -> str:
    stem = sanitize_event_name(event_name)
    if stem.endswith(".json"):
        return stem[:-5]
    return stem


def get_event_dir(output_dir: str, event_name: str) -> str:
    return os.path.join(output_dir, event_stem(event_name))


def get_event_json_path(output_dir: str, event_name: str) -> str:
    return os.path.join(get_event_dir(output_dir, event_name), f"{event_stem(event_name)}.json")


def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _process_now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def process_timer(script_name: str):
    start_time = perf_counter()
    print(f"[1/1] {script_name} [PY START] {script_name} at {_process_now_text()}", flush=True)
    try:
        yield
    except Exception:
        print(f"[1/1] {script_name} [PY FAILED] {script_name} at {_process_now_text()} after {perf_counter() - start_time:.2f}s", flush=True)
        raise
    print(f"[1/1] {script_name} [PY DONE] {script_name} at {_process_now_text()} after {perf_counter() - start_time:.2f}s", flush=True)


def timed_process(script_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with process_timer(script_name):
                return func(*args, **kwargs)

        wrapper.__name__ = getattr(func, "__name__", "wrapper")
        wrapper.__doc__ = getattr(func, "__doc__", None)
        wrapper.__module__ = getattr(func, "__module__", None)
        return wrapper

    return decorator


def get_stage_tracking_path(output_dir, stage_name):
    return os.path.join(output_dir, f"{stage_name}.json")


def get_tracking_stage(data, stage_name):
    if not isinstance(data, dict):
        return None
    if "stage" not in data:
        return data
    if data.get("stage") == stage_name:
        return data

    stages = data.get("stage", {})
    if isinstance(stages, dict):
        stage = stages.get(stage_name)
        if isinstance(stage, dict):
            return stage

    return None


# Coordinate conversions -------------------------------------------------------------

def _get_reference_points():
    study_area = config.get("STUDY_AREA")
    if not isinstance(study_area, dict):
        raise ValueError("STUDY_AREA is missing from config.")
    corners = study_area["corners"]

    required = ["bottom_SW", "bottom_SE", "bottom_NW", "top_SW"]
    for name in required:
        if name not in corners:
            raise ValueError(f"Missing STUDY_AREA corner '{name}'.")

    return (
        corners["bottom_SW"],  # origin
        corners["bottom_SE"],  # x axis
        corners["bottom_NW"],  # y axis
        corners["top_SW"],     # z axis
    )


def _corner_geo(corner):
    lat, lon, depth = corner["geo"]
    return float(lat), float(lon), float(depth)


def _corner_local(corner):
    x, y, z = corner["local"]
    return float(x), float(y), float(z)


def _geo_to_ecef(lat_deg, lon_deg, depth_km):
    """
    Geo -> ECEF.

    Σύμβαση:
      - depth_km θετικό προς τα κάτω
    """
    lat_rad = np.radians(float(lat_deg))
    lon_rad = np.radians(float(lon_deg))
    r = 6371.0 - float(depth_km)

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return np.array([x, y, z], dtype=float)


def _normalize_vector(v, name="vector"):
    v = np.asarray(v, dtype=float).reshape(3,)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError(f"Zero-length {name}")
    return v / n


def geo_to_cartesian(
    lat_origin, lon_origin, depth_origin,
    lat_x_axis, lon_x_axis, depth_x_axis,
    lat_y_axis, lon_y_axis, depth_y_axis,
    lat_z_axis, lon_z_axis, depth_z_axis,
    lat_point, lon_point, depth_point
):
    """
    Convert geo -> local Cartesian in the coordinate system defined by STUDY_AREA.

    Σύμβαση:
      - depth_km θετικό προς τα κάτω
      - local z αυξάνει προς τα πάνω
    """
    origin_cart = _geo_to_ecef(lat_origin, lon_origin, depth_origin)
    x_axis_cart = _geo_to_ecef(lat_x_axis, lon_x_axis, depth_x_axis)
    y_axis_cart = _geo_to_ecef(lat_y_axis, lon_y_axis, depth_y_axis)
    z_axis_cart = _geo_to_ecef(lat_z_axis, lon_z_axis, depth_z_axis)
    point_cart = _geo_to_ecef(lat_point, lon_point, depth_point)

    x_axis_vec = x_axis_cart - origin_cart
    y_axis_vec = y_axis_cart - origin_cart
    z_axis_vec = z_axis_cart - origin_cart

    x_axis_unit = _normalize_vector(x_axis_vec, "x_axis_vec")

    y_axis_vec = y_axis_vec - np.dot(y_axis_vec, x_axis_unit) * x_axis_unit
    y_axis_unit = _normalize_vector(y_axis_vec, "y_axis_vec")

    z_axis_vec = (
        z_axis_vec
        - np.dot(z_axis_vec, x_axis_unit) * x_axis_unit
        - np.dot(z_axis_vec, y_axis_unit) * y_axis_unit
    )
    z_axis_unit = _normalize_vector(z_axis_vec, "z_axis_vec")

    dx = point_cart - origin_cart

    x_new = float(np.dot(dx, x_axis_unit))
    y_new = float(np.dot(dx, y_axis_unit))
    z_new = float(np.dot(dx, z_axis_unit))

    return x_new, y_new, z_new


def geo_to_cartesian_wrapper(lat_point, lon_point, depth_point):
    origin, _, _, _ = _get_reference_points()

    earth_radius_km = 6371.0
    lat_origin, lon_origin, depth_origin = _corner_geo(origin)
    o_local = np.asarray(_corner_local(origin), dtype=float)

    lat_origin_rad = np.radians(lat_origin)
    x = o_local[0] + earth_radius_km * np.cos(lat_origin_rad) * np.radians(float(lon_point) - lon_origin)
    y = o_local[1] + earth_radius_km * np.radians(float(lat_point) - lat_origin)
    z = o_local[2] + float(depth_origin) - float(depth_point)

    return float(x), float(y), float(z)


def compute_distance_and_angles_geo(lat1, lon1, depth1, lat2, lon2, depth2):
    """
    lat/lon σε μοίρες
    depth σε km, θετικό προς τα κάτω

    Point 1: event
    Point 2: station
    """
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius = 6371.0
    horizontal_distance = earth_radius * c

    dz = depth2 - depth1
    distance = np.sqrt(horizontal_distance ** 2 + dz ** 2)

    azimuth = np.arctan2(
        np.sin(dlon) * np.cos(lat2_rad),
        np.cos(lat1_rad) * np.sin(lat2_rad)
        - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    )
    azimuth = (np.degrees(azimuth) + 360.0) % 360.0

    if distance == 0:
        polar_angle = 0.0
    else:
        cos_theta = np.clip(dz / distance, -1.0, 1.0)
        polar_angle = np.degrees(np.arccos(cos_theta))

    return float(distance), float(azimuth), float(polar_angle)


def get_session_info_path():
    path = os.path.join(REPO_ROOT, SESSION_INFO_FILE)
    if not os.path.exists(path):
        open(path, "w", encoding="utf-8").close()
    return path


def make_qc_folder_name():
    return datetime.now().strftime("%Y_%m_%d_%H:%M:%S")


def write_session_folder_name(folder_name):
    with open(get_session_info_path(), "w", encoding="utf-8") as f:
        f.write(str(folder_name).strip())


def read_session_folder_name():
    with open(get_session_info_path(), "r", encoding="utf-8") as f:
        folder_name = f.read().strip()
    return folder_name or None


def clear_session_info():
    global _QC_FOLDER_NAME
    _QC_FOLDER_NAME = None
    with open(get_session_info_path(), "w", encoding="utf-8") as f:
        f.write("")
    config.activate(config.CONFIG_PATH)


def start_new_qc_folder():
    global _QC_FOLDER_NAME
    folder_name = make_qc_folder_name()
    folder_path = os.path.join(config.base_all().get("LOGS_DIR"), folder_name)
    os.makedirs(folder_path, exist_ok=True)

    run_config = dict(config.base_all())
    run_config["RUN_OUTPUT_FOLDER_NAME"] = folder_name
    run_config["RUN_OUTPUT_FOLDER_PATH"] = folder_path
    run_config["RUN_CONFIG_PATH"] = os.path.join(folder_path, "config.json")
    save_json(run_config["RUN_CONFIG_PATH"], run_config)
    config.activate(run_config["RUN_CONFIG_PATH"])

    _QC_FOLDER_NAME = folder_name
    write_session_folder_name(folder_name)
    return folder_name


def latest_qc_folder_name():
    logs_dir = config.base_all().get("LOGS_DIR")
    if not logs_dir or not os.path.isdir(logs_dir):
        return None

    candidates = []
    for name in os.listdir(logs_dir):
        folder_path = os.path.join(logs_dir, name)
        run_config_path = os.path.join(folder_path, "config.json")
        if os.path.isdir(folder_path) and os.path.exists(run_config_path):
            candidates.append(name)

    return sorted(candidates)[-1] if candidates else None


def resume_qc_folder(folder_name):
    global _QC_FOLDER_NAME
    folder_path = os.path.join(config.base_all().get("LOGS_DIR"), folder_name)
    run_config_path = os.path.join(folder_path, "config.json")
    if not os.path.isdir(folder_path):
        raise RuntimeError(f"Active voxel run folder from session_info.txt does not exist: {folder_path}")
    if not os.path.exists(run_config_path):
        raise RuntimeError(f"Active voxel run config does not exist: {run_config_path}")
    config.activate(run_config_path)
    _QC_FOLDER_NAME = folder_name
    write_session_folder_name(folder_name)
    return folder_name


def start_or_resume_qc_folder():
    folder_name = read_session_folder_name()
    if folder_name:
        return resume_qc_folder(folder_name)
    return start_new_qc_folder()


def get_qc_folder_name():
    global _QC_FOLDER_NAME
    if _QC_FOLDER_NAME:
        return _QC_FOLDER_NAME

    folder_name = read_session_folder_name()
    if folder_name is None:
        raise RuntimeError(
            "No active voxel run exists: session_info.txt is empty. "
            "Run create_voxel_info.py / compute_voxel_info first."
        )
    return resume_qc_folder(folder_name)


def get_qc_folder_path():
    folder_name = get_qc_folder_name()
    full_path = os.path.join(config.get("LOGS_DIR"), folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def resolve_stage_output_dir(output_dir=None):
    if output_dir is None:
        return get_qc_folder_path()

    if os.path.isabs(output_dir):
        resolved = os.path.abspath(output_dir)
    elif os.path.dirname(str(output_dir)):
        resolved = os.path.abspath(output_dir)
    else:
        resolved = os.path.join(config.get("LOGS_DIR"), str(output_dir))

    logs_dir = os.path.abspath(config.base_all().get("LOGS_DIR"))
    folder_name = os.path.basename(os.path.abspath(resolved))
    run_config_path = os.path.join(resolved, "config.json")
    if os.path.exists(run_config_path) and os.path.dirname(os.path.abspath(resolved)) == logs_dir:
        resume_qc_folder(folder_name)
        os.makedirs(resolved, exist_ok=True)
        return resolved

    active_dir = get_qc_folder_path()
    if os.path.abspath(resolved) != os.path.abspath(active_dir):
        raise RuntimeError(
            "Stage output folder does not match active voxel run from session_info.txt: "
            f"{resolved} != {active_dir}"
        )
    os.makedirs(resolved, exist_ok=True)
    return resolved


def stage_tracking_complete(output_dir, stage_name):
    path = get_stage_tracking_path(output_dir, stage_name)
    data = load_json(path)
    stage = get_tracking_stage(data, stage_name)
    if not isinstance(stage, dict):
        return False

    required_keys = ("included", "excluded", "nof_included_channels", "nof_excluded_channels")
    if not all(key in stage for key in required_keys):
        return False

    if not isinstance(stage.get("included"), dict) or not isinstance(stage.get("excluded"), dict):
        return False

    return stage.get("status") == "complete"


# STUDY_AREA basis + ZNE -> STUDY_AREA rotation --------------------------------------

def build_box_basis_from_corners(origin_geo, x_geo, y_geo, z_geo):
    """
    Build the STUDY_AREA basis in ECEF.

    Είσοδος:
        origin_geo = (lat, lon, depth)
        x_geo      = geo point on the STUDY_AREA X axis
        y_geo      = geo point on the STUDY_AREA Y axis
        z_geo      = geo point on the STUDY_AREA Z axis

    Έξοδος:
        R_ecef_to_box : 3x3 matrix that maps an ECEF vector to STUDY_AREA coordinates
        R_box_to_ecef : 3x3 matrix that maps a STUDY_AREA vector to ECEF
    """
    O = _geo_to_ecef(*origin_geo)
    Px = _geo_to_ecef(*x_geo)
    Py = _geo_to_ecef(*y_geo)
    Pz = _geo_to_ecef(*z_geo)

    ex = Px - O
    ey = Py - O
    ez = Pz - O

    ex = _normalize_vector(ex, "BOX X axis")

    ey = ey - np.dot(ey, ex) * ex
    ey = _normalize_vector(ey, "BOX Y axis")

    ez = ez - np.dot(ez, ex) * ex - np.dot(ez, ey) * ey
    ez = _normalize_vector(ez, "BOX Z axis")

    R_ecef_to_box = np.vstack([ex, ey, ez])
    R_box_to_ecef = R_ecef_to_box.T

    return R_ecef_to_box, R_box_to_ecef


def build_box_basis_from_config():
    """
    Παίρνει τα 4 reference corners από το config και επιστρέφει
    τους πίνακες περιστροφής ECEF <-> BOX.
    """
    origin, x_axis, y_axis, z_axis = _get_reference_points()

    origin_geo = _corner_geo(origin)
    x_geo = _corner_geo(x_axis)
    y_geo = _corner_geo(y_axis)
    z_geo = _corner_geo(z_axis)

    return build_box_basis_from_corners(
        origin_geo=origin_geo,
        x_geo=x_geo,
        y_geo=y_geo,
        z_geo=z_geo
    )


def build_local_zne_basis_in_ecef(lat_deg, lon_deg):
    """
    Επιστρέφει τις μοναδιαίες διευθύνσεις Z, N, E στο ECEF
    για το συγκεκριμένο γεωγραφικό σημείο.

    Σύμβαση:
      Z = up
      N = north
      E = east
    """
    lat_rad = np.radians(float(lat_deg))
    lon_rad = np.radians(float(lon_deg))

    e_hat = np.array([
        -np.sin(lon_rad),
         np.cos(lon_rad),
         0.0
    ], dtype=float)

    n_hat = np.array([
        -np.sin(lat_rad) * np.cos(lon_rad),
        -np.sin(lat_rad) * np.sin(lon_rad),
         np.cos(lat_rad)
    ], dtype=float)

    z_hat = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ], dtype=float)

    e_hat = _normalize_vector(e_hat, "E basis")
    n_hat = _normalize_vector(n_hat, "N basis")
    z_hat = _normalize_vector(z_hat, "Z basis")

    return z_hat, n_hat, e_hat


def rotate_vector_zne_to_box(z, n, e, lat_deg, lon_deg, R_ecef_to_box):
    """
    Rotate a vector from local ZNE to STUDY_AREA coordinates.
    """
    z_hat, n_hat, e_hat = build_local_zne_basis_in_ecef(lat_deg, lon_deg)

    v_ecef = float(z) * z_hat + float(n) * n_hat + float(e) * e_hat
    v_box = np.asarray(R_ecef_to_box, dtype=float) @ v_ecef

    return v_box.astype(float)


def rotate_vector_series_zne_to_box(vector_z, vector_n, vector_e, lat_deg, lon_deg, R_ecef_to_box):
    """
    Rotate a vector series from ZNE to STUDY_AREA coordinates.
    """
    vector_z = np.asarray(vector_z, dtype=float)
    vector_n = np.asarray(vector_n, dtype=float)
    vector_e = np.asarray(vector_e, dtype=float)

    if not (vector_z.shape == vector_n.shape == vector_e.shape):
        raise ValueError("vector_z, vector_n, vector_e must have the same shape")

    z_hat, n_hat, e_hat = build_local_zne_basis_in_ecef(lat_deg, lon_deg)
    R = np.asarray(R_ecef_to_box, dtype=float)

    v_ecef = (
        vector_z[:, None] * z_hat[None, :]
        + vector_n[:, None] * n_hat[None, :]
        + vector_e[:, None] * e_hat[None, :]
    )

    v_box = v_ecef @ R.T

    vector_x = v_box[:, 0].astype(np.float32)
    vector_y = v_box[:, 1].astype(np.float32)
    vector_z_box = v_box[:, 2].astype(np.float32)

    return vector_x, vector_y, vector_z_box


def transform_traces_zne_to_box(trZ, trN, trE, station_lat, station_lon, R_ecef_to_box):
    """
    Μετατρέπει traces από ZNE -> BOX XYZ.

    Σύμβαση:
      - trZ = vertical, positive up
      - trN = north
      - trE = east
    """
    Z = trZ.data.astype(float)
    N = trN.data.astype(float)
    E = trE.data.astype(float)

    if not (len(Z) == len(N) == len(E)):
        raise ValueError("Τα traces Z, N, E πρέπει να έχουν το ίδιο μήκος")

    X, Y, Zbox = rotate_vector_series_zne_to_box(
        vector_z=Z,
        vector_n=N,
        vector_e=E,
        lat_deg=station_lat,
        lon_deg=station_lon,
        R_ecef_to_box=R_ecef_to_box
    )

    def mk(data, ref, ch):
        tr = Trace(data=np.asarray(data, dtype=np.float32))
        tr.stats = ref.stats.copy()
        tr.stats.channel = ch
        return tr

    trX = mk(X, trZ, "HHX")
    trY = mk(Y, trZ, "HHY")
    trZbox = mk(Zbox, trZ, "HHZ")

    return trX, trY, trZbox


def _caller_script_name(default: str = "process.py") -> str:
    this_file = os.path.abspath(__file__)
    for frame in inspect.stack()[2:]:
        path = os.path.abspath(frame.filename)
        try:
            in_repo = os.path.commonpath([REPO_ROOT, path]) == REPO_ROOT
        except ValueError:
            in_repo = False
        if (
            path != this_file
            and path.endswith(".py")
            and in_repo
        ):
            return os.path.basename(path)
    return default


def _script_name_from_logfile(logfile_path: str | None, default: str = "process.py") -> str:
    if logfile_path:
        base = os.path.basename(str(logfile_path))
        stem, _ = os.path.splitext(base)
        if stem:
            return f"{stem}.py"
    return default


class LinePrefixWriter:
    def __init__(self, stream, script_name: str | None = None):
        self.stream = stream
        self.script_name = script_name or _caller_script_name()
        self._at_line_start = True

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def _prefix(self, line: str) -> str:
        if _LINE_PREFIX_RE.match(line):
            return line
        match = _EMBEDDED_LINE_PREFIX_RE.search(line)
        if match:
            before = line[: match.start()]
            after = line[match.end() :]
            cleaned = f"{before}{after}".lstrip()
            return f"[{match.group(1)}] {match.group(2)} {cleaned}"
        return f"[0/0] {self.script_name} {line}"

    def write(self, message):
        text = str(message)
        if not text:
            return 0
        parts = text.splitlines(keepends=True)
        for part in parts:
            if self._at_line_start:
                part = self._prefix(part)
            self.stream.write(part)
            self._at_line_start = part.endswith("\n")
        return len(text)

    def flush(self):
        self.stream.flush()

    def close(self):
        close = getattr(self.stream, "close", None)
        if callable(close):
            close()


def install_stdout_prefix(script_name: str | None = None):
    if isinstance(sys.stdout, LinePrefixWriter):
        return sys.stdout
    sys.stdout = LinePrefixWriter(sys.stdout, script_name or _caller_script_name())
    return sys.stdout


class TeeLogger:
    def __init__(self, logfile_path, script_name: str | None = None):
        self.terminal = sys.stdout
        self.script_name = script_name or _caller_script_name(_script_name_from_logfile(logfile_path))
        self.logfile_path = os.path.abspath(logfile_path)
        self.logfile = open(logfile_path, "a", encoding="utf-8")
        self._at_line_start = True

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    def _prefix(self, line: str) -> str:
        if _LINE_PREFIX_RE.match(line):
            return line
        match = _EMBEDDED_LINE_PREFIX_RE.search(line)
        if match:
            before = line[: match.start()]
            after = line[match.end() :]
            cleaned = f"{before}{after}".lstrip()
            return f"[{match.group(1)}] {match.group(2)} {cleaned}"
        return f"[0/0] {self.script_name} {line}"

    def write(self, message):
        text = str(message)
        if not text:
            return 0
        for part in text.splitlines(keepends=True):
            if self._at_line_start:
                part = self._prefix(part)
            self.terminal.write(part)
            self.logfile.write(part)
            self._at_line_start = part.endswith("\n")
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


@contextmanager
def tee_stdout(logfile_path):
    logfile_path = os.path.abspath(logfile_path)
    if getattr(sys.stdout, "logfile_path", None) == logfile_path:
        yield
        return

    os.makedirs(os.path.dirname(logfile_path) or ".", exist_ok=True)
    original_stdout = sys.stdout
    logger = TeeLogger(logfile_path)
    sys.stdout = logger
    try:
        yield
    finally:
        sys.stdout = original_stdout
        logger.close()


# Fourier NPZ helpers ---------------------------------------------------------------

FFT_BASE_FIELDS = ("freqs", "real", "imag", "magnitude", "phase", "sampling_rate", "original_n_samples")
FFT_AC_FIELDS = FFT_BASE_FIELDS + ("correction_filter",)


def fft_payload_from_signal(values: np.ndarray, sampling_rate: float, window_seconds: float) -> dict[str, np.ndarray]:
    sr = float(sampling_rate)
    if not np.isfinite(sr) or sr <= 0.0:
        raise ValueError(f"invalid_sampling_rate:{sampling_rate}")
    npts = max(2, int(round(float(window_seconds) * sr)))
    if npts % 2:
        npts += 1
    window = np.zeros(npts, dtype=np.float32)
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    n = min(npts, int(source.size))
    if n > 0:
        window[:n] = source[:n]
    spectrum = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(npts, d=1.0 / sr)
    return fft_payload_from_complex(spectrum, freqs, sr, npts)


def fft_payload_from_complex(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    sampling_rate: float,
    original_n_samples: int,
    correction_filter: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    complex_values = np.asarray(spectrum, dtype=np.complex128).reshape(-1)
    freq_values = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if complex_values.shape != freq_values.shape:
        raise ValueError("fft_complex_frequency_shape_mismatch")
    sr = float(sampling_rate)
    n_samples = int(original_n_samples)
    if not np.isfinite(sr) or sr <= 0.0:
        raise ValueError("invalid_sampling_rate")
    if n_samples <= 0:
        raise ValueError("invalid_original_n_samples")

    payload: dict[str, np.ndarray] = {
        "freqs": freq_values.astype(np.float32),
        "real": complex_values.real.astype(np.float32),
        "imag": complex_values.imag.astype(np.float32),
        "magnitude": np.abs(complex_values).astype(np.float32),
        "phase": np.angle(complex_values).astype(np.float32),
        "sampling_rate": np.asarray(sr, dtype=np.float32),
        "original_n_samples": np.asarray(n_samples, dtype=np.int64),
    }
    if correction_filter is not None:
        filt = np.asarray(correction_filter, dtype=np.float32).reshape(-1)
        if filt.shape != freq_values.shape:
            raise ValueError("fft_correction_filter_shape_mismatch")
        payload["correction_filter"] = filt
    return payload


def _validate_fft_payload(payload: Mapping[str, Any], require_correction_filter: bool = False) -> dict[str, np.ndarray]:
    required = list(FFT_BASE_FIELDS)
    if require_correction_filter:
        required.append("correction_filter")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"fft_npz_missing_fields:{','.join(missing)}")

    freqs = np.asarray(payload["freqs"], dtype=np.float32).reshape(-1)
    if freqs.size == 0:
        raise ValueError("empty_frequency_axis")
    arrays: dict[str, np.ndarray] = {"freqs": freqs}
    for field in ("real", "imag", "magnitude", "phase"):
        values = np.asarray(payload[field], dtype=np.float32).reshape(-1)
        if values.shape != freqs.shape:
            raise ValueError(f"fft_{field}_shape_mismatch")
        arrays[field] = values
    if "correction_filter" in payload:
        filt = np.asarray(payload["correction_filter"], dtype=np.float32).reshape(-1)
        if filt.shape != freqs.shape:
            raise ValueError("fft_correction_filter_shape_mismatch")
        arrays["correction_filter"] = filt

    sampling_rate = float(np.asarray(payload["sampling_rate"]))
    original_n_samples = int(np.asarray(payload["original_n_samples"]))
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("invalid_sampling_rate")
    if original_n_samples <= 0:
        raise ValueError("invalid_original_n_samples")
    arrays["sampling_rate"] = np.asarray(sampling_rate, dtype=np.float32)
    arrays["original_n_samples"] = np.asarray(original_n_samples, dtype=np.int64)
    return arrays


def write_fft_npz(path: str, payload: Mapping[str, Any]) -> None:
    output_path = os.path.abspath(str(path))
    arrays = _validate_fft_payload(payload, require_correction_filter=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, output_path)


def read_fft_npz(path: str, require_correction_filter: bool = False) -> dict[str, np.ndarray]:
    input_path = os.path.abspath(str(path))
    with np.load(input_path, allow_pickle=False) as data:
        payload = {field: np.asarray(data[field]) for field in data.files}
    return _validate_fft_payload(payload, require_correction_filter=require_correction_filter)


def fft_npz_file_valid(path: Any, expected_suffix: str, require_correction_filter: bool = False) -> bool:
    if not isinstance(path, str) or not os.path.basename(path).lower().endswith(expected_suffix.lower()):
        return False
    if not os.path.exists(path):
        return False
    try:
        read_fft_npz(path, require_correction_filter=require_correction_filter)
    except Exception:
        return False
    return True


def fft_plot_arrays(
    path: str,
    value: str = "magnitude",
    energy: bool = False,
    require_correction_filter: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    payload = read_fft_npz(path, require_correction_filter=require_correction_filter)
    freqs = np.asarray(payload["freqs"], dtype=np.float64).reshape(-1)
    if energy:
        magnitude = np.asarray(payload["magnitude"], dtype=np.float64).reshape(-1)
        return freqs, magnitude * magnitude, "Energy (|U(f)|^2)"

    field = str(value).strip().lower()
    if field not in payload:
        available = ", ".join(sorted(k for k, v in payload.items() if np.asarray(v).ndim > 0))
        raise ValueError(f"unknown_fft_plot_value:{field}; available={available}")
    values = np.asarray(payload[field], dtype=np.float64).reshape(-1)
    if values.shape != freqs.shape:
        raise ValueError(f"fft_plot_value_shape_mismatch:{field}")
    return freqs, values, field


def plot_fft_npz_spectrum(
    path: str,
    output_path: str | None = None,
    value: str = "magnitude",
    energy: bool = False,
    require_correction_filter: bool = False,
    title: str | None = None,
    xlim_hz: float | None = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_path = os.path.abspath(str(path))
    freqs, y_values, ylabel = fft_plot_arrays(
        input_path,
        value=value,
        energy=energy,
        require_correction_filter=require_correction_filter,
    )
    if output_path is None:
        suffix = "energy" if energy else str(value).strip().lower()
        output_path = os.path.splitext(input_path)[0] + f"_{suffix}_spectrum.png"
    output_path = os.path.abspath(str(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(freqs, y_values, linewidth=1.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or os.path.basename(input_path))
    ax.grid(True, alpha=0.3)
    if xlim_hz is not None:
        limit = float(xlim_hz)
        if np.isfinite(limit) and limit > 0.0:
            ax.set_xlim(0.0, limit)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
