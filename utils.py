import os
import json
import sys
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from obspy import Trace

from phd_configuration import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_INFO_FILE = "session_info.txt"
_QC_FOLDER_NAME = None


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


# Coordination conversions -----------------------------------------------------------

def _get_reference_points():
    eod = config.get("EOD")
    corners = eod["corners"]

    required = ["bottom_SW", "bottom_SE", "bottom_NW", "top_SW"]
    for name in required:
        if name not in corners:
            raise ValueError(f"Λείπει το corner '{name}' από το EOD.")

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
    Μετατροπή geo -> local Cartesian με origin το bottom_SW του EOD.

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
    active_dir = get_qc_folder_path()
    if output_dir is None:
        return active_dir
    if os.path.isabs(output_dir):
        resolved = os.path.abspath(output_dir)
    elif os.path.dirname(str(output_dir)):
        resolved = os.path.abspath(output_dir)
    else:
        resolved = os.path.join(config.get("LOGS_DIR"), str(output_dir))
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


# BOX / EOD basis + ZNE -> EOD rotation ----------------------------------------------

def build_box_basis_from_corners(origin_geo, x_geo, y_geo, z_geo):
    """
    Χτίζει basis του EOD/BOX στο ECEF.

    Είσοδος:
        origin_geo = (lat, lon, depth)
        x_geo      = geo σημείο πάνω στον άξονα X του BOX
        y_geo      = geo σημείο πάνω στον άξονα Y του BOX
        z_geo      = geo σημείο πάνω στον άξονα Z του BOX

    Έξοδος:
        R_ecef_to_box : 3x3 πίνακας που παίρνει vector στο ECEF και το γράφει σε BOX
        R_box_to_ecef : 3x3 πίνακας που παίρνει vector στο BOX και το γράφει σε ECEF
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
    Περιστρέφει ένα διάνυσμα από τοπικό ZNE -> BOX/EOD.
    """
    z_hat, n_hat, e_hat = build_local_zne_basis_in_ecef(lat_deg, lon_deg)

    v_ecef = float(z) * z_hat + float(n) * n_hat + float(e) * e_hat
    v_box = np.asarray(R_ecef_to_box, dtype=float) @ v_ecef

    return v_box.astype(float)


def rotate_vector_series_zne_to_box(vector_z, vector_n, vector_e, lat_deg, lon_deg, R_ecef_to_box):
    """
    Περιστρέφει σειρά διανυσμάτων από ZNE -> BOX/EOD.
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


class TeeLogger:
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.logfile_path = os.path.abspath(logfile_path)
        self.logfile = open(logfile_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

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
