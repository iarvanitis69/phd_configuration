import os
import json
import sys
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from obspy import UTCDateTime, Trace
from obspy.clients.syngine import Client

from phd_configuration import config

_SYNGINE_CLIENT = Client()
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


def is_station_excluded(json_paths, event_name, station_name):
    """
    Επιστρέφει True αν σε οποιοδήποτε JSON αρχείο
    υπάρχει:
        data[event_name][station_name]
    Αγνοεί το key 'COUNT'.
    """
    for path in json_paths:
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if event_name not in data:
            continue

        event_block = data[event_name]
        if not isinstance(event_block, dict):
            continue

        if station_name in event_block:
            return True

    return False


def is_channel_excluded(excluded_json_paths, event_name, station_name, channel_name):
    """
    Επιστρέφει True αν το συγκεκριμένο event/station/channel είναι excluded
    σε κάποιο από τα exclusion json files.
    """
    for path in excluded_json_paths:
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        event_block = data.get(event_name, {})
        if not isinstance(event_block, dict):
            continue

        station_block = event_block.get(station_name, {})
        if not isinstance(station_block, dict):
            continue

        channels_block = station_block.get("channels", {})
        if isinstance(channels_block, dict) and channel_name in channels_block:
            return True

        # Optional support for JSON files without a "channels" wrapper.
        if channel_name in station_block and isinstance(station_block[channel_name], dict):
            return True

    return False


# DENSITY relation functions ----------------------------------------------------

def transform(points_to_transform, voxel_density_pairs, alpha=1.0, bins=None):
    points_to_transform = np.asarray(points_to_transform, dtype=float)

    if points_to_transform.ndim != 2 or points_to_transform.shape[1] != 3:
        raise ValueError("points_to_transform must have shape (M, 3)")

    if not voxel_density_pairs:
        raise ValueError("voxel_density_pairs is empty")

    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    voxel_centers = []
    densities_gaussian = []

    for center, density in voxel_density_pairs:
        center = np.asarray(center, dtype=float)

        if center.shape != (3,):
            raise ValueError("Each voxel center must be shape (3,)")

        voxel_centers.append(center)
        densities_gaussian.append(float(density))

    voxel_centers = np.asarray(voxel_centers, dtype=float)
    densities_gaussian = np.asarray(densities_gaussian, dtype=float)

    if bins is None:
        bins = max(10, int(np.cbrt(len(voxel_centers))))

    weights = np.power(np.maximum(densities_gaussian, 1e-15), alpha)

    x_ref = voxel_centers[:, 0]
    y_ref = voxel_centers[:, 1]
    z_ref = voxel_centers[:, 2]

    def compute_cdf(values, w, bins_):
        hist, edges = np.histogram(values, bins=bins_, weights=w)

        if np.sum(hist) == 0:
            hist = np.ones_like(hist, dtype=float)

        cdf = np.cumsum(hist).astype(float)
        cdf /= cdf[-1]

        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, cdf

    x_bins, Fx_vals = compute_cdf(x_ref, weights, bins)
    y_bins, Fy_vals = compute_cdf(y_ref, weights, bins)
    z_bins, Fz_vals = compute_cdf(z_ref, weights, bins)

    Fx = lambda v: np.interp(v, x_bins, Fx_vals, left=Fx_vals[0], right=Fx_vals[-1])
    Fy = lambda v: np.interp(v, y_bins, Fy_vals, left=Fy_vals[0], right=Fy_vals[-1])
    Fz = lambda v: np.interp(v, z_bins, Fz_vals, left=Fz_vals[0], right=Fz_vals[-1])

    x_min, x_max = x_ref.min(), x_ref.max()
    y_min, y_max = y_ref.min(), y_ref.max()
    z_min, z_max = z_ref.min(), z_ref.max()

    x = points_to_transform[:, 0]
    y = points_to_transform[:, 1]
    z = points_to_transform[:, 2]

    x_prime = x_min + (x_max - x_min) * Fx(x)
    y_prime = y_min + (y_max - y_min) * Fy(y)
    z_prime = z_min + (z_max - z_min) * Fz(z)

    transformed_points = np.column_stack([x_prime, y_prime, z_prime])

    result_dict = {}

    for i in range(len(points_to_transform)):
        original = tuple(points_to_transform[i])
        transformed = tuple(transformed_points[i])
        result_dict[original] = transformed

    transform_data = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "x_bins": x_bins,
        "Fx_vals": Fx_vals,
        "y_bins": y_bins,
        "Fy_vals": Fy_vals,
        "z_bins": z_bins,
        "Fz_vals": Fz_vals,
        "alpha": alpha,
        "bins": bins,
    }

    return result_dict, transform_data


def inverse_transform(transformed_points, transform_data):
    transformed_points = np.asarray(transformed_points, dtype=float)

    if transformed_points.ndim != 2 or transformed_points.shape[1] != 3:
        raise ValueError("transformed_points must have shape (N, 3)")

    x_min = transform_data["x_min"]
    x_max = transform_data["x_max"]
    y_min = transform_data["y_min"]
    y_max = transform_data["y_max"]
    z_min = transform_data["z_min"]
    z_max = transform_data["z_max"]

    x_bins = transform_data["x_bins"]
    Fx_vals = transform_data["Fx_vals"]
    y_bins = transform_data["y_bins"]
    Fy_vals = transform_data["Fy_vals"]
    z_bins = transform_data["z_bins"]
    Fz_vals = transform_data["Fz_vals"]

    xp = transformed_points[:, 0]
    yp = transformed_points[:, 1]
    zp = transformed_points[:, 2]

    if x_max > x_min:
        ux = (xp - x_min) / (x_max - x_min)
    else:
        ux = np.zeros_like(xp)

    if y_max > y_min:
        uy = (yp - y_min) / (y_max - y_min)
    else:
        uy = np.zeros_like(yp)

    if z_max > z_min:
        uz = (zp - z_min) / (z_max - z_min)
    else:
        uz = np.zeros_like(zp)

    ux = np.clip(ux, 0.0, 1.0)
    uy = np.clip(uy, 0.0, 1.0)
    uz = np.clip(uz, 0.0, 1.0)

    x_back = np.interp(ux, Fx_vals, x_bins, left=x_bins[0], right=x_bins[-1])
    y_back = np.interp(uy, Fy_vals, y_bins, left=y_bins[0], right=y_bins[-1])
    z_back = np.interp(uz, Fz_vals, z_bins, left=z_bins[0], right=z_bins[-1])

    original_points = np.column_stack([x_back, y_back, z_back])

    return original_points


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


def cartesian_to_geo(
    lat_origin, lon_origin, depth_origin,
    lat_x_axis, lon_x_axis, depth_x_axis,
    lat_y_axis, lon_y_axis, depth_y_axis,
    lat_z_axis, lon_z_axis, depth_z_axis,
    x_point, y_point, z_point
):
    """
    Μετατροπή local Cartesian -> geo.

    Σύμβαση:
      - depth_km θετικό προς τα κάτω
      - local z αυξάνει προς τα πάνω

    Επιστρέφει:
      lat_point, lon_point, depth_point
    """
    def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x * x + y * y + z * z)
        if r == 0:
            raise ValueError("Μηδενική ακτίνα στο cartesian_to_spherical")

        lat_rad = np.arcsin(z / r)
        lon_rad = np.arctan2(y, x)
        depth_km = 6371.0 - r

        lat = np.degrees(lat_rad)
        lon = np.degrees(lon_rad)

        return lat, lon, depth_km

    origin_cart = _geo_to_ecef(lat_origin, lon_origin, depth_origin)
    x_axis_cart = _geo_to_ecef(lat_x_axis, lon_x_axis, depth_x_axis)
    y_axis_cart = _geo_to_ecef(lat_y_axis, lon_y_axis, depth_y_axis)
    z_axis_cart = _geo_to_ecef(lat_z_axis, lon_z_axis, depth_z_axis)

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

    point_cart = (
        origin_cart
        + float(x_point) * x_axis_unit
        + float(y_point) * y_axis_unit
        + float(z_point) * z_axis_unit
    )

    lat_point, lon_point, depth_point = cartesian_to_spherical(
        point_cart[0],
        point_cart[1],
        point_cart[2]
    )

    return float(lat_point), float(lon_point), float(depth_point)


def cartesian_to_geo_wrapper(x_point, y_point, z_point):
    origin, _, _, _ = _get_reference_points()

    earth_radius_km = 6371.0
    lat_origin, lon_origin, depth_origin = _corner_geo(origin)
    o_local = np.asarray(_corner_local(origin), dtype=float)

    lat_origin_rad = np.radians(lat_origin)
    lat_point = lat_origin + np.degrees((float(y_point) - o_local[1]) / earth_radius_km)
    lon_point = lon_origin + np.degrees((float(x_point) - o_local[0]) / (earth_radius_km * np.cos(lat_origin_rad)))
    depth_point = float(depth_origin) - (float(z_point) - o_local[2])

    return float(lat_point), float(lon_point), float(depth_point)


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


def compute_distance_and_angles_cartesian(x1, y1, z1, x2, y2, z2):
    """
    Υπολογίζει απόσταση, azimuth και polar angle
    από δύο σημεία σε local Cartesian coordinates.

    Point 1: αρχικό σημείο (π.χ. event)
    Point 2: τελικό σημείο (π.χ. voxel ή station)

    Επιστρέφει:
        distance     : ευκλείδεια απόσταση
        azimuth_deg  : γωνία στο επίπεδο XY, σε μοίρες [0, 360)
        polar_deg    : γωνία ως προς τον +Z άξονα, σε μοίρες [0, 180]
    """
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    dz = float(z2 - z1)

    distance = np.sqrt(dx * dx + dy * dy + dz * dz)

    azimuth_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

    if distance == 0.0:
        polar_deg = 0.0
    else:
        cos_theta = np.clip(dz / distance, -1.0, 1.0)
        polar_deg = np.degrees(np.arccos(cos_theta))

    return float(distance), float(azimuth_deg), float(polar_deg)


def get_eod_grid_size():
    eod = config.get("EOD")
    if not isinstance(eod, dict):
        raise RuntimeError("EOD not found in config")

    grid = eod.get("grid_shape")
    if not isinstance(grid, dict):
        raise RuntimeError("EOD.grid_shape not found in config")

    return f"{int(grid['n_eod'])}x{int(grid['m_eod'])}x{int(grid['l_eod'])}"


def get_qc_folder_name():
    selected_voxels_top_k = config.get("selected_voxels_top_k")

    folder_name = (
        f"({get_eod_grid_size()})_"
        f"(minSnr_{config.get('minSnr')})"
        f"_(minEventDuration_{config.get('minEventDuration')})"
        f"_(maxEventDuration_{config.get('maxEventDuration')})"
        f"_(minDepth_{config.get('minDepth')})"
        f"_(maxDepth_{config.get('maxDepth')})"
        f"_(low_frequency_{config.get('low_frequency')})"
        f"_(high_frequency_{config.get('high_frequency')})"
        f"_(sigma_km_{config.get('sigma_km')})"
        f"_(selected_voxels_top_k_{selected_voxels_top_k})"
    )
    return folder_name


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
    os.makedirs(resolved, exist_ok=True)
    return resolved


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


def transform_traces_xyz_to_transformed_box(
    trX,
    trY,
    trZ,
    event_cart,
    station_cart,
    event_cart_T,
    station_cart_T,
    eps=1e-12,
):
    """
    Μετασχηματίζει τριάδα XYZ traces στο T coordinate space με τοπική
    component-wise κλίμακα από το event-station vector.
    """
    event_cart = np.asarray(event_cart, dtype=float)
    station_cart = np.asarray(station_cart, dtype=float)
    event_cart_T = np.asarray(event_cart_T, dtype=float)
    station_cart_T = np.asarray(station_cart_T, dtype=float)

    if not (
        event_cart.shape == station_cart.shape == event_cart_T.shape == station_cart_T.shape == (3,)
    ):
        raise ValueError("event/station cart and cart_T values must have shape (3,)")

    x = np.asarray(trX.data, dtype=float)
    y = np.asarray(trY.data, dtype=float)
    z = np.asarray(trZ.data, dtype=float)

    if not (len(x) == len(y) == len(z)):
        raise ValueError("Τα traces X, Y, Z πρέπει να έχουν το ίδιο μήκος")

    delta = station_cart - event_cart
    delta_T = station_cart_T - event_cart_T
    scale = np.ones(3, dtype=float)
    valid = np.abs(delta) > float(eps)
    scale[valid] = delta_T[valid] / delta[valid]

    def mk(data, ref, ch):
        tr = Trace(data=np.asarray(data, dtype=np.float32))
        tr.stats = ref.stats.copy()
        tr.stats.channel = ch
        return tr

    trX_T = mk(x * scale[0], trX, "HHX")
    trY_T = mk(y * scale[1], trY, "HHY")
    trZ_T = mk(z * scale[2], trZ, "HHZ")

    return trX_T, trY_T, trZ_T


# Syngine -----------------------------------------------------------------------

def create_green_zne_for_station_event(
    *,
    source_lat,
    source_lon,
    source_depth_m,
    station_lat,
    station_lon,
    origin_time_utc,
    model="ak135f_5s",
    dt=0.0125,
    pre_event_time=0.0,
    end_time=11.0,
    sourcedoublecouple=None,
    components="ZNE",
    units="displacement",
    format="miniseed",
):
    """
    Κατεβάζει Green function από Syngine σε ZNE
    και την επιστρέφει μόνο στη μνήμη, χωρίς save στο disk.
    """
    client = Client()

    if sourcedoublecouple is None:
        sourcedoublecouple = [0, 90, 0, 1e19]

    st = client.get_waveforms(
        model=model,
        receiverlatitude=station_lat,
        receiverlongitude=station_lon,
        sourcelatitude=source_lat,
        sourcelongitude=source_lon,
        sourcedepthinmeters=source_depth_m,
        sourcedoublecouple=sourcedoublecouple,
        origintime=UTCDateTime(origin_time_utc),
        starttime=-pre_event_time,
        endtime=end_time - pre_event_time,
        dt=dt,
        components=components,
        units=units,
        format=format,
    )

    trZ = None
    trN = None
    trE = None

    for tr in st:
        comp = tr.stats.channel[-1].upper()
        if comp == "Z":
            trZ = tr
        elif comp == "N":
            trN = tr
        elif comp == "E":
            trE = tr

    if trZ is None or trN is None or trE is None:
        raise RuntimeError("Syngine did not return full ZNE set")

    return trZ, trN, trE


def create_transform_green_for_station_event_in_memory(
    *,
    source_lat,
    source_lon,
    source_depth_m,
    station_lat,
    station_lon,
    origin_time_utc,
    R_ecef_to_box,
    station,
    event_id,
    model="ak135f_5s",
    dt=0.014285,
    pre_event_time=0.0,
    end_time=None,
    sourcedoublecouple=None,
    components="ZNE",
    units="displacement",
    format="miniseed"
):
    """
    1. Κατεβάζει Green ZNE από Syngine
    2. Κάνει transform σε BOX XYZ
    3. ΔΕΝ αποθηκεύει τίποτα στο δίσκο (in-memory)

    Επιστρέφει:
        trX, trY, trZbox
    """
    trZ, trN, trE = create_green_zne_for_station_event(
        source_lat=source_lat,
        source_lon=source_lon,
        source_depth_m=source_depth_m,
        station_lat=station_lat,
        station_lon=station_lon,
        origin_time_utc=origin_time_utc,
        model=model,
        dt=dt,
        pre_event_time=pre_event_time,
        end_time=end_time,
        sourcedoublecouple=sourcedoublecouple,
        components=components,
        units=units,
        format=format
    )

    trX, trY, trZbox = transform_traces_zne_to_box(
        trZ=trZ,
        trN=trN,
        trE=trE,
        station_lat=station_lat,
        station_lon=station_lon,
        R_ecef_to_box=R_ecef_to_box,
    )

    return trX, trY, trZbox


def fetch_green_window_zne(
    *,
    source_lat,
    source_lon,
    source_depth_m,
    station_lat,
    station_lon,
    origin_time_utc,
    t_target,
    model="ak135f_5s",
    dt=0.25,
    window_samples_each_side=10,
    sourcedoublecouple=None,
    components="ZNE",
    units="displacement",
    format="miniseed",
):
    """
    Κατεβάζει μικρό χρονικό παράθυρο Green function γύρω από t_target
    """
    client = Client()

    if sourcedoublecouple is None:
        sourcedoublecouple = [0, 90, 0, 1e19]

    half_window = window_samples_each_side * dt
    start_offset = max(0.0, t_target - half_window)
    end_offset = t_target + half_window

    st = client.get_waveforms(
        model=model,
        receiverlatitude=station_lat,
        receiverlongitude=station_lon,
        sourcelatitude=source_lat,
        sourcelongitude=source_lon,
        sourcedepthinmeters=source_depth_m,
        sourcedoublecouple=sourcedoublecouple,
        origintime=UTCDateTime(origin_time_utc),
        starttime=start_offset,
        endtime=end_offset,
        dt=dt,
        components=components,
        units=units,
        format=format,
    )

    trZ = trN = trE = None

    for tr in st:
        comp = tr.stats.channel[-1].upper()
        if comp == "Z":
            trZ = tr
        elif comp == "N":
            trN = tr
        elif comp == "E":
            trE = tr

    if trZ is None or trN is None or trE is None:
        raise RuntimeError("Syngine did not return full ZNE set")

    return trZ, trN, trE


def interpolate_trace_at_time(tr, t_target, origin_time_utc):
    """
    Linear interpolation της τιμής του trace στη χρονική στιγμή t_target
    """
    origin = UTCDateTime(origin_time_utc)
    t_abs = origin + t_target

    start = tr.stats.starttime
    dt = tr.stats.delta
    n = tr.stats.npts

    x = np.array(
        [(start + i * dt).timestamp for i in range(n)],
        dtype=np.float64
    )
    y = tr.data.astype(np.float64)

    t_req = t_abs.timestamp

    if t_req < x[0] or t_req > x[-1]:
        raise ValueError("t_target outside trace range")

    return float(np.interp(t_req, x, y))


def get_green_at_time(
    source_lat,
    source_lon,
    source_depth_m,
    station_lat,
    station_lon,
    origin_time_utc,
    R_ecef_to_box,
    station,
    event_id,
    t_target,
    model="ak135f_5s",
    dt=0.25,
    window_samples_each_side=10,
    sourcedoublecouple=None,
    components="ZNE",
    units="displacement",
    format="miniseed",
):
    """
    Κατεβάζει μόνο μικρό χρονικό παράθυρο γύρω από το t_target
    και επιστρέφει απευθείας numpy arrays (X, Y, Zbox).
    """
    if origin_time_utc is None:
        raise ValueError(f"Missing origin_time_utc for event_id={event_id}")

    if not isinstance(origin_time_utc, UTCDateTime):
        origin_time_utc = UTCDateTime(origin_time_utc)

    half_window_sec = float(window_samples_each_side) * float(dt)

    start_offset = float(t_target) - half_window_sec
    end_offset = float(t_target) + half_window_sec

    if start_offset < 0:
        start_offset = 0.0

    try:
        st = _SYNGINE_CLIENT.get_waveforms(
            model=model,
            sourcelatitude=float(source_lat),
            sourcelongitude=float(source_lon),
            sourcedepthinmeters=float(source_depth_m),
            receiverlatitude=float(station_lat),
            receiverlongitude=float(station_lon),
            origintime=origin_time_utc,
            starttime=start_offset,
            endtime=end_offset,
            dt=float(dt),
            components=components,
            units=units,
            format=format,
            sourcedoublecouple=sourcedoublecouple,
        )
    except Exception as e:
        print(
            f"ERROR in get_green_at_time: "
            f"event_id={event_id} station={station} t_target={t_target} "
            f"start_offset={start_offset} end_offset={end_offset} error={e}"
        )
        return None, None, None

    if st is None or len(st) == 0:
        print(
            f"WARNING: empty stream returned from syngine "
            f"for event_id={event_id} station={station}"
        )
        return None, None, None

    trZ = None
    trN = None
    trE = None

    for tr in st:
        ch = str(getattr(tr.stats, "channel", "")).upper()

        if ch.endswith("Z"):
            trZ = tr
        elif ch.endswith("N"):
            trN = tr
        elif ch.endswith("E"):
            trE = tr

    if trZ is None or trN is None or trE is None:
        print(
            f"WARNING: missing components in syngine response "
            f"for event_id={event_id} station={station}"
        )
        return None, None, None

    X, Y, Zbox = rotate_vector_series_zne_to_box(
        vector_z=np.asarray(trZ.data, dtype=np.float32),
        vector_n=np.asarray(trN.data, dtype=np.float32),
        vector_e=np.asarray(trE.data, dtype=np.float32),
        lat_deg=station_lat,
        lon_deg=station_lon,
        R_ecef_to_box=R_ecef_to_box
    )

    return X, Y, Zbox


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
