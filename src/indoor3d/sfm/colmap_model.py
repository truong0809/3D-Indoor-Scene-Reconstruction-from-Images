"""Đọc model COLMAP dạng nhị phân (cameras.bin, images.bin, points3D.bin).

Định dạng đối chiếu với mã nguồn COLMAP (src/colmap/scene/reconstruction_io_binary.cc,
kiểm tra ngày 09/10/2026). Các file rigs.bin / frames.bin của COLMAP >= 3.12 không cần
cho việc thống kê nên không đọc. points3D.bin có thể đọc đầy đủ và ghi lại (để lọc điểm).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INVALID_POINT3D_ID = 2**64 - 1

# model_id -> (tên, số tham số), theo src/colmap/sensor/models/*.h
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
    12: ("SIMPLE_DIVISION", 4),
    13: ("DIVISION", 5),
    14: ("SIMPLE_FISHEYE", 3),
    15: ("FISHEYE", 4),
    16: ("EUCM", 6),
    18: ("SKEWED_PINHOLE", 5),
}

_POINT2D_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("point3d_id", "<u8")])


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class Image:
    image_id: int
    name: str
    camera_id: int
    qvec: tuple[float, float, float, float]  # (w, x, y, z), cam_from_world
    tvec: tuple[float, float, float]
    num_points2d: int
    num_observations: int  # số điểm 2D có điểm 3D tương ứng


@dataclass
class PointsSummary:
    count: int
    mean_error: float | None
    median_error: float | None
    mean_track_length: float | None


def _read(handle, fmt: str):
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of COLMAP binary file")
    return struct.unpack(fmt, data)


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read(handle, "<IiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id {model_id} in {path}")
            name, num_params = CAMERA_MODELS[model_id]
            params = list(_read(handle, f"<{num_params}d"))
            cameras[camera_id] = Camera(camera_id, name, width, height, params)
    return cameras


def read_images_binary(path: Path) -> dict[int, Image]:
    images = {}
    with open(path, "rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(handle, "<I7dI")
            name_bytes = bytearray()
            while True:
                char = handle.read(1)
                if not char:
                    raise ValueError("unexpected end of file while reading image name")
                if char == b"\x00":
                    break
                name_bytes += char
            (num_points2d,) = _read(handle, "<Q")
            raw = handle.read(num_points2d * _POINT2D_DTYPE.itemsize)
            if len(raw) != num_points2d * _POINT2D_DTYPE.itemsize:
                raise ValueError("unexpected end of file while reading 2D points")
            points = np.frombuffer(raw, dtype=_POINT2D_DTYPE)
            observations = int(np.count_nonzero(points["point3d_id"] != INVALID_POINT3D_ID))
            images[image_id] = Image(image_id, name_bytes.decode("utf-8"), camera_id,
                                     (qw, qx, qy, qz), (tx, ty, tz), num_points2d, observations)
    return images


def read_points3d_summary(path: Path) -> PointsSummary:
    """Thống kê điểm 3D mà không giữ toàn bộ dữ liệu trong bộ nhớ."""
    errors = []
    track_lengths = []
    with open(path, "rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            # id (uint64), xyz (3 double), rgb (3 uint8), error (double), track length (uint64)
            _pid, _x, _y, _z, _r, _g, _b, error, track_length = _read(handle, "<Q3d3BdQ")
            handle.seek(8 * track_length, 1)  # each element: image_id (uint32) + point2D_idx (uint32)
            errors.append(error)
            track_lengths.append(track_length)
    if not errors:
        return PointsSummary(0, None, None, None)
    return PointsSummary(count, float(np.mean(errors)), float(np.median(errors)), float(np.mean(track_lengths)))


@dataclass
class Point3D:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: np.ndarray  # (n, 2) uint32: image_id, point2D_idx


_TRACK_DTYPE = np.dtype([("image_id", "<u4"), ("point2d_idx", "<u4")])


def read_points3d_binary(path: Path) -> list[Point3D]:
    points = []
    with open(path, "rb") as handle:
        (count,) = _read(handle, "<Q")
        for _ in range(count):
            point_id, x, y, z, r, g, b, error, track_length = _read(handle, "<Q3d3BdQ")
            raw = handle.read(track_length * _TRACK_DTYPE.itemsize)
            if len(raw) != track_length * _TRACK_DTYPE.itemsize:
                raise ValueError("unexpected end of file while reading a point track")
            track = np.frombuffer(raw, dtype=_TRACK_DTYPE)
            points.append(Point3D(point_id, (x, y, z), (r, g, b), error,
                                  np.stack([track["image_id"], track["point2d_idx"]], axis=1)))
    return points


def write_points3d_binary(path: Path, points: list[Point3D]) -> None:
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for point in points:
            track = np.asarray(point.track, dtype="<u4").reshape(-1, 2)
            handle.write(struct.pack("<Q3d3BdQ", point.point_id, *point.xyz, *point.rgb, point.error, len(track)))
            handle.write(track.tobytes())


def model_exists(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))
