"""Ghi model COLMAP nhị phân tổng hợp cho test (cùng định dạng với COLMAP)."""

import struct
from pathlib import Path

INVALID_POINT3D_ID = 2**64 - 1


def write_cameras(path: Path, cameras):
    """cameras: list (camera_id, model_id, width, height, params)."""
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera_id, model_id, width, height, params in cameras:
            handle.write(struct.pack("<IiQQ", camera_id, model_id, width, height))
            handle.write(struct.pack(f"<{len(params)}d", *params))


def write_images(path: Path, images):
    """images: list dict(image_id, name, camera_id, qvec, tvec, points2d=[(x, y, point3d_id)])."""
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image in images:
            handle.write(struct.pack("<I7dI", image["image_id"], *image["qvec"], *image["tvec"],
                                     image["camera_id"]))
            handle.write(image["name"].encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", len(image["points2d"])))
            for x, y, point_id in image["points2d"]:
                handle.write(struct.pack("<ddQ", x, y, point_id))


def write_points3d(path: Path, points):
    """points: list dict(id, xyz, rgb, error, track=[(image_id, point2d_idx)])."""
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for point in points:
            handle.write(struct.pack("<Q3d3Bd", point["id"], *point["xyz"], *point["rgb"], point["error"]))
            handle.write(struct.pack("<Q", len(point["track"])))
            for image_id, index in point["track"]:
                handle.write(struct.pack("<II", image_id, index))


def make_model(model_dir: Path, names, camera_model_id=1, error=0.8, points_per_image=3):
    """Model nhỏ: mỗi ảnh có `points_per_image` điểm 3D (track dài 2) và một keypoint không khớp."""
    model_dir.mkdir(parents=True, exist_ok=True)
    params = {1: [500.0, 500.0, 320.0, 240.0],
              4: [500.0, 500.0, 320.0, 240.0, 0.01, -0.02, 0.001, 0.002]}[camera_model_id]
    write_cameras(model_dir / "cameras.bin", [(1, camera_model_id, 640, 480, params)])
    images, points, point_id = [], [], 1
    for image_id, name in enumerate(names, start=1):
        points2d = []
        for k in range(points_per_image):
            points2d.append((10.0 * k, 5.0 * k, point_id))
            points.append({"id": point_id, "xyz": (float(k), float(image_id), 1.0), "rgb": (10, 20, 30),
                           "error": error, "track": [(image_id, k), (image_id, k)]})
            point_id += 1
        points2d.append((1.0, 1.0, INVALID_POINT3D_ID))
        images.append({"image_id": image_id, "name": name, "camera_id": 1, "qvec": (1.0, 0.0, 0.0, 0.0),
                       "tvec": (0.0, 0.0, float(image_id)), "points2d": points2d})
    write_images(model_dir / "images.bin", images)
    write_points3d(model_dir / "points3D.bin", points)
    return model_dir
