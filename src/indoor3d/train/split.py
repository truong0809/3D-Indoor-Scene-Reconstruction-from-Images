"""Chia ảnh train/test và ghi lại, để mọi chế độ huấn luyện đánh giá trên đúng cùng một tập ảnh.

Quy tắc mặc định (`every_n`) giống hệt hai bản cài đặt đang dùng (đã đối chiếu mã nguồn):

- Inria @ 54c035f, `scene/dataset_readers.py`: sắp xếp tên ảnh COLMAP, ảnh thứ i là test khi i % 8 == 0.
- gsplat @ 6e8c837, `examples/datasets/colmap.py`: sắp xếp tên ảnh, tập "val" gồm các ảnh i % test_every == 0.

Quy tắc thứ hai (`test_list`): danh sách ảnh test cho trước, mỗi dòng một tên ảnh (cùng định dạng
`sparse/0/test.txt` của Inria). Dùng khi có đoạn quay kiểm tra riêng (docs/data/capture_protocol.md);
áp dụng cho gsplat qua `indoor3d.train.gsplat_launcher`. Khi đó đoạn kiểm tra nằm trong cùng model COLMAP,
nên `make_train_view` tạo bản dữ liệu mà points3D.bin chỉ giữ các điểm đủ ảnh train quan sát: điểm chỉ
được dựng nhờ ảnh test không được dùng để khởi tạo Gaussian.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from indoor3d.sfm.colmap_model import read_images_binary, read_points3d_binary, write_points3d_binary


@dataclass
class Split:
    rule: str  # "every_n" | "test_list"
    test_every: int | None
    train: list[str]
    test: list[str]

    def digest(self) -> str:
        """SHA-256 của hai danh sách (đã sắp xếp); hai lượt chạy cùng digest là cùng tập train/test."""
        canonical = json.dumps({"train": sorted(self.train), "test": sorted(self.test)}, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "test_every": self.test_every,
            "num_train": len(self.train),
            "num_test": len(self.test),
            "sha256": self.digest(),
            "train": self.train,
            "test": self.test,
        }


def every_nth_split(names: list[str], test_every: int) -> Split:
    if test_every < 2:
        raise ValueError("test_every must be >= 2")
    ordered = sorted(names)
    test = [name for index, name in enumerate(ordered) if index % test_every == 0]
    train = [name for index, name in enumerate(ordered) if index % test_every != 0]
    if not train:
        raise ValueError("no training images left after the split")
    return Split("every_n", test_every, train, test)


def list_split(names: list[str], test_names: list[str]) -> Split:
    ordered = sorted(names)
    known = set(ordered)
    unknown = sorted(set(test_names) - known)
    if unknown:
        raise ValueError(f"{len(unknown)} test image(s) are not registered in the COLMAP model, "
                         f"e.g. {unknown[:3]}")
    wanted = set(test_names)
    test = [name for name in ordered if name in wanted]
    train = [name for name in ordered if name not in wanted]
    if not test:
        raise ValueError("the test list is empty")
    if not train:
        raise ValueError("no training images left after removing the test list")
    return Split("test_list", None, train, test)


def read_test_list(path: Path) -> list[str]:
    """Một tên ảnh mỗi dòng; bỏ dòng trống và dòng bắt đầu bằng '#'."""
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate names in {path}")
    return names


def registered_image_names(scene_dir: Path) -> list[str]:
    """Tên các ảnh đã đăng ký trong `<cảnh>/sparse/0/images.bin`."""
    images = read_images_binary(Path(scene_dir) / "sparse" / "0" / "images.bin")
    return sorted(image.name for image in images.values())


def make_split(scene_dir: Path, test_every: int = 8, test_list: Path | None = None) -> Split:
    names = registered_image_names(scene_dir)
    if test_list is not None:
        return list_split(names, read_test_list(test_list))
    return every_nth_split(names, test_every)


def write_split(split: Split, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_split(path: Path) -> Split:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Split(data["rule"], data.get("test_every"), list(data["train"]), list(data["test"]))


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        target.symlink_to(source, target_is_directory=source.is_dir())
        return "symlink"
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return "copy"


def make_train_view(scene_dir: Path, view_dir: Path, split: Split, min_train_views: int = 2) -> dict:
    """Thư mục dữ liệu cho gsplat khi tập test là danh sách cho trước.

    `images/`, `cameras.bin`, `images.bin` giữ nguyên (liên kết tới cảnh gốc; ảnh test vẫn cần pose để
    render khi đánh giá). `points3D.bin` chỉ giữ các điểm được ít nhất `min_train_views` ảnh train khác
    nhau quan sát, tức là dựng được mà không cần ảnh test.
    """
    model = Path(scene_dir) / "sparse" / "0"
    images = read_images_binary(model / "images.bin")
    train_names = set(split.train)
    train_ids = {image_id for image_id, image in images.items() if image.name in train_names}
    points = read_points3d_binary(model / "points3D.bin")
    kept = [point for point in points
            if len({int(image_id) for image_id in point.track[:, 0]} & train_ids) >= min_train_views]
    if not kept:
        raise ValueError("no SfM point is observed by enough training images")

    out_model = Path(view_dir) / "sparse" / "0"
    out_model.mkdir(parents=True, exist_ok=True)
    methods = {_link_or_copy((Path(scene_dir) / "images").resolve(), Path(view_dir) / "images")}
    for name in ("cameras.bin", "images.bin"):
        methods.add(_link_or_copy((model / name).resolve(), out_model / name))
    write_points3d_binary(out_model / "points3D.bin", kept)
    return {"data_dir": str(view_dir), "min_train_views": min_train_views, "points_total": len(points),
            "points_kept": len(kept), "link_method": sorted(methods)}
