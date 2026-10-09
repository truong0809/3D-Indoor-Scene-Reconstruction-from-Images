"""Test chia train/test: phải trùng quy tắc của Inria (54c035f) và gsplat (6e8c837).

Dữ liệu là tên ảnh tổng hợp để kiểm tra logic.
"""

import pytest

from colmap_writer import make_model, write_points3d
from indoor3d.sfm.colmap_model import read_points3d_binary, write_points3d_binary
from indoor3d.train.split import (
    every_nth_split,
    list_split,
    load_split,
    make_split,
    make_train_view,
    read_test_list,
    write_split,
)


def names(n):
    return [f"{i:05d}.jpg" for i in range(1, n + 1)]


def inria_llffhold_test(cam_names, llffhold=8):
    """Chép đúng logic của scene/dataset_readers.py (Inria @ 54c035f)."""
    cam_names = sorted(cam_names)
    return [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]


def gsplat_val(image_names, test_every=8):
    """Chép đúng logic của examples/datasets/colmap.py (gsplat @ 6e8c837): sắp xếp rồi lấy i % test_every == 0."""
    image_names = sorted(image_names)
    return [name for index, name in enumerate(image_names) if index % test_every == 0]


def test_every_nth_matches_inria_and_gsplat():
    shuffled = names(17)[::-1]
    split = every_nth_split(shuffled, 8)
    assert split.test == inria_llffhold_test(shuffled) == gsplat_val(shuffled)
    assert split.test == ["00001.jpg", "00009.jpg", "00017.jpg"]
    assert len(split.train) == 14 and not set(split.train) & set(split.test)


def test_digest_ignores_order_but_not_content():
    a = every_nth_split(names(20), 8)
    b = every_nth_split(names(20)[::-1], 8)
    c = every_nth_split(names(20), 10)
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_every_nth_rejects_bad_values():
    with pytest.raises(ValueError):
        every_nth_split(names(5), 1)
    with pytest.raises(ValueError):
        every_nth_split(names(1), 8)  # chỉ còn ảnh test


def test_list_split_and_validation(tmp_path):
    split = list_split(names(6), ["00006.jpg", "00002.jpg"])
    assert split.rule == "test_list"
    assert split.test == ["00002.jpg", "00006.jpg"]
    assert split.train == ["00001.jpg", "00003.jpg", "00004.jpg", "00005.jpg"]
    with pytest.raises(ValueError, match="not registered"):
        list_split(names(3), ["99999.jpg"])
    with pytest.raises(ValueError, match="empty"):
        list_split(names(3), [])
    with pytest.raises(ValueError, match="no training"):
        list_split(names(2), names(2))


def test_read_test_list(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("# đoạn quay kiểm tra\n00002.jpg\n\n  00004.jpg  \n", encoding="utf-8")
    assert read_test_list(path) == ["00002.jpg", "00004.jpg"]
    path.write_text("a.jpg\na.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_test_list(path)


def test_make_split_from_colmap_model_and_roundtrip(tmp_path):
    scene = tmp_path / "scene"
    make_model(scene / "sparse" / "0", names(10))
    split = make_split(scene, test_every=4)
    assert split.test == ["00001.jpg", "00005.jpg", "00009.jpg"]
    test_list = tmp_path / "test.txt"
    test_list.write_text("00010.jpg\n", encoding="utf-8")
    assert make_split(scene, test_every=4, test_list=test_list).test == ["00010.jpg"]

    out = tmp_path / "out" / "split.json"
    write_split(split, out)
    loaded = load_split(out)
    assert loaded.digest() == split.digest() and loaded.rule == "every_n" and loaded.test_every == 4


def test_points3d_roundtrip_and_train_view(tmp_path):
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    for name in names(6):
        (scene / "images" / name).write_bytes(b"jpg")
    make_model(scene / "sparse" / "0", names(6))
    write_points3d(scene / "sparse" / "0" / "points3D.bin", [
        {"id": 7, "xyz": (0.5, -1.0, 2.0), "rgb": (9, 8, 7), "error": 0.25, "track": [(1, 0), (2, 3), (3, 1)]},
        {"id": 8, "xyz": (1.0, 1.0, 1.0), "rgb": (1, 1, 1), "error": 1.5, "track": [(5, 0), (6, 0)]},
        {"id": 9, "xyz": (2.0, 2.0, 2.0), "rgb": (1, 1, 1), "error": 1.0, "track": [(1, 2), (1, 4), (6, 1)]},
    ])
    points = read_points3d_binary(scene / "sparse" / "0" / "points3D.bin")
    assert [p.point_id for p in points] == [7, 8, 9] and points[0].track.tolist() == [[1, 0], [2, 3], [3, 1]]
    copy = tmp_path / "copy.bin"
    write_points3d_binary(copy, points)
    assert copy.read_bytes() == (scene / "sparse" / "0" / "points3D.bin").read_bytes()

    split = list_split(names(6), ["00005.jpg", "00006.jpg"])
    info = make_train_view(scene, tmp_path / "view", split)
    kept = read_points3d_binary(tmp_path / "view" / "sparse" / "0" / "points3D.bin")
    # điểm 8 chỉ ảnh test thấy; điểm 9 có hai quan sát nhưng cùng một ảnh train -> đều bị bỏ
    assert [p.point_id for p in kept] == [7]
    assert info["points_total"] == 3 and info["points_kept"] == 1
    assert (tmp_path / "view" / "images" / "00006.jpg").is_file()
    assert (tmp_path / "view" / "sparse" / "0" / "images.bin").is_file()
    with pytest.raises(ValueError, match="enough training images"):
        make_train_view(scene, tmp_path / "view2", split, min_train_views=4)
