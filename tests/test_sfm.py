"""Test đọc model COLMAP, báo cáo chất lượng và điều phối COLMAP (dùng chương trình colmap giả).

Mọi model và số liệu trong file này là dữ liệu tổng hợp để kiểm tra logic.
"""

import json
import os
import stat
import sys
import textwrap

import pytest

from colmap_writer import make_model
from indoor3d.sfm.colmap_model import read_cameras_binary, read_images_binary, read_points3d_summary
from indoor3d.sfm.colmap_runner import run_colmap
from indoor3d.sfm.report import analyze_scene


def names(n):
    return [f"{i:05d}.jpg" for i in range(1, n + 1)]


def make_scene(tmp_path, n_input, models, error=0.8, final_from=0):
    """models: list số ảnh đăng ký của từng model trong distorted/sparse/<k>."""
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    for name in names(n_input):
        (scene / "input" / name).write_bytes(b"jpg")
    for index, registered in enumerate(models):
        make_model(scene / "distorted" / "sparse" / str(index), names(registered), camera_model_id=4, error=error)
    if models:
        make_model(scene / "sparse" / "0", names(models[final_from]), camera_model_id=1, error=error)
    return scene


def test_readers_roundtrip(tmp_path):
    model = make_model(tmp_path / "m", names(4), camera_model_id=4, error=0.5, points_per_image=3)
    cameras = read_cameras_binary(model / "cameras.bin")
    assert cameras[1].model == "OPENCV" and len(cameras[1].params) == 8
    assert (cameras[1].width, cameras[1].height) == (640, 480)
    images = read_images_binary(model / "images.bin")
    assert [images[i].name for i in sorted(images)] == names(4)
    assert images[2].num_points2d == 4 and images[2].num_observations == 3
    assert images[3].tvec == (0.0, 0.0, 3.0)
    points = read_points3d_summary(model / "points3D.bin")
    assert points.count == 12
    assert points.mean_error == pytest.approx(0.5)
    assert points.mean_track_length == pytest.approx(2.0)


def test_report_pass(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 20, [20]))
    assert report["verdict"] == "PASS"
    assert report["registration_ratio"] == 1.0
    assert report["model"]["cameras"][0]["model"] == "PINHOLE"
    assert report["unregistered_images"] == []


def test_report_warn_when_split_into_models(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 20, [20, 4]))
    assert report["verdict"] == "WARN"
    assert any("split into 2 models" in w for w in report["warnings"])


def test_report_warn_and_fail_on_low_registration(tmp_path):
    assert analyze_scene(make_scene(tmp_path / "a", 20, [17]))["verdict"] == "WARN"
    report = analyze_scene(make_scene(tmp_path / "b", 20, [10]))
    assert report["verdict"] == "FAIL"
    assert report["unregistered_images"] == names(20)[10:]


def test_report_warn_on_high_reprojection_error(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 10, [10], error=2.0))
    assert report["verdict"] == "WARN"
    assert any("reprojection error" in w for w in report["warnings"])


def test_report_fail_without_model(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 5, []))
    assert report["verdict"] == "FAIL"


FAKE_COLMAP = textwrap.dedent("""\
    #!{python}
    import json, os, shutil, sys
    from pathlib import Path
    args = sys.argv[1:]
    with open(os.environ["FAKE_COLMAP_LOG"], "a") as log:
        log.write(json.dumps(args) + "\\n")
    new = os.environ.get("FAKE_COLMAP_STYLE", "new") == "new"
    cmd = args[0]
    if "-h" in args:
        if cmd == "feature_extractor":
            print("--FeatureExtraction.use_gpu" if new else "--SiftExtraction.use_gpu")
        else:
            print("--FeatureMatching.use_gpu" if new else "--SiftMatching.use_gpu")
        sys.exit(0)
    def opt(name):
        return args[args.index("--" + name) + 1]
    if cmd == "feature_extractor":
        Path(opt("database_path")).write_bytes(b"db")
    elif cmd in ("exhaustive_matcher", "sequential_matcher"):
        assert Path(opt("database_path")).exists()
    elif cmd in ("mapper", "global_mapper"):
        for model in Path(os.environ["FAKE_COLMAP_TEMPLATE"]).iterdir():
            shutil.copytree(model, Path(opt("output_path")) / model.name)
    elif cmd == "image_undistorter":
        out, src = Path(opt("output_path")), Path(opt("input_path"))
        shutil.copytree(opt("image_path"), out / "images")
        (out / "sparse").mkdir(exist_ok=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            shutil.copy(src / name, out / "sparse" / name)
    else:
        sys.exit(2)
    """)


@pytest.fixture
def fake_colmap(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "colmap"
    exe.parent.mkdir()
    exe.write_text(FAKE_COLMAP.format(python=sys.executable), encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    template = tmp_path / "template"
    make_model(template / "0", names(2), camera_model_id=4)
    make_model(template / "1", names(5), camera_model_id=4)  # largest model, not index 0
    log = tmp_path / "colmap_calls.jsonl"
    monkeypatch.setenv("FAKE_COLMAP_TEMPLATE", str(template))
    monkeypatch.setenv("FAKE_COLMAP_LOG", str(log))
    return exe, log


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize("style,extract_opt,match_opt", [
    ("new", "FeatureExtraction.use_gpu", "FeatureMatching.use_gpu"),
    ("old", "SiftExtraction.use_gpu", "SiftMatching.use_gpu"),
])
def test_runner_orchestrates_colmap(tmp_path, fake_colmap, monkeypatch, style, extract_opt, match_opt):
    exe, log = fake_colmap
    monkeypatch.setenv("FAKE_COLMAP_STYLE", style)
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    for name in names(6):
        (scene / "input" / name).write_bytes(b"jpg")

    report = run_colmap(scene, colmap=str(exe))

    settings = report["colmap_settings"]
    assert settings["extract_gpu_option"] == extract_opt
    assert settings["match_gpu_option"] == match_opt
    assert settings["undistorted_model"].endswith(os.sep + "1")
    run_calls = [c for c in calls(log) if "-h" not in c]
    assert [c[0] for c in run_calls] == ["feature_extractor", "exhaustive_matcher", "mapper", "image_undistorter"]
    assert run_calls[0][run_calls[0].index(f"--{extract_opt}") + 1] == "1"
    assert "--Mapper.ba_global_function_tolerance=0.000001" in run_calls[2]
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        assert (scene / "sparse" / "0" / name).is_file()
    assert report["registration_ratio"] == pytest.approx(5 / 6)
    assert report["unregistered_images"] == ["00006.jpg"]
    assert report["verdict"] == "WARN"  # 83 % registered and split into 2 models
    assert json.loads((scene / "sfm_report.json").read_text(encoding="utf-8"))["verdict"] == "WARN"


def test_runner_options_and_overwrite_guard(tmp_path, fake_colmap):
    exe, log = fake_colmap
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    for name in names(5):
        (scene / "input" / name).write_bytes(b"jpg")

    report = run_colmap(scene, colmap=str(exe), use_gpu=False, matcher="sequential",
                        sequential_overlap=15, mapper="global")
    assert report["verdict"] == "WARN"  # all 5 registered, but split into 2 models
    run_calls = [c for c in calls(log) if "-h" not in c]
    assert run_calls[0][run_calls[0].index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert run_calls[1][0] == "sequential_matcher" and "15" in run_calls[1]
    assert run_calls[2][0] == "global_mapper"

    with pytest.raises(RuntimeError, match="--overwrite"):
        run_colmap(scene, colmap=str(exe))
    assert run_colmap(scene, colmap=str(exe), overwrite=True)["verdict"] == "WARN"
