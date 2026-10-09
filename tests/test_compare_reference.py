"""Test logic tổng hợp và so sánh kết quả.

Mọi con số trong file này là dữ liệu giả để kiểm tra logic, KHÔNG phải kết quả thực nghiệm.
"""

import json

from indoor3d.eval.compare_reference import (
    compare,
    load_scene,
    main,
    peak_gpu_memory_mib,
    read_ply_vertex_count,
)

REFERENCE = {"PSNR": 29.690, "SSIM": 0.906, "LPIPS": 0.238}
TOLERANCE = {"PSNR": 0.5, "SSIM": 0.01, "LPIPS": 0.01}


def make_model_dir(root, scene, psnr, ssim, lpips, *, with_results=True, vertices=1234):
    model = root / scene
    (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
    if with_results:
        results = {
            "ours_7000": {"PSNR": psnr - 1.0, "SSIM": ssim - 0.02, "LPIPS": lpips + 0.05},
            "ours_30000": {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips},
        }
        (model / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (model / "run_meta.json").write_text(json.dumps({"train_seconds": 1500}), encoding="utf-8")
    (model / "gpu_log.csv").write_text(
        "2026/10/09 10:00:00.000, 1200, 10\n"
        "2026/10/09 10:00:02.000, 9000, 98\n"
        "garbage line\n"
        "2026/10/09 10:00:04.000, 8500, 97\n",
        encoding="utf-8",
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {vertices}\nproperty float x\nend_header\n"
    ).encode("ascii")
    (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_bytes(header + b"\x00\xff" * 64)
    return model


def write_config(path, scenes):
    config = {
        "name": "test",
        "scenes": scenes,
        "method": "ours_30000",
        "iteration": 30000,
        "reference": {"metrics": REFERENCE, "source": "test"},
        "tolerance": TOLERANCE,
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_read_ply_vertex_count_binary_body(tmp_path):
    model = make_model_dir(tmp_path, "a", 29.7, 0.906, 0.238, vertices=987654)
    ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    assert read_ply_vertex_count(ply) == 987654


def test_read_ply_vertex_count_missing_file(tmp_path):
    assert read_ply_vertex_count(tmp_path / "missing.ply") is None


def test_peak_gpu_memory_ignores_bad_lines(tmp_path):
    model = make_model_dir(tmp_path, "a", 29.7, 0.906, 0.238)
    assert peak_gpu_memory_mib(model / "gpu_log.csv") == 9000.0
    assert peak_gpu_memory_mib(tmp_path / "missing.csv") is None


def test_load_scene_reads_all_fields(tmp_path):
    make_model_dir(tmp_path, "a", 29.80, 0.910, 0.230)
    scene = load_scene("a", tmp_path / "a", "ours_30000", 30000)
    assert scene.status == "OK"
    assert scene.metrics == {"PSNR": 29.80, "SSIM": 0.910, "LPIPS": 0.230}
    assert scene.train_seconds == 1500
    assert scene.peak_vram_mib == 9000.0
    assert scene.num_gaussians == 1234


def test_compare_pass_when_mean_within_tolerance(tmp_path):
    make_model_dir(tmp_path, "a", 29.40, 0.900, 0.240)
    make_model_dir(tmp_path, "b", 30.10, 0.910, 0.236)
    scenes = [load_scene(s, tmp_path / s, "ours_30000", 30000) for s in ("a", "b")]
    summary = compare(scenes, REFERENCE, TOLERANCE)
    assert summary["verdict"] == "PASS"
    assert abs(summary["mean"]["PSNR"] - 29.75) < 1e-9
    assert abs(summary["delta"]["PSNR"] - 0.06) < 1e-9


def test_compare_fail_when_psnr_too_low(tmp_path):
    make_model_dir(tmp_path, "a", 28.90, 0.905, 0.239)
    make_model_dir(tmp_path, "b", 29.00, 0.905, 0.239)
    scenes = [load_scene(s, tmp_path / s, "ours_30000", 30000) for s in ("a", "b")]
    summary = compare(scenes, REFERENCE, TOLERANCE)
    assert summary["verdict"] == "FAIL"
    assert summary["within_tolerance"]["PSNR"] is False
    assert summary["within_tolerance"]["SSIM"] is True


def test_compare_incomplete_when_results_missing(tmp_path):
    make_model_dir(tmp_path, "a", 29.70, 0.906, 0.238)
    make_model_dir(tmp_path, "b", 29.70, 0.906, 0.238, with_results=False)
    scenes = [load_scene(s, tmp_path / s, "ours_30000", 30000) for s in ("a", "b")]
    assert scenes[1].status == "MISSING"
    assert compare(scenes, REFERENCE, TOLERANCE)["verdict"] == "INCOMPLETE"


def test_main_writes_summary_and_strict_exit_code(tmp_path, capsys):
    root = tmp_path / "models"
    root.mkdir()
    make_model_dir(root, "a", 27.00, 0.880, 0.300)  # clearly outside tolerance
    config = write_config(tmp_path / "config.json", ["a"])
    out = tmp_path / "summary.json"

    assert main(["--config", str(config), "--model-root", str(root), "--out", str(out)]) == 0
    assert main(["--config", str(config), "--model-root", str(root), "--strict"]) == 1

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"] == "FAIL"
    assert report["scenes"][0]["num_gaussians"] == 1234
    assert "Kết luận: **FAIL**" in capsys.readouterr().out
