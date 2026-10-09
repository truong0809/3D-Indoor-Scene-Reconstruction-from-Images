"""Test điều phối một lượt huấn luyện (indoor3d.train.run) với "runner" giả thay cho gsplat.

Runner giả ghi đúng các file mà simple_trainer.py và gsplat_launcher ghi (stats, ckpt, ply, split_applied).
Mọi con số là dữ liệu giả để kiểm tra logic.
"""

import json
import sys
from pathlib import Path

import pytest

from colmap_writer import make_model, write_cameras, write_points3d
from indoor3d.sfm.colmap_model import read_points3d_binary
from indoor3d.train.run import main, run_command

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_modes.json"


def names(n):
    return [f"{i:05d}.jpg" for i in range(1, n + 1)]


@pytest.fixture()
def env(tmp_path):
    scene = tmp_path / "scenes" / "phong01"
    (scene / "images").mkdir(parents=True)
    for name in names(17):
        (scene / "images" / name).write_bytes(b"jpg")
    make_model(scene / "sparse" / "0", names(17))
    gsplat = tmp_path / "gsplat"
    (gsplat / "examples").mkdir(parents=True)
    (gsplat / "examples" / "simple_trainer.py").write_text("# placeholder\n", encoding="utf-8")
    return {"scene": scene, "gsplat": gsplat, "out": tmp_path / "out"}


def value_after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class FakeRunner:
    """Giả lập launcher + simple_trainer: ghi kết quả vào --result-dir."""

    def __init__(self, num_gs=1_500_000, fail_on=None):
        self.calls = []
        self.num_gs = num_gs
        self.fail_on = fail_on

    def __call__(self, cmd, cwd, env, log_path):
        step = "eval" if "--ckpt" in cmd else "train"
        self.calls.append({"step": step, "cmd": cmd, "cwd": cwd, "env": env})
        Path(log_path).write_text("fake log\n", encoding="utf-8")
        if step == self.fail_on:
            return 3
        result = Path(value_after(cmd, "--result-dir"))
        final = int(value_after(cmd, "--max-steps")) - 1
        split = json.loads((result / "split.json").read_text(encoding="utf-8"))
        record = Path(value_after(cmd, "--record-split"))
        record.write_text(json.dumps({"splits": {"train": split["train"], "val": split["test"]}}), encoding="utf-8")
        Path(value_after(cmd, "--resolved-config")).write_text("{}", encoding="utf-8")
        (result / "stats").mkdir(exist_ok=True)
        if step == "train":
            for sub in ("ckpts", "ply"):
                (result / sub).mkdir(exist_ok=True)
            (result / "ckpts" / f"ckpt_{final}_rank0.pt").write_bytes(b"0" * 1024)
            header = f"ply\nformat binary_little_endian 1.0\nelement vertex {self.num_gs}\nend_header\n"
            (result / "ply" / f"point_cloud_{final}.ply").write_bytes(header.encode("ascii"))
            stats = {"mem": 6.5, "ellipse_time": 1000.0, "num_GS": self.num_gs}
            (result / "stats" / f"train_step{final:04d}_rank0.json").write_text(json.dumps(stats))
        else:
            stats = {"psnr": 28.0, "ssim": 0.88, "lpips": 0.2, "ellipse_time": 0.005, "num_GS": self.num_gs}
            (result / "stats" / f"val_step{final:04d}.json").write_text(json.dumps(stats))
        return 0


def run(env, *extra, runner=None, mode="default", out=None, allow_mismatch=True):
    # thư mục gsplat giả không phải git checkout, nên phải cho phép lệch commit (trừ test dành riêng)
    argv = ["--config", str(CONFIG), "--mode", mode, "--scene", str(env["scene"]), "--out", str(out or env["out"]),
            "--gsplat-dir", str(env["gsplat"]), *(["--allow-gsplat-mismatch"] if allow_mismatch else []), *extra]
    return main(argv, runner=runner or FakeRunner())


def test_default_then_mcmc_with_equal_budget(env, tmp_path):
    runner = FakeRunner(num_gs=1_234_567)
    assert run(env, runner=runner, out=tmp_path / "default") == 0
    summary = json.loads((tmp_path / "default" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "OK" and summary["num_gaussians"] == 1_234_567
    assert summary["metrics"] == {"PSNR": 28.0, "SSIM": 0.88, "LPIPS": 0.2}
    assert summary["split_applied_ok"] is True and summary["seed"] == 42
    meta = json.loads((tmp_path / "default" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "DONE" and meta["split"]["num_test"] == 3 and meta["cap_max"] is None
    train = runner.calls[0]
    assert train["cmd"][1:3] == ["-m", "indoor3d.train.gsplat_launcher"]
    assert value_after(train["cmd"], "--seed") == "42" and train["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert train["cwd"] == (env["gsplat"] / "examples").resolve()
    assert "--split-file" not in train["cmd"]  # quy tắc mỗi ảnh thứ 8: dùng nguyên logic của gsplat

    runner = FakeRunner(num_gs=1_234_567)
    assert run(env, "--cap-max-from", str(tmp_path / "default"), runner=runner, mode="mcmc",
               out=tmp_path / "mcmc") == 0
    assert value_after(runner.calls[0]["cmd"], "--strategy.cap-max") == "1234567"
    meta = json.loads((tmp_path / "mcmc" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["cap_max"]["policy"] == "match_baseline" and meta["cap_max"]["value"] == 1_234_567


def test_mcmc_without_budget_source_is_refused(env):
    assert run(env, mode="mcmc") == 2
    assert not env["out"].exists()


def test_dry_run_writes_nothing(env, capsys):
    assert run(env, "--dry-run", "--protocol", "max_steps=3000") == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["spec"]["deviations"]["max_steps"]["used"] == 3000
    assert "--max-steps 3000" in plan["train"] and "--ckpt" in plan["eval"]
    assert not env["out"].exists()


def test_test_list_uses_split_file_and_filtered_points(env, tmp_path):
    # điểm 1: hai ảnh train thấy; điểm 2: chỉ ảnh test thấy; điểm 3: một train + một test -> chỉ giữ điểm 1
    write_points3d(env["scene"] / "sparse" / "0" / "points3D.bin", [
        {"id": 1, "xyz": (0.0, 0.0, 1.0), "rgb": (1, 2, 3), "error": 0.5, "track": [(1, 0), (2, 0)]},
        {"id": 2, "xyz": (1.0, 0.0, 1.0), "rgb": (1, 2, 3), "error": 0.5, "track": [(16, 0), (17, 0)]},
        {"id": 3, "xyz": (2.0, 0.0, 1.0), "rgb": (1, 2, 3), "error": 0.5, "track": [(1, 1), (17, 1)]},
    ])
    test_list = tmp_path / "test.txt"
    test_list.write_text("00016.jpg\n00017.jpg\n", encoding="utf-8")
    runner = FakeRunner()
    assert run(env, "--test-list", str(test_list), runner=runner) == 0
    out = env["out"].resolve()
    assert value_after(runner.calls[0]["cmd"], "--split-file") == str(out / "split.json")
    assert value_after(runner.calls[0]["cmd"], "--data-dir") == str(out / "data")
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert split["rule"] == "test_list" and split["test"] == ["00016.jpg", "00017.jpg"]
    assert [p.point_id for p in read_points3d_binary(out / "data" / "sparse" / "0" / "points3D.bin")] == [1]
    assert (out / "data" / "images" / "00016.jpg").is_file()  # ảnh test vẫn có để render khi đánh giá
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["train_view"]["points_kept"] == 1 and meta["train_view"]["points_total"] == 3


def test_failures_and_reruns(env):
    assert run(env, runner=FakeRunner(fail_on="train")) == 1
    meta = json.loads((env["out"] / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "FAILED" and meta["failed_step"] == "train" and meta["exit_code"] == 3
    assert run(env) == 2  # thư mục dở dang: phải --rerun-incomplete hoặc --force
    assert run(env, "--rerun-incomplete") == 0
    backups = [p for p in env["out"].parent.iterdir() if ".prev_" in p.name]
    assert len(backups) == 1  # bản cũ được đổi tên, không bị xóa
    runner = FakeRunner()
    assert run(env, "--rerun-incomplete", runner=runner) == 0 and runner.calls == []  # cùng cấu hình, đã OK: bỏ qua
    # lượt OK nhưng cấu hình khác (vd. chạy thử ít vòng) thì không được bỏ qua lặng lẽ
    assert run(env, "--protocol", "max_steps=3000", "--rerun-incomplete", runner=runner) == 2
    assert run(env, "--seed", "43", runner=runner) == 2 and runner.calls == []
    assert run(env, "--seed", "43", "--force", runner=runner) == 0 and len(runner.calls) == 2


def test_gsplat_commit_must_match_configuration(env):
    assert run(env, allow_mismatch=False) == 2
    assert not env["out"].exists()


def test_mcmc_budget_from_other_seed_is_refused(env, tmp_path):
    assert run(env, runner=FakeRunner(), out=tmp_path / "default") == 0
    assert run(env, "--seed", "43", "--cap-max-from", str(tmp_path / "default"), mode="mcmc",
               out=tmp_path / "mcmc") == 2


def test_run_command_throttles_progress_lines(tmp_path):
    script = tmp_path / "progress.py"
    script.write_text("import sys\n"
                      "for i in range(200):\n"
                      "    sys.stdout.write(f'\\rstep {i}')\n"
                      "sys.stdout.write('\\nStep 600: done\\r\\nend')\n", encoding="utf-8")
    log = tmp_path / "log.txt"
    assert run_command([sys.executable, str(script)], tmp_path, {}, log, progress_interval=3600) == 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("$ ")
    # tiến độ đầu tiên được ghi, các cập nhật sau bị bỏ trong khoảng progress_interval;
    # dòng kết thúc bằng xuống dòng luôn được ghi (kể cả "\r\n")
    assert lines[1:] == ["step 0", "step 199", "Step 600: done", "end"]


def test_scene_checks(env, tmp_path):
    (env["scene"] / "images" / "00003.jpg").unlink()
    assert run(env) == 2
    (env["scene"] / "images" / "00003.jpg").write_bytes(b"jpg")
    write_cameras(env["scene"] / "sparse" / "0" / "cameras.bin", [(1, 1, 1920, 1080, [1500.0, 1500.0, 960.0, 540.0])])
    assert run(env) == 0
    meta = json.loads((env["out"] / "run_meta.json").read_text(encoding="utf-8"))
    assert any("wider than 1600" in warning for warning in meta["warnings"])
    assert main(["--config", str(CONFIG), "--mode", "default", "--scene", str(tmp_path / "none"),
                 "--out", str(tmp_path / "x"), "--gsplat-dir", str(env["gsplat"])]) == 2
