"""Test trích và chọn frame. Video và ảnh trong test là dữ liệu tổng hợp."""

import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from indoor3d.data.frames import main, prepare_from_images, prepare_from_video, select_frames, sharpness

FFMPEG = shutil.which("ffmpeg")


def texture(height=360, width=1200, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), 1.0)
    tiles = ((np.indices((height, width)).sum(axis=0) // 24) % 2 * 120).astype(np.uint8)
    return cv2.cvtColor(cv2.addWeighted(base, 0.6, tiles, 0.4, 0), cv2.COLOR_GRAY2BGR)


def heavy_blur(image):
    return cv2.GaussianBlur(image, (0, 0), 6.0)


def test_select_frames_picks_sharpest_per_window():
    scores = [1, 9, 2, 3, 3, 8, 7, 1]  # two windows of four
    selected, stats = select_frames(scores, target=2, blur_ratio=0.0)
    assert selected == [1, 5]
    assert stats["windows"] == 2 and stats["selected"] == 2


def test_select_frames_rejects_blurry_windows():
    scores = [100, 90, 95, 2, 3, 1, 98, 97, 99]  # middle window is blurry
    selected, stats = select_frames(scores, target=3, blur_ratio=0.4)
    assert selected == [0, 8]
    assert stats["rejected_blurry"] == 1


def test_select_frames_handles_more_windows_than_frames():
    selected, stats = select_frames([5.0, 6.0], target=10, blur_ratio=0.4)
    assert selected == [0, 1]
    assert stats["windows"] == 2


def test_select_frames_empty():
    selected, stats = select_frames([], target=10, blur_ratio=0.4)
    assert selected == [] and stats["candidates"] == 0


def test_sharpness_orders_sharp_above_blurred(tmp_path):
    image = texture()
    cv2.imwrite(str(tmp_path / "sharp.png"), image)
    cv2.imwrite(str(tmp_path / "blur.png"), heavy_blur(image))
    assert sharpness(tmp_path / "sharp.png") > 5 * sharpness(tmp_path / "blur.png")


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_prepare_from_video_skips_blurred_frames(tmp_path):
    frames_dir = tmp_path / "src"
    frames_dir.mkdir()
    base = texture()
    blurred = set(range(1, 48, 4)) | set(range(32, 40))  # periodic blur + one fully blurred stretch
    for i in range(48):
        frame = np.ascontiguousarray(base[:, i * 8: i * 8 + 640])
        if i in blurred:
            frame = heavy_blur(frame)
        cv2.imwrite(str(frames_dir / f"f_{i:03d}.png"), frame)
    video = tmp_path / "pan.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-framerate", "12",
                    "-i", str(frames_dir / "f_%03d.png"), "-c:v", "mpeg4", "-q:v", "2", str(video)],
                   check=True)

    scene = tmp_path / "scene"
    report = prepare_from_video(video, scene, fps=12, target=12, blur_ratio=0.4)

    assert report["candidates"] == 48
    assert report["rejected_blurry"] == 2  # windows 32-35 and 36-39
    assert report["selected"] == 10
    chosen = {record["candidate_index"] for record in report["frames"]}
    assert not chosen & blurred
    assert sorted(p.name for p in (scene / "input").iterdir()) == [f"{k:05d}.jpg" for k in range(1, 11)]
    assert not (scene / "frames_candidates").exists()


def test_prepare_from_images_converts_and_filters(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    base = texture()
    for i in range(6):
        image = base[:, i * 50: i * 50 + 640]
        cv2.imwrite(str(src / f"IMG_{i}.png"), heavy_blur(image) if i == 2 else image)
    report = prepare_from_images(src, tmp_path / "scene", target=6, blur_ratio=0.4)
    assert report["selected"] == 5
    assert "IMG_2.png" not in {record["source_name"] for record in report["frames"]}
    assert cv2.imread(str(tmp_path / "scene" / "input" / "00001.jpg")) is not None


def test_main_refuses_to_overwrite_existing_input(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    cv2.imwrite(str(src / "a.png"), texture()[:, :640])
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    (scene / "input" / "old.jpg").write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["--images", str(src), "--scene-dir", str(scene)])
    assert main(["--images", str(src), "--scene-dir", str(scene), "--overwrite"]) == 0
    report = json.loads((scene / "frames_report.json").read_text(encoding="utf-8"))
    assert report["selected"] == 1
    assert not (scene / "input" / "old.jpg").exists()
