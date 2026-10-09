# Tái tạo không gian nội thất 3D bằng ảnh chụp

**3D Indoor Scene Reconstruction from Images** — Đồ án tốt nghiệp

Hệ thống nhận ảnh hoặc video quay không gian nội thất bằng camera thông thường (không yêu cầu LiDAR), tái tạo cảnh 3D bằng **3D Gaussian Splatting (3DGS)**, cho phép xem và tương tác trên trình xem 3D, và đánh giá chất lượng bằng thực nghiệm.

> **Trạng thái:** đã có script cài đặt, tái lập baseline và chuẩn bị dữ liệu, kiểm thử trên CPU. Chưa chạy trên GPU, chưa có mô hình hay kết quả thực nghiệm.
> Chi tiết xem [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Pipeline

```
Video / ảnh → chọn frame nét → COLMAP (camera + điểm thưa) → huấn luyện 3DGS → đánh giá → xuất & nén → viewer web
```

Phương pháp và các bước kiểm chứng: [docs/plan.md](docs/plan.md).

## Cấu trúc repository

| Đường dẫn | Nội dung |
|---|---|
| `PROJECT_STATUS.md` | Trạng thái hiện tại, quyết định đã chốt, việc tiếp theo |
| `docs/plan.md` | Kế hoạch tổng thể: phạm vi, mốc nghiệm thu, rủi ro |
| `docs/environment.md` | Môi trường trên pod, phiên bản ghim, mốc đối chiếu, sự cố đã biết |
| `docs/data/capture_protocol.md` | Quy trình quay video không gian nội thất |
| `docs/research/reading_list.md` | Danh sách tài liệu cần khảo sát |
| `environment/` | Phiên bản ghim: `gs-inria.conf` (3DGS gốc), `gs-tools.conf` (COLMAP, ffmpeg) |
| `configs/` | Cấu hình thực nghiệm, gồm số liệu tham chiếu và ngưỡng chấp nhận |
| `src/indoor3d/` | Mã nguồn pipeline: `data/` (chọn frame), `sfm/` (COLMAP), `eval/` (so sánh kết quả) |
| `scripts/` | Lệnh chạy trên pod (cài đặt, tái lập baseline, chuẩn bị cảnh) |
| `tests/` | Kiểm thử chạy trên CPU, dùng dữ liệu tổng hợp |

## Chạy trên pod Runpod

Thứ tự lần đầu (chi tiết trong [docs/environment.md](docs/environment.md)):

```bash
bash scripts/env_probe.sh                         # 1. ghi nhận môi trường (chỉ đọc)
bash scripts/setup_3dgs_inria.sh                  # 2. cài 3DGS gốc (sau khi chốt phiên bản)
bash scripts/setup_data_tools.sh                  # 3. cài COLMAP + ffmpeg
bash scripts/run_baseline_deepblending.sh         # 4. tái lập baseline trên Deep Blending
bash scripts/prepare_scene.sh <video.mp4> <cảnh>  # 5. chuẩn bị một cảnh tự quay
```

Script cài đặt chỉ chạy khi phiên bản trong `environment/gs-inria.conf` đã được chốt từ báo cáo môi trường.

## Kiểm thử trên máy không có GPU

```bash
pip install numpy opencv-python-headless pytest
python -m pytest            # cần ffmpeg cho một test tích hợp; test đó tự bỏ qua nếu thiếu
```

## Quy ước

- Không đưa dataset, checkpoint, file `.ply`, video hay log thô lên Git.
- Không đưa token, mật khẩu, API key hoặc file `.env` lên Git.
- Mọi số liệu trong báo cáo phải truy được về file kết quả, cấu hình và commit tương ứng.
- Dữ liệu trong `tests/` là dữ liệu tổng hợp để kiểm tra logic, không phải kết quả thực nghiệm.
