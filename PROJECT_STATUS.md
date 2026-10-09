# PROJECT STATUS

_Cập nhật: 09/10/2026_

## Mục tiêu hiện tại

Hoàn thiện các bước không cần GPU (cài đặt, tái lập baseline, chuẩn bị dữ liệu, viewer), rồi chạy kiểm chứng toàn bộ trên Runpod trong một lượt ở cuối (quyết định D3).

## Quy ước trạng thái

`Đã viết` → `Đã chạy` → `Đã kiểm thử` → `Đã xác nhận đạt mục tiêu`.
Chỉ mức cuối cùng mới được tính là hoàn thành.

## Quyết định đã chốt

| # | Ngày | Quyết định | Ghi chú |
|---|---|---|---|
| D1 | 09/10/2026 | Phương pháp trọng tâm là 3DGS; baseline là 3DGS gốc (Kerbl et al., 2023) | Bản cài đặt (Inria chính chủ hay gsplat) chọn sau khi so sánh số liệu |
| D2 | 09/10/2026 | Hạ tầng tính toán: Runpod | Cấu hình cụ thể xem mục dưới |
| D3 | 09/10/2026 | Chạy kiểm chứng trên Runpod gộp vào cuối, sau khi các bước không cần GPU đã xong | Mọi script phải có kiểm thử CPU trước đó |
| D4 | 09/10/2026 | Dùng `src/indoor3d/sfm/colmap_runner.py` thay cho `convert.py` của Inria | `convert.py` dùng tên tùy chọn GPU cũ, không chạy với COLMAP 4.x (xem `docs/environment.md`, mục 4.1) |

## Đề xuất chờ xác nhận

| Hạng mục | Đề xuất | Lý do |
|---|---|---|
| Loại cloud | Secure Cloud | Network volume chỉ dùng được với pod trên Secure Cloud |
| Lưu trữ | Network volume 100 GB, gói Standard, cùng data center với GPU | Dữ liệu tồn tại độc lập với pod |
| GPU | RTX A6000 48 GB (thay thế: A40 48 GB, RTX 4090 24 GB) | Xem tiêu chí bên dưới |

**Tiêu chí chọn GPU**

- Bắt buộc:
  - VRAM ≥ 24 GB (mức Inria khuyến nghị để huấn luyện đạt chất lượng như paper).
  - Kiến trúc Ampere hoặc Ada, tương thích CUDA 11–12. Không dùng Blackwell vì cần CUDA ≥ 12.8.
  - Có trên Secure Cloud, cùng data center với network volume.
- Ưu tiên:
  - Giá/giờ thấp nhất trong số GPU đạt các tiêu chí bắt buộc.
  - Cùng dòng với máy tham chiếu của Inria (A6000) để so sánh được thời gian huấn luyện.

## Đã thực hiện

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Kế hoạch Giai đoạn 0 | Đã viết, đã được duyệt | `docs/plan.md` |
| Danh sách tài liệu khảo sát ban đầu | Đã viết | `docs/research/reading_list.md` (venue/link chưa kiểm chứng hết) |
| Khung repository | Đã viết | `README.md`, `PROJECT_STATUS.md`, `.gitignore`, `.gitattributes`, `pyproject.toml` |
| `scripts/env_probe.sh` (v2) | Đã viết, đã chạy thử trong sandbox không có GPU | Shellcheck sạch; mục GPU báo FAIL đúng như mong đợi; không in biến môi trường bí mật. **Chưa chạy trên Runpod.** |
| Khảo sát yêu cầu cài đặt repo Inria | Đã kiểm tra trên mã nguồn | Commit ghim, submodule, `environment.yml`, cách chia train/test, mốc số liệu Deep Blending — `docs/environment.md` |
| Cài môi trường 3DGS gốc: `scripts/setup_3dgs_inria.sh`, `scripts/smoke_test_3dgs.py`, `environment/gs-inria.conf` | Đã viết; chạy thử một phần trong sandbox | Cổng `provisional` và kiểm tra thiếu GPU dừng đúng; bước clone repo Inria + submodule (kể cả mirror simple-knn) cho đúng commit ghim. **Cài PyTorch, build extension, smoke test chưa chạy trên GPU.** Phiên bản PyTorch/CUDA còn `provisional` |
| Quy trình quay video | Đã viết | `docs/data/capture_protocol.md` — sẽ hiệu chỉnh sau lần quay thử |
| Bước 4 — tái lập Deep Blending: `scripts/run_baseline_deepblending.sh`, `configs/baseline_deepblending.json`, `src/indoor3d/eval/compare_reference.py` | Đã viết; đã kiểm thử phần không cần GPU | Tham số giống `full_eval.py` của Inria; 8 test cho phần tổng hợp/so sánh (dữ liệu giả); đọc config, tìm thư mục dataset và cổng kiểm tra môi trường đã chạy thử. **Chưa huấn luyện trên GPU.** |
| Bước 5 — chuẩn bị cảnh: `src/indoor3d/data/frames.py`, `src/indoor3d/sfm/{colmap_model,colmap_runner,report}.py`, `scripts/setup_data_tools.sh`, `scripts/prepare_scene.sh`, `environment/gs-tools.conf` | Đã viết; đã kiểm thử trên CPU | 17 test: chọn frame nét với video tổng hợp qua ffmpeg thật; đọc/ghi model COLMAP nhị phân (đối chiếu khớp với bộ đọc của Inria); điều phối COLMAP bằng chương trình colmap giả cho cả tên tùy chọn mới và cũ; chạy trọn `prepare_scene.sh` với colmap giả. **Chưa chạy với COLMAP thật và video thật.** |

## Chưa xác minh / còn mở

- Câu trả lời của GVHD cho Q1–Q3 và Q5–Q12 (`docs/plan.md`, mục 2) — đang dùng giả định tạm.
- Cấu hình pod thực tế: loại GPU, data center, template và image tag, giá/giờ.
- Bản COLMAP CUDA từ conda-forge có cài được với driver của pod không (dự phòng: bản CPU).
- Cách chia train/test cho dữ liệu tự quay: code Inria gán cứng mỗi ảnh thứ 8 làm test; cần quyết định ở M4 (thay đổi nhỏ trong code Inria hoặc công cụ chia riêng).
- Chưa chọn LICENSE cho repository (repo hiện để private).
- Chưa đăng ký quyền truy cập ScanNet++.

## Môi trường

Phương án và bố cục `/workspace`: `docs/environment.md`. Chưa có số liệu thực tế; sẽ ghi lại sau khi chạy `scripts/env_probe.sh` trên pod.

## Dữ liệu, checkpoint, kết quả

Chưa có. Dự kiến lưu trên network volume của Runpod tại `/workspace`; không đưa lên Git.

## Việc tiếp theo

**Trước lượt chạy Runpod (không cần GPU)**

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| 6 | Xuất mô hình cho web và viewer web bản thô | Mở được mô hình mẫu trên trình duyệt; có số đo thời gian tải; có kiểm thử chuyển đổi định dạng |
| — | Quay thử 1 cảnh theo `docs/data/capture_protocol.md` | Có video chính + đoạn quay kiểm tra, đặt tên và ghi metadata đúng quy ước |

**Lượt chạy Runpod (theo thứ tự)**

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| 1 | `scripts/env_probe.sh` | SUMMARY báo PASS cho VRAM ≥ 24 GB, PyTorch nhận GPU và mọi host cần thiết truy cập được; `/workspace` là network volume còn trống ≥ 50 GB |
| 3 | Chốt `environment/gs-inria.conf` từ báo cáo Bước 1; `scripts/setup_3dgs_inria.sh`; `scripts/setup_data_tools.sh` | Hết 9 bước, smoke test PASSED; COLMAP và ffmpeg chạy được; có file pip freeze / conda env |
| 4 | `scripts/run_baseline_deepblending.sh` | Trung bình 2 cảnh lệch so với Inria (PSNR 29.690, SSIM 0.906, LPIPS 0.238) không quá 0,5 dB / 0,01 / 0,01 |
| 5 | `scripts/prepare_scene.sh` với cảnh quay thử, rồi huấn luyện để xem | `sfm_report.json` không FAIL; mô hình xem được trên viewer |
