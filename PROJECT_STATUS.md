# PROJECT STATUS

_Cập nhật: 09/10/2026_

## Mục tiêu hiện tại

Giai đoạn 2 (môi trường) — Bước 1: tạo pod Runpod và ghi nhận môi trường thực tế (GPU, driver, CUDA, PyTorch, lưu trữ, kết nối mạng) bằng `scripts/env_probe.sh`.

## Quy ước trạng thái

`Đã viết` → `Đã chạy` → `Đã kiểm thử` → `Đã xác nhận đạt mục tiêu`.
Chỉ mức cuối cùng mới được tính là hoàn thành.

## Quyết định đã chốt

| # | Ngày | Quyết định | Ghi chú |
|---|---|---|---|
| D1 | 09/10/2026 | Phương pháp trọng tâm là 3DGS; baseline là 3DGS gốc (Kerbl et al., 2023) | Bản cài đặt (Inria chính chủ hay gsplat) chọn sau khi so sánh số liệu |
| D2 | 09/10/2026 | Hạ tầng tính toán: Runpod | Cấu hình cụ thể xem mục dưới |

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
| Khung repository | Đã viết | `README.md`, `PROJECT_STATUS.md`, `.gitignore`, `.gitattributes` |
| `scripts/env_probe.sh` (v2) | Đã viết, đã chạy thử trong sandbox không có GPU | Cú pháp và shellcheck sạch; các mục GPU báo FAIL đúng như mong đợi; không in biến môi trường bí mật (đã thử với API key giả). v2 kiểm tra thêm gitlab.inria.fr, download.pytorch.org, conda.anaconda.org. **Chưa chạy trên Runpod.** |
| Khảo sát yêu cầu cài đặt repo Inria | Đã kiểm tra trên mã nguồn | Commit ghim, submodule, `environment.yml`, mốc số liệu Deep Blending — ghi trong `docs/environment.md` |
| `scripts/setup_3dgs_inria.sh`, `scripts/smoke_test_3dgs.py`, `environment/gs-inria.conf` | Đã viết; đã chạy thử một phần trong sandbox không có GPU | Cú pháp và shellcheck/py_compile sạch. Đã chạy thử: cổng `provisional` và kiểm tra thiếu GPU dừng đúng; bước clone repo Inria và submodule (kể cả dự phòng mirror simple-knn) cho đúng commit ghim. **Các bước cài PyTorch, build extension và smoke test chưa chạy trên GPU.** Phiên bản PyTorch/CUDA đang ở trạng thái `provisional` |

## Chưa xác minh / còn mở

- Câu trả lời của GVHD cho Q1–Q3 và Q5–Q12 (`docs/plan.md`, mục 2) — đang dùng giả định tạm.
- Cấu hình pod thực tế: loại GPU, data center, template và image tag, giá/giờ.
- Chưa chọn LICENSE cho repository (repo hiện để private).
- Chưa đăng ký quyền truy cập ScanNet++.

## Môi trường

Phương án đề xuất và bố cục `/workspace`: `docs/environment.md`. Chưa có số liệu thực tế; sẽ ghi lại sau khi chạy `scripts/env_probe.sh` trên pod.

## Dữ liệu, checkpoint, kết quả

Chưa có. Dự kiến lưu trên network volume của Runpod tại `/workspace`; không đưa lên Git.

## Việc tiếp theo

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| 1 | Tạo pod Runpod, chạy `scripts/env_probe.sh` | SUMMARY báo PASS cho VRAM ≥ 24 GB, PyTorch nhận GPU và mọi host cần thiết truy cập được; `/workspace` là network volume còn trống ≥ 50 GB; báo cáo lưu trong `/workspace/reports`; pod đã Stop |
| 3 | Chốt phiên bản trong `environment/gs-inria.conf` từ báo cáo Bước 1, rồi chạy `scripts/setup_3dgs_inria.sh` | Script chạy hết 9 bước; smoke test báo PASSED; có file pip freeze và conda env trong `/workspace/reports` |
| 4 | Huấn luyện và đánh giá 3DGS gốc trên Deep Blending (playroom, drjohnson) | Trung bình 2 cảnh lệch so với số liệu Inria (PSNR 29.690, SSIM 0.906, LPIPS 0.238) không quá 0,5 dB / 0,01 / 0,01; đủ log, cấu hình và thời gian huấn luyện |
| 5 | Cài COLMAP, chạy một cảnh tự quay từ video | Có camera + điểm thưa, tỉ lệ ảnh đăng ký được ghi lại, mô hình 3DGS xem được |
| 6 | Viewer web bản thô | Mở được mô hình đã xuất trên trình duyệt, đo thời gian tải |
