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
| `scripts/env_probe.sh` | Đã viết, đã chạy thử trong sandbox không có GPU | Cú pháp và shellcheck sạch; các mục GPU báo FAIL đúng như mong đợi; không in biến môi trường bí mật (đã thử với API key giả). **Chưa chạy trên Runpod.** |

## Chưa xác minh / còn mở

- Câu trả lời của GVHD cho Q1–Q3 và Q5–Q12 (`docs/plan.md`, mục 2) — đang dùng giả định tạm.
- Cấu hình pod thực tế: loại GPU, data center, template và image tag, giá/giờ.
- Chưa chọn LICENSE cho repository (repo đang public).
- Chưa đăng ký quyền truy cập ScanNet++.

## Môi trường

Chưa có số liệu thực tế. Sẽ ghi lại sau khi chạy `scripts/env_probe.sh` trên pod.

## Dữ liệu, checkpoint, kết quả

Chưa có. Dự kiến lưu trên network volume của Runpod tại `/workspace`; không đưa lên Git.

## Việc tiếp theo

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| 1 | Tạo pod Runpod, chạy `scripts/env_probe.sh` | SUMMARY báo PASS cho VRAM ≥ 24 GB và PyTorch nhận GPU; `/workspace` là network volume còn trống ≥ 50 GB; github.com, pypi.org, repo-sam.inria.fr trả mã 2xx/3xx; báo cáo lưu trong `/workspace/reports`; pod đã Stop |
| 3 | Cài môi trường 3DGS gốc có khóa phiên bản trên `/workspace`, build CUDA extension | Build thành công; có file khóa phiên bản; chạy được smoke test |
| 4 | Huấn luyện và đánh giá trên 1–2 cảnh nội thất công khai | Chỉ số nằm trong ngưỡng so với số liệu Inria công bố cho cùng phiên bản code |
| 5 | Cài COLMAP, chạy một cảnh tự quay từ video | Có camera + điểm thưa, tỉ lệ ảnh đăng ký được ghi lại, mô hình 3DGS xem được |
| 6 | Viewer web bản thô | Mở được mô hình đã xuất trên trình duyệt, đo thời gian tải |
