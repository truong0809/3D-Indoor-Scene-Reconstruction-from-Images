# Tái tạo không gian nội thất 3D bằng ảnh chụp

**3D Indoor Scene Reconstruction from Images** — Đồ án tốt nghiệp

Hệ thống nhận ảnh hoặc video quay không gian nội thất bằng camera thông thường (không yêu cầu LiDAR), tái tạo cảnh 3D bằng **3D Gaussian Splatting (3DGS)**, cho phép xem và tương tác trên trình xem 3D, và đánh giá chất lượng bằng thực nghiệm.

> **Trạng thái:** đang thiết lập môi trường. Chưa có pipeline, mô hình hay kết quả thực nghiệm.
> Chi tiết xem [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Pipeline dự kiến

```
Ảnh / video → chọn frame → ước lượng camera (SfM) → huấn luyện 3DGS → đánh giá → xuất & nén → viewer web
```

Các thành phần cụ thể sẽ được chốt sau khảo sát và thử nghiệm — xem [docs/plan.md](docs/plan.md).

## Cấu trúc repository

| Đường dẫn | Nội dung |
|---|---|
| `PROJECT_STATUS.md` | Trạng thái hiện tại, quyết định đã chốt, việc tiếp theo |
| `docs/plan.md` | Kế hoạch tổng thể: phạm vi, mốc nghiệm thu, rủi ro |
| `docs/research/reading_list.md` | Danh sách tài liệu cần khảo sát |
| `docs/environment.md` | Môi trường thực nghiệm: bố cục trên pod, phiên bản ghim, mốc đối chiếu |
| `environment/gs-inria.conf` | Phiên bản ghim cho môi trường 3DGS gốc |
| `scripts/env_probe.sh` | Kiểm tra môi trường GPU của pod (chỉ đọc) |
| `scripts/setup_3dgs_inria.sh` | Cài môi trường 3DGS gốc của Inria trong `/workspace` |
| `scripts/smoke_test_3dgs.py` | Kiểm tra nhanh các CUDA extension sau khi build |

## Hạ tầng tính toán

Huấn luyện chạy trên GPU thuê theo giờ tại Runpod. Dữ liệu, checkpoint và kết quả lưu trên network volume của pod (`/workspace`), không đưa lên Git.

### Kiểm tra môi trường pod

```bash
bash scripts/env_probe.sh                  # báo cáo lưu vào /workspace/reports/
OUT_DIR=/tmp bash scripts/env_probe.sh     # đổi nơi lưu báo cáo
```

Script chỉ đọc thông tin hệ thống, không cài đặt gì và không in biến môi trường bí mật.

### Cài môi trường 3DGS gốc

```bash
bash scripts/setup_3dgs_inria.sh
```

Chỉ chạy sau khi phiên bản trong `environment/gs-inria.conf` đã được chốt từ báo cáo môi trường. Chi tiết xem [docs/environment.md](docs/environment.md).

## Quy ước

- Không đưa dataset, checkpoint, file `.ply`, video hay log thô lên Git.
- Không đưa token, mật khẩu, API key hoặc file `.env` lên Git.
- Mọi số liệu trong báo cáo phải truy được về file kết quả, cấu hình và commit tương ứng.
