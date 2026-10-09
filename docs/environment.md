# Môi trường thực nghiệm

_Cập nhật: 09/10/2026 · Trạng thái: **đề xuất** — phiên bản PyTorch/CUDA sẽ chốt sau báo cáo `scripts/env_probe.sh` (Bước 1)._

## 1. Bố cục trên pod Runpod

`/workspace` là network volume: còn nguyên khi Stop hoặc xóa pod. Container disk (mọi thứ ngoài `/workspace`, kể cả `/root`) bị xóa khi Stop pod, nên mọi thứ cần giữ phải nằm trong `/workspace`.

```
/workspace/
├── repo/                     # clone repo GitHub của đồ án (private, qua deploy key)
├── code/gaussian-splatting/  # repo 3DGS của Inria, ghim theo commit
├── miniforge3/               # conda (Miniforge) + env gs-inria (3DGS) + env gs-tools (COLMAP, ffmpeg)
├── data/
│   ├── downloads/            # file tải về (tandt_db.zip, ...)
│   ├── tandt_db/             # dữ liệu COLMAP do Inria cung cấp (Deep Blending, Tanks&Temples)
│   ├── raw/                  # video/ảnh gốc tự quay
│   └── scenes/<tên_cảnh>/    # input/ (frame đã chọn), images/ + sparse/0 (sau COLMAP), báo cáo
├── outputs/                  # mô hình đã huấn luyện, ảnh render, chỉ số
├── reports/                  # báo cáo môi trường, pip freeze, conda env
├── logs/                     # log cài đặt và huấn luyện
└── .ssh/                     # deploy key (chmod 600) — không bao giờ đưa lên Git
```

## 2. Truy cập repo private từ pod

Dùng **deploy key chỉ đọc**: khóa chỉ có quyền đọc đúng một repo này.

```bash
mkdir -p /workspace/.ssh && chmod 700 /workspace/.ssh
ssh-keygen -t ed25519 -N "" -C "runpod-indoor3d" -f /workspace/.ssh/github_deploy
cat /workspace/.ssh/github_deploy.pub
```

1. Copy dòng khóa công khai vừa in ra.
2. Trên GitHub: repo → **Settings → Deploy keys → Add deploy key**, dán khóa, **không** tick "Allow write access".
3. Clone repo trên pod:

```bash
SSH_CMD="ssh -i /workspace/.ssh/github_deploy -o IdentitiesOnly=yes -o UserKnownHostsFile=/workspace/.ssh/known_hosts -o StrictHostKeyChecking=accept-new"
GIT_SSH_COMMAND="$SSH_CMD" git clone git@github.com:truong0809/3D-Indoor-Scene-Reconstruction-from-Images.git /workspace/repo
git -C /workspace/repo config core.sshCommand "$SSH_CMD"
```

Sau đó cập nhật code bằng `git -C /workspace/repo pull`. Pod chỉ đọc được repo, không push được.

## 3. Môi trường cho 3DGS gốc (baseline B0)

### 3.1 Mã nguồn được ghim

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| graphdeco-inria/gaussian-splatting | commit `54c035f` (30/10/2024) | Commit mới nhất trên `main` khi kiểm tra ngày 09/10/2026 |
| diff-gaussian-rasterization | `9c5c202` (nhánh `dr_aa`) | Submodule, kèm thư viện glm |
| simple-knn | `86710c2` | Submodule, tải từ gitlab.inria.fr; dự phòng mirror GitHub `camenduru/simple-knn` (có đúng commit này) |
| fused-ssim | `1272e21` | Submodule |
| SIBR_viewers | không cài | Viewer desktop, không cần cho huấn luyện |

### 3.2 Vì sao không dùng nguyên `environment.yml` của Inria

File gốc ghim Python 3.7.13, PyTorch 1.12.1, `cudatoolkit` 11.6. Không dùng nguyên trạng vì:

- Python 3.7 đã hết hỗ trợ từ lâu.
- Gói `cudatoolkit` của conda chỉ có runtime, không có `nvcc` để build CUDA extension. README của Inria cũng cho biết họ build extension bằng CUDA SDK 11.8 cài riêng.
- CUDA 11.x không nhận GCC mới hơn 11 làm trình biên dịch host.

README của Inria ghi nhận thử nghiệm giới hạn cho thấy code chạy được với PyTorch 2.0 và CUDA 12.

### 3.3 Phương án đề xuất

- Miniforge và conda env `gs-inria` cài trong `/workspace`, độc lập với Python của template.
- Python 3.10; PyTorch và biến thể CUDA ghim trong `environment/gs-inria.conf` (hiện tạm đặt 2.7.1 + CUDA 12.8).
- Build extension bằng `nvcc` của template, cùng major version với CUDA của PyTorch.
- Build sẵn cho các kiến trúc 8.0, 8.6, 8.9, 9.0, để dùng lại env khi đổi giữa A100, A6000/A40, RTX 4090/L40S, H100.

**Quy tắc chọn biến thể CUDA của PyTorch:** không mới hơn phiên bản CUDA mà driver hỗ trợ (dòng "CUDA Version" của `nvidia-smi`). GPU Blackwell cần CUDA ≥ 12.8.

**Khác biệt so với môi trường gốc** (phải nêu trong báo cáo khi so sánh số liệu):

- Phiên bản Python, PyTorch và CUDA.
- `opencv-python-headless` thay cho `opencv-python`, vì server thường không có libGL.

## 4. Cài đặt và kiểm tra

```bash
cd /workspace/repo
bash scripts/setup_3dgs_inria.sh        # log: /workspace/logs, báo cáo: /workspace/reports
```

Script dừng ngay nếu:

- phiên bản trong file conf còn ở trạng thái `provisional`;
- driver không hỗ trợ CUDA đã chọn;
- không thấy `nvcc` cùng major version;
- thiếu dung lượng, hoặc không truy cập được GitHub, kho PyTorch hay conda;
- một submodule cần thiết bị thiếu hoặc không đúng commit được ghim.

Bước cuối chạy `scripts/smoke_test_3dgs.py` để kiểm tra cả ba extension trên GPU.

Kích hoạt env trong phiên terminal mới:

```bash
. /workspace/miniforge3/etc/profile.d/conda.sh && conda activate gs-inria
```

### 4.1 Công cụ xử lý dữ liệu (env `gs-tools`)

COLMAP và ffmpeg được cài vào conda env riêng `gs-tools`, tách khỏi `gs-inria` để tránh xung đột thư viện. Phiên bản ghim trong `environment/gs-tools.conf`: COLMAP 4.2.1 từ conda-forge (bản mới nhất khi kiểm tra ngày 09/10/2026, có bản build CUDA và bản CPU).

```bash
bash scripts/setup_data_tools.sh                          # bản CUDA
COLMAP_BUILD_OVERRIDE=cpu bash scripts/setup_data_tools.sh  # nếu driver không hợp với bản CUDA
```

Vì sao không dùng `convert.py` của Inria: script này truyền `--SiftExtraction.use_gpu` và `--SiftMatching.use_gpu`. Mã nguồn COLMAP hiện hành đã đổi thành `--FeatureExtraction.use_gpu` và `--FeatureMatching.use_gpu` (thay đổi nằm trong đợt tái cấu trúc trích đặc trưng của bản 3.13), nên `convert.py` không chạy được với COLMAP 4.x. `src/indoor3d/sfm/colmap_runner.py` làm đúng các bước của `convert.py` nhưng tự nhận tên tùy chọn theo phiên bản COLMAP.

### 4.2 Các lệnh chạy chính trên pod

```bash
bash scripts/run_baseline_deepblending.sh                       # Bước 4: tái lập trên Deep Blending
bash scripts/prepare_scene.sh <video.mp4> <tên_cảnh>            # Bước 5: frame -> COLMAP -> báo cáo
```

## 5. Mốc đối chiếu cho Bước 4

Bước 4 tái lập kết quả trên **Deep Blending**: 2 cảnh trong nhà (playroom, drjohnson), Inria cung cấp sẵn dữ liệu COLMAP trong `tandt_db.zip`.

Số liệu Inria công bố cho code hiện hành (`results.md`, rasterizer mặc định, cấu hình gốc), trung bình 2 cảnh:

| PSNR | SSIM | LPIPS |
|---|---|---|
| 29.690 | 0.906 | 0.238 |

- **Ngưỡng chấp nhận đề xuất:** lệch PSNR ≤ 0,5 dB, SSIM ≤ 0,01, LPIPS ≤ 0,01.
- **Cách chạy:** dùng đúng `train.py --eval`, `render.py` và `metrics.py` của Inria.
- **Thời gian tham khảo:** Inria báo cáo khoảng 45 phút huấn luyện với rasterizer gốc khi bật đủ tính năng; biểu đồ không ghi rõ GPU.

## 6. Sự cố đã biết

- `cuda_rasterizer/rasterizer_impl.h` dùng `uint32_t` nhưng không `#include <cstdint>`. Với GCC ≥ 13 có thể lỗi biên dịch. Cách xử lý: dùng GCC ≤ 12, hoặc thêm dòng include và ghi nhận đây là thay đổi so với mã gốc.
- `simple-knn` tải từ gitlab.inria.fr. Trang web của host này có lớp chống bot và từ chối một số truy cập tự động. Nếu git không clone được, script tự lấy cùng commit `86710c2` từ mirror GitHub; vì commit được xác định bằng mã SHA nên mã nguồn giống hệt bản gốc. Đường dự phòng này đã chạy thử thành công trong sandbox.
- `metrics.py` tải trọng số VGG và LPIPS từ Internet ở lần chạy đầu.
- Khi bật bù phơi sáng (`--train_test_exp`), Inria đưa nửa trái ảnh test vào huấn luyện và chỉ đánh giá nửa phải. Số liệu khi đó không so được với cấu hình chuẩn.
- Với `--eval`, code Inria luôn lấy mỗi ảnh thứ 8 (theo thứ tự tên) làm ảnh test: giá trị `llffhold=8` được gán cứng ở chỗ gọi hàm. Nhánh đọc danh sách ảnh test từ `sparse/0/test.txt` có trong code nhưng không bật được qua tham số dòng lệnh. Muốn dùng đoạn quay kiểm tra riêng (theo `docs/data/capture_protocol.md`), cần thêm một thay đổi nhỏ hoặc công cụ chia dữ liệu riêng — sẽ quyết định ở mốc M4 và ghi nhận là khác biệt so với mã gốc.
- COLMAP từ bản 3.12 ghi thêm `rigs.bin` và `frames.bin`. Định dạng `cameras.bin`, `images.bin`, `points3D.bin` giữ nguyên (đã đối chiếu mã nguồn COLMAP), nên bộ đọc của Inria vẫn dùng được.

## 7. Kết quả thực tế

Chưa có. Sẽ điền sau Bước 1 và Bước 3: GPU, driver, template/image tag, phiên bản đã cài, thời gian build, kết quả smoke test.
