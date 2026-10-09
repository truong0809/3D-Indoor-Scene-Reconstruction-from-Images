# Thiết kế: hai chế độ huấn luyện 3DGS — Default (baseline) và MCMC

_09/10/2026 · Trạng thái: **đã viết, đã kiểm thử trên CPU** (kể cả với mã gsplat thật ở mức phân giải tham số và chia dữ liệu). **Chưa huấn luyện trên GPU, chưa có số liệu.**_

## 1. Mục tiêu và ranh giới

- **Baseline:** 3DGS với `DefaultStrategy` của gsplat. Đây là cài đặt lại heuristic densification của 3DGS gốc (Kerbl et al., 2023), tương ứng B1 trong `docs/plan.md` (mục 3.3).
- **Chế độ thứ hai:** 3DGS-MCMC (Kheradmand et al., NeurIPS 2024) qua `MCMCStrategy` của gsplat, lấy từ notebook demo của nhóm. Kết quả kiểm tra demo: `docs/research/demo_review.md`.
- **Ranh giới trung thực:** MCMC là phương pháp đã công bố và đã có sẵn trong gsplat.
  - Đưa nó vào pipeline và đánh giá trên dữ liệu nội thất quay bằng điện thoại thuộc loại đóng góp (b) ở `docs/plan.md` mục 3.4.
  - Đây không phải thuật toán mới. Đổi `cap_max` hay hệ số regularization cũng không phải đóng góp thuật toán.
- **Quan hệ với B0:** bản Inria (B0, `scripts/run_baseline_deepblending.sh`) vẫn là mốc đối chiếu.
  - Default trên gsplat chỉ được dùng làm baseline sau khi chứng minh số liệu tương đương B0 trên Deep Blending, theo điều kiện đã ghi ở mục 3.3 của kế hoạch.
  - Bước 4 của `scripts/run_modes_deepblending.sh` làm việc đối chiếu này, theo hai cách:
    - đối chiếu từng cảnh với lượt B0 tự chạy (`--b0-root`, cùng tập test), là điều kiện chính;
    - đối chiếu trung bình với số liệu Inria công bố.

## 2. Vị trí trong pipeline

```
video ─ frames.py ─ colmap_runner.py (COLMAP 4.2.1, undistort) ─▶ <cảnh>/images + sparse/0   (dùng chung)
                                                                   │
              ┌────────────────────────────────────────────────────┤
              ▼                                                    ▼
  B0: Inria train.py/render.py/metrics.py            gsplat simple_trainer.py @ 6e8c837
      (đối chiếu số liệu công bố)                    qua indoor3d.train.run + gsplat_launcher
                                                       ├─ default    baseline
                                                       ├─ mcmc       cùng protocol, cùng số Gaussian cuối với default
                                                       └─ mcmc_demo  đúng tham số notebook demo (khác protocol)
                                                                   │  summary.json mỗi lượt chạy
                                                                   ▼
                                                     indoor3d.eval.compare_modes → bảng + kiểm tra điều kiện
```

Mọi chế độ dùng cùng một thư mục cảnh: cùng frame, cùng pose camera và cùng điểm SfM để khởi tạo. Khác biệt giữa các chế độ chỉ nằm ở chiến lược quản lý Gaussian và tham số đi kèm.

## 3. Các chế độ (`configs/train_modes.json`)

Giá trị dưới đây đọc từ cấu hình mà `simple_trainer.py` thật (gsplat `6e8c837`) phân giải ra, xem mục 8.

| Chế độ | Preset gsplat | Tham số chính | Số Gaussian | Dùng cho |
|---|---|---|---|---|
| `default` | `default` | `DefaultStrategy`: densify mỗi 100 vòng, từ vòng 600 đến 14.900; ngưỡng gradient 0,0002; reset opacity ở các vòng 3.000, 6.000, 9.000, 12.000; opacity khởi tạo 0,1 | Do heuristic quyết định | Baseline |
| `mcmc` | `mcmc` | `MCMCStrategy`: relocate và thêm Gaussian mỗi 100 vòng, từ vòng 600 đến 24.900 (mỗi lần thêm 5% cho tới `cap_max`); `noise_lr` 5e5; ngưỡng "chết" opacity 0,005. Opacity khởi tạo 0,5; hệ số scale khởi tạo 0,1 (nhân với căn trung bình bình phương khoảng cách tới 3 điểm SfM gần nhất); `opacity_reg` = `scale_reg` = 0,01 | `cap_max` = số Gaussian cuối của lượt `default` cùng cảnh, cùng seed, cùng protocol, cùng tập test | So sánh công bằng với baseline |
| `mcmc_demo` | `mcmc` | Như `mcmc`, cộng thêm `max_steps` 40.000, `refine_stop_iter` 35.000, antialiased | `cap_max` = 3.500.000 (cố định) | Tái hiện cấu hình demo; **không** vào bảng so sánh công bằng |

Tham số riêng của chế độ được đổi qua `--set`, ví dụ `--set opacity_reg=0.001` (giá trị paper dùng cho Deep Blending). Thay đổi này được ghi vào `run_meta.json`.

## 4. Giao thức so sánh

### 4.1 Protocol chung

| Khóa | Giá trị | Lý do |
|---|---|---|
| `data_factor` | 1 | Giữ độ phân giải gốc. Inria tự thu ảnh rộng hơn 1.600 px về 1.600 px, gsplat thì không. Vì vậy cảnh tự quay nên trích frame với `--max-side 1600`; `indoor3d.train.run` cảnh báo nếu ảnh rộng hơn |
| `test_every` | 8 | Cùng quy tắc với Inria (`llffhold=8`) và gsplat: sắp xếp tên ảnh, ảnh thứ i là test khi i % 8 == 0. Đã đối chiếu mã nguồn cả hai, nên hai bản dùng **cùng tập test** |
| `max_steps` | 30.000 | Như Inria và code chính chủ 3DGS-MCMC |
| `lpips_net` | `vgg` | Như Inria; gsplat mặc định AlexNet nên phải đặt rõ |
| `antialiased` | `false` | Inria mặc định tắt; gsplat ghi chú antialiased có thể làm giảm nhẹ chỉ số |
| `save_ply` | `true` | Đo dung lượng mô hình; dùng cho viewer |
| `disable_video` | `false` | Video quỹ đạo để đánh giá định tính |
| `init_type` | `sfm` | Cả hai chế độ khởi tạo từ điểm SfM, như B0. Code chính chủ 3DGS-MCMC mặc định khởi tạo ngẫu nhiên; paper báo cả hai kiểu (Bảng 1, Bảng 5), nên khi đối chiếu với paper phải lấy các dòng khởi tạo SfM |

Đổi một khóa protocol (trong file cấu hình hoặc bằng `--protocol`) sẽ được ghi thành `deviations`. Bảng so sánh đánh dấu lượt chạy đó là khác điều kiện.

### 4.2 Các quy tắc khác

- **Cùng ngân sách Gaussian.** Chất lượng của MCMC phụ thuộc trực tiếp vào `cap_max`. Paper so sánh ở cùng số Gaussian với 3DGS (mục 4.1, Bảng 1 và Bảng 5), nên chế độ `mcmc` lấy `cap_max` bằng số Gaussian cuối của lượt `default` tương ứng.
  - `run.py` từ chối nếu lượt baseline chưa xong, sai chế độ, khác protocol, có sai khác protocol, khác seed hoặc dùng tập test khác.
  - `compare_modes` ghi `gaussian_budget: EQUAL/UNEQUAL`.
- **Đánh giá tách khỏi huấn luyện**, giống script benchmark của gsplat.
  - Khi huấn luyện: `--eval-steps -1`. Sau đó một lệnh riêng nạp checkpoint cuối (`--ckpt`) với đúng cấu hình đã huấn luyện.
  - Nhờ vậy thời gian huấn luyện không gồm thời gian đánh giá, và mọi chế độ dùng cùng một đoạn mã tính chỉ số.
- **Một GPU, cùng loại GPU.**
  - Nếu thấy nhiều GPU, gsplat tự chia sang huấn luyện phân tán (tăng batch thực tế). Vì vậy `run.py` đặt `CUDA_VISIBLE_DEVICES` về một GPU, còn launcher từ chối khi thấy nhiều hơn một.
  - Tên GPU được ghi vào kết quả và có trong mục kiểm tra điều kiện.
- **Seed.** `simple_trainer.py` gán cứng seed 42. Launcher cho phép đổi seed để chạy lặp. Trước khi nói "cải thiện" cần ít nhất 3 seed trên một tập cảnh (`docs/plan.md` mục 4.2).
  - Seed cố định thứ tự ảnh huấn luyện (DataLoader xáo trộn), cùng bộ sinh số ngẫu nhiên dùng cho nhiễu và relocate của MCMC và cho bước tách Gaussian của Default.
  - Một số kernel CUDA không tất định (cộng dồn gradient song song), nên hai lần chạy cùng seed vẫn có thể lệch nhau một chút.
- **Tập test của cảnh tự quay.** Frame test xen kẽ trong cùng một video chỉ đo khả năng nội suy (xem demo_review mục 3.3).
  - `--test-list <file>` (mỗi dòng một tên ảnh, cùng định dạng `sparse/0/test.txt` của Inria) cho phép đánh giá trên đoạn quay kiểm tra riêng.
  - Đoạn quay kiểm tra phải được đăng ký trong cùng model COLMAP với video chính để có pose. Thêm bước chuẩn bị dữ liệu này vào `prepare_scene.sh` thuộc mốc M4.
  - Khi đó điểm SfM có thể chỉ được dựng nhờ ảnh test. `run.py` tạo bản dữ liệu `<lượt chạy>/data/`: ảnh và pose giữ nguyên, nhưng `points3D.bin` chỉ giữ điểm được ít nhất 2 ảnh train khác nhau quan sát, nên Gaussian không được khởi tạo từ thông tin riêng của ảnh test. Số điểm giữ lại được ghi vào `run_meta.json`.
  - gsplat vẫn chuẩn hóa hệ tọa độ theo mọi camera, kể cả camera test. Việc này chỉ dùng pose, không dùng nội dung ảnh test; các benchmark công khai cũng làm như vậy.
  - Với B0, code Inria chưa đọc được danh sách này; quyết định ở M4 (xem `docs/environment.md` mục 6).
- **Không chỉnh tham số trên ảnh test.** Nếu cần chọn `opacity_reg`, `cap_max`... cho dữ liệu tự quay, chỉ chọn trên Deep Blending hoặc trên một phần ảnh train tách ra làm tập kiểm định, không chọn theo chỉ số trên ảnh test.
- **Một thư mục kết quả cho một cấu hình.**
  - `run.py` chỉ bỏ qua lượt đã xong khi preset, protocol, tham số, seed, `cap_max` và tập test đều giống yêu cầu mới; khác thì dừng và yêu cầu `--force` (thư mục cũ được đổi tên, không xóa).
  - `train_scene.sh` bắt buộc đặt `TAG` khi dùng `--protocol` hoặc `--set`, để lượt chạy thử không ghi vào thư mục của lượt chuẩn.
  - Lượt dở dang (pod bị dừng giữa chừng) được chạy lại với `--rerun-incomplete`; hai script trên pod luôn bật tùy chọn này.

### 4.3 Khi nào được nói "cải thiện"

Chỉ khi đủ tất cả các điều kiện sau:

- mọi mục kiểm tra trong báo cáo `compare_modes` là OK, và ngân sách Gaussian là EQUAL;
- hai chế độ dùng cùng một bộ seed, ít nhất 3 seed;
- báo cáo trung bình ± độ lệch chuẩn theo từng cảnh, kèm chênh lệch theo từng cảnh, không chỉ trung bình gộp;
- chênh lệch nhất quán trên các cảnh. Nếu không nhất quán, nêu rõ cảnh nào tốt hơn, cảnh nào kém hơn.

Kết quả không cải thiện vẫn được báo cáo đầy đủ (`docs/plan.md`, rủi ro R13).

## 5. Thành phần

| File | Vai trò |
|---|---|
| `configs/train_modes.json` | Protocol chung, ba chế độ, commit gsplat mong đợi, seed mặc định |
| `src/indoor3d/train/profiles.py` | Đọc cấu hình, ghép protocol và tham số; chặn tham số do wrapper quản lý; sinh dòng lệnh cho `simple_trainer.py` (huấn luyện và đánh giá); chính sách `cap_max` |
| `src/indoor3d/train/split.py` | Chia train/test theo đúng quy tắc của Inria và gsplat, hoặc theo danh sách; ghi `split.json` kèm SHA-256 |
| `src/indoor3d/train/gsplat_launcher.py` | Chạy `simple_trainer.py` **không sửa file nào của gsplat**: đổi seed, áp tập test cho trước, ghi tập ảnh thực sự dùng và cấu hình đã phân giải; `--resolve-only` để kiểm tra tham số không cần GPU |
| `src/indoor3d/train/run.py` | Điều phối một lượt chạy: kiểm tra cảnh → split → huấn luyện → đánh giá → `summary.json`; ghi lệnh, commit, GPU, thời gian vào `run_meta.json`; `--dry-run` |
| `src/indoor3d/eval/gsplat_results.py` | Đọc kết quả gsplat (`stats/*.json`, checkpoint, PLY, nhật ký GPU) thành một bản tóm tắt |
| `src/indoor3d/eval/compare_modes.py` | Nhóm các lượt chạy theo (cảnh, biến thể). Biến thể = chế độ + tham số riêng; lượt sai khác protocol bị loại.<br>Tính trung bình ± độ lệch chuẩn theo seed và chênh lệch so với baseline.<br>Kiểm tra điều kiện: protocol, tập test, commit, GPU, bộ seed, ngân sách.<br>Đối chiếu baseline với số liệu công bố và với lượt B0 tự chạy, từng cảnh |
| `scripts/setup_gsplat.sh`, `environment/gs-gsplat.conf`, `environment/gs-gsplat.in`, `environment/gs-gsplat.lock.txt` | Env `gs-gsplat`: gsplat `6e8c837` (build sẵn extension), fused-ssim `328dc98`, phụ thuộc ghim theo ngày |
| `scripts/smoke_test_gsplat.py` | Kiểm tra trên GPU: extension, rasterization, hai strategy, fused-ssim, LPIPS-VGG, tham số của hai chế độ |
| `scripts/train_scene.sh` | Một cảnh tự quay, một chế độ; tự tìm lượt `default` cùng seed cho `mcmc` |
| `scripts/run_modes_deepblending.sh` | Deep Blending: `default` → `mcmc` (cùng ngân sách) → bảng so sánh; đối chiếu `default` với lượt B0 tự chạy (từng cảnh) và với số liệu Inria công bố |

## 6. Cách chạy trên pod

```bash
bash scripts/setup_gsplat.sh                               # một lần: env gs-gsplat
bash scripts/run_modes_deepblending.sh                     # kiểm chứng trên Deep Blending (SEEDS="42 43 44" để chạy lặp)
bash scripts/train_scene.sh phong01 default                # cảnh tự quay, đã qua prepare_scene.sh
bash scripts/train_scene.sh phong01 mcmc                   # cap_max = số Gaussian cuối của default_s42
bash scripts/train_scene.sh phong01 mcmc_demo              # cấu hình của notebook demo
TAG=thu bash scripts/train_scene.sh phong01 default --protocol max_steps=3000   # chạy thử, thư mục riêng
. /workspace/miniforge3/etc/profile.d/conda.sh && conda activate gs-gsplat
PYTHONPATH=src python -m indoor3d.eval.compare_modes --root /workspace/outputs/modes \
    --modes default mcmc mcmc_demo --include-deviating --markdown /workspace/reports/modes_scenes.md
```

`mcmc_demo` khác protocol nên chỉ hiện trong bảng khi có `--include-deviating`, và luôn là biến thể riêng.

Mỗi lượt chạy ghi vào `$OUT_ROOT/<cảnh>/<chế_độ>[_<TAG>]_s<seed>/`:

| File | Nội dung |
|---|---|
| `run_meta.json` | Chế độ, protocol, sai khác, lệnh đã chạy, `cap_max` và nguồn của nó, commit repo/gsplat, GPU, thời gian huấn luyện và đánh giá, cảnh báo |
| `split.json`, `split_applied.json`, `split_applied_eval.json` | Tập train/test dự kiến và tập gsplat thực sự dùng khi huấn luyện / đánh giá |
| `resolved_config.json`, `resolved_config_eval.json` | Toàn bộ cấu hình `simple_trainer.py` sau khi phân giải, kể cả tham số strategy |
| `train.log`, `eval.log`, `gpu_log.csv` | Log và nhật ký `nvidia-smi` mỗi 2 giây |
| `stats/`, `ckpts/`, `ply/`, `renders/`, `videos/`, `tb/` | Output gốc của gsplat |
| `data/` | Chỉ có khi dùng `--test-list`: bản dữ liệu đã lọc điểm SfM (mục 4.2) |
| `summary.json` | PSNR/SSIM/LPIPS, số Gaussian, thời gian (cả tiến trình và riêng vòng lặp huấn luyện), VRAM đỉnh (nvidia-smi và PyTorch), thời gian render mỗi ảnh, dung lượng PLY/checkpoint, kết quả kiểm tra tập ảnh |

Dung lượng: checkpoint và PLY mỗi file khoảng 236 byte mỗi Gaussian (59 số thực). Một lượt 2,5 triệu Gaussian tốn khoảng 1,2 GB trên network volume.

## 7. Khác biệt đã biết giữa Default (gsplat) và B0 (Inria)

Đây là lý do phải đối chiếu bằng số liệu, không mặc định hai bản tương đương:

- **Hàm mất SSIM:** gsplat gọi `fused_ssim(..., padding="valid")`, Inria dùng padding mặc định ("same"). Commit fused-ssim cũng khác nhau (`328dc98` và `1272e21`).
- **Chuẩn hóa cảnh:** gsplat chuẩn hóa hệ tọa độ theo toàn bộ camera và trục chính của đám điểm (`normalize_world_space`). Inria tính bán kính cảnh từ camera huấn luyện.
- **Mã tính chỉ số:** gsplat dùng torchmetrics (PSNR, SSIM, LPIPS-VGG với `normalize=False`; gsplat ghi đây là cách tương đương code Inria). B0 dùng `metrics.py` của Inria.
  - So sánh `default` với `mcmc` dùng cùng một đoạn mã nên không bị ảnh hưởng.
  - So sánh B1 với B0 có thể lệch nhỏ do mã tính, nên dùng ngưỡng chấp nhận như ở `docs/environment.md` mục 5.

## 8. Đã kiểm chứng và chưa

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Logic cấu hình, split, điều phối, tổng hợp, so sánh | Đã kiểm thử (CPU) | 41 test mới trong `tests/` (tổng 66 test đều qua). Dùng gsplat giả lập và runner giả |
| File khóa phụ thuộc | Đã chạy (CPU) | Trong sandbox ngày 09/10/2026: cài vào Python 3.10 cùng torch 2.7.1 (PyPI). `examples/simple_trainer.py` của gsplat `6e8c837` import được |
| Tham số sinh ra so với `simple_trainer.py` thật | Đã chạy (CPU) | Launcher `--resolve-only` cho `default`, `mcmc`, `mcmc_demo` (huấn luyện và đánh giá). Strategy, `cap_max`, `max_steps`, `eval_steps [-1]`, `lpips_net`, antialiased ra đúng như mục 3 |
| Áp tập test và seed lên `Parser`/`Dataset` thật | Đã chạy (CPU) | Cảnh COLMAP tổng hợp 17 ảnh, đọc bằng fork pycolmap thật. Quy tắc mỗi ảnh thứ 8 và danh sách cho trước đều cho đúng tập ảnh; seed 42 → 7 đúng. Bản dữ liệu đã lọc điểm (`data/`) đọc được: 185/272 điểm |
| Gói gsplat đang cài trùng mã nguồn checkout | Đã chạy (CPU) | Launcher so SHA-256 của `strategy/{default,mcmc,ops}.py` và `rendering.py`; bản cài từ checkout `6e8c837` khớp. Test giả lập cho trường hợp lệch |
| Rà soát độc lập | Đã làm | Một lượt rà soát riêng (không tham gia viết code) đã chỉ ra 4 vấn đề lớn: bỏ qua lượt chạy khác cấu hình; trộn các biến thể khi so sánh; điểm SfM chỉ có nhờ ảnh test; đối chiếu với số công bố thay vì với B0 tự chạy. Cả 4 vấn đề đã được sửa, kèm test; các góp ý nhỏ đã được sửa hoặc ghi thành quy tắc ở mục 4 |
| Script shell | Đã kiểm tra tĩnh; đã chạy giả lập | `shellcheck` sạch. `train_scene.sh` và `run_modes_deepblending.sh` chạy được với conda / `nvidia-smi` giả và chế độ dry-run |
| Build extension, smoke test GPU, huấn luyện, số liệu | **Chưa chạy** | Chờ lượt Runpod (quyết định D3) |
