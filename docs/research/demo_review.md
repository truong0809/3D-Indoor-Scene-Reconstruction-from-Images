# Kiểm tra bản demo 3DGS-MCMC và repo SpatialScene3D

_Ngày kiểm tra: 09/10/2026 · Phương pháp: đọc mã và output đã lưu; không chạy lại notebook (cần GPU)._

## 1. Nguồn được kiểm tra

| Nguồn | Phiên bản | Ghi chú |
|---|---|---|
| Notebook demo của nhóm | `references/demo_3dgs_mcmc/3D_demo.ipynb`, SHA-256 `618277f5…` | 11 cell, kèm output một lần chạy ngày 03/10/2026 trên Colab (A100-SXM4-40GB, PyTorch 2.11.0+cu130) |
| [Ziro21/SpatialScene3D](https://github.com/Ziro21/SpatialScene3D) | commit `64e9e88` (08/06/2026) | Nguồn tham khảo của demo; dùng gsplat 1.3.0 |
| Paper 3DGS-MCMC: Kheradmand et al., *3D Gaussian Splatting as Markov Chain Monte Carlo*, NeurIPS 2024 | [arXiv 2404.09591](https://arxiv.org/abs/2404.09591) (v3), [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract.html) | Code chính chủ [ubc-vision/3dgs-mcmc](https://github.com/ubc-vision/3dgs-mcmc) (commit `7b4fc9f`), dựng trên code 3DGS của Inria |
| [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | v1.3.0, v1.5.3, `6e8c837`, `9b9f98a` | Đối chiếu mã nguồn các phiên bản mà demo và pipeline dùng |

## 2. Notebook demo làm gì (theo mã và output đã lưu)

| Bước | Cách làm trong notebook | Kết quả trong output |
|---|---|---|
| Trích frame | ffmpeg `fps=5`, chiều cao 1080 px, JPEG `-q:v 2`; bỏ các frame có phương sai Laplacian dưới phân vị 10; lưu lại thành PNG | 142 frame (bỏ 16) |
| SfM | COLMAP 3.9.1 cài từ apt, trích đặc trưng trên CPU, camera mặc định SIMPLE_RADIAL, `exhaustive_matcher`, `mapper`; chọn model nhiều ảnh nhất; không undistort | 142/142 ảnh đăng ký, 28.381 điểm, 1 model |
| Cài gsplat | `pip install gsplat==1.3.0`; clone tag v1.3.0; thay `examples/datasets/colmap.py` bằng bản ở commit `9b9f98a` (15/05/2026); tải `exif.py` từ nhánh `main`; dùng `sed` đổi tên hàm `align_principal_axes` | Extension biên dịch JIT |
| Huấn luyện | `simple_trainer.py mcmc --data_factor 1 --max_steps 40000 --strategy.cap-max 3500000 --strategy.refine-stop-iter 35000 --opacity_reg 0.01 --scale_reg 0.01 --antialiased` | 44 phút 12 giây, gồm cả đánh giá và render video ở bước cuối |
| Đánh giá | gsplat tự giữ mỗi ảnh thứ 8 làm tập val (18 ảnh) | PSNR 39,969 · SSIM 0,9818 · LPIPS 0,037 · 3.500.000 Gaussian · 0,026 s/ảnh |
| Xuất | Bỏ Gaussian có opacity < 0,005, có scale hoặc khoảng cách tới tâm trên phân vị 99,9%; ghi `.ply` chuẩn 3DGS và `.splat` 32 byte/Gaussian | 3,5M → 2.016.234 Gaussian; `.ply` 500 MB; `.splat` 64,5 MB |
| Viewer | Trang HTML dùng gsplat.js 1.2.9 tải từ CDN, phục vụ bằng `http.server` | — |

## 3. Vấn đề phát hiện

### 3.1 Trộn phiên bản gsplat, không tái lập được

Thư viện là bản 1.3.0 (08/2024), parser dữ liệu lấy từ một commit tháng 05/2026, `exif.py` lấy từ nhánh `main` (thay đổi theo thời gian), cộng thêm một lệnh `sed`.

- Nguyên nhân gốc: notebook cài `pycolmap` bản chính thức, không ghim phiên bản. Trong khi đó `examples/` của gsplat từ v1.3.0 đến v1.5.3 dùng fork `rmbrualla/pycolmap@cc7ea4b` (có lớp `SceneManager`). Bản parser ở `9b9f98a` mới chuyển sang `pycolmap` chính thức.
- Hệ quả: notebook có thể hỏng bất cứ lúc nào `main` thay đổi, và không ai dựng lại được đúng môi trường đã cho ra kết quả.
- **Xử lý trong pipeline:** thư viện và `examples/` lấy từ cùng một commit được ghim. Fork pycolmap được cài đúng commit mà gsplat yêu cầu. Toàn bộ phụ thuộc Python ghim trong `environment/gs-gsplat.lock.txt`.

### 3.2 Lỗi trong các bản phát hành gsplat ảnh hưởng tới baseline Default

| Phiên bản | Hành vi reset opacity của `DefaultStrategy` | So với 3DGS gốc |
|---|---|---|
| v1.3.0 (bản demo dùng) | Reset cả ở bước 0, nên opacity khởi tạo 0,1 bị kẹp xuống 0,01 ngay vòng đầu | Lệch; sửa ở PR #735 |
| v1.5.3 (bản phát hành mới nhất) | Điều kiện `step % reset_every == 0 & step > 0`. Toán tử `&` được ưu tiên hơn `==` và `>`, nên biểu thức luôn sai và **không bao giờ reset** | Lệch nhiều; sửa ở PR #776 (commit `6e8c837`, 22/08/2025) |
| `6e8c837` (pipeline dùng) | Reset ở 3000, 6000, 9000, 12000, như code Inria | Khớp |

Demo chạy MCMC nên không bị lỗi này ảnh hưởng. Nhưng chế độ Default là baseline, nên pipeline ghim `6e8c837`. Commit này bằng v1.5.3 cộng 4 commit: định dạng `setup.py`, thêm script nhiều GPU, sửa một thông báo và sửa lỗi trên. File `gsplat/strategy/mcmc.py` giống hệt nhau từ v1.3.0 đến `6e8c837`; các hàm dùng chung trong `ops.py` chỉ đổi cách giữ thuộc tính `requires_grad`.

### 3.3 Chỉ số của demo không dùng được để so sánh

- **LPIPS khác mạng:** gsplat mặc định `lpips_net="alex"`, notebook không đổi. LPIPS 0,037 là LPIPS-AlexNet, không so được với LPIPS-VGG mà Inria dùng (0,238 trên Deep Blending). Pipeline đặt `lpips_net=vgg`, đúng cách gsplat ghi là tương đương với code Inria.
- **Tập test là nội suy giữa các frame gần nhau:** 18 ảnh test xen kẽ trong cùng một video 5 fps. Mỗi ảnh test chỉ cách ảnh train gần nhất 0,2 giây.
  - Cách chia này giống benchmark (mỗi ảnh thứ 8), nên không phải "ảnh test lọt vào tập train".
  - Nhưng nó đo khả năng nội suy giữa những khung gần như trùng nhau, không đo chất lượng khi nhìn từ góc khác. PSNR ~40 dB vì vậy lạc quan.
  - Pipeline ghi lại tập test của mọi lượt chạy, và hỗ trợ đánh giá trên đoạn quay kiểm tra riêng (`--test-list`).
- **Không có đối chứng:** chỉ có một lần chạy MCMC, không có lần chạy Default trên cùng dữ liệu. Vì vậy chưa thể nói MCMC tốt hơn.

### 3.4 Cấu hình huấn luyện khác giao thức của paper

- Notebook chạy 40.000 vòng (code chính chủ của 3DGS-MCMC và gsplat dùng 30.000) và bật antialiased. gsplat ghi chú rằng antialiased "might slightly hurt quantitative metrics".
- Notebook đặt `cap_max` = 3,5 triệu, không gắn với baseline. Paper so sánh ở cùng số Gaussian với 3DGS (mục 4.1: "we simply set the number of Gaussians used in the original 3DGS [19]" … "set it as our maximum number of Gaussians to be used during training and inference"; Bảng 1 và Bảng 5 ghi "same number of Gaussians"). README của code chính chủ cũng ghi `cap_max` lấy theo số Gaussian cuối của lần chạy 3DGS gốc trên từng cảnh.
- Paper dùng λ_opacity = 0,001 cho Deep Blending và 0,01 cho các tập khác. Config chính chủ chỉ đặt 0,001 cho `drjohnson`; `playroom` dùng giá trị mặc định 0,01.
- Quan sát thêm, chưa phải kết luận: ở checkpoint cuối, trung vị opacity là 0,012. Bước xuất bỏ 1,48 triệu trên 3,5 triệu Gaussian, gần như toàn bộ do ngưỡng opacity 0,005 (hai ngưỡng phân vị 99,9% chỉ bỏ khoảng 0,1% mỗi ngưỡng). Có thể ngân sách 3,5 triệu lớn hơn mức cảnh cần; cần thực nghiệm để kiểm chứng.

### 3.5 Các điểm khác

- **COLMAP:** notebook dùng tên tùy chọn cũ `--SiftExtraction.use_gpu`, không chạy được với COLMAP ≥ 3.13. Notebook cũng không undistort: gsplat tự undistort trong parser, nhưng baseline Inria cần ảnh PINHOLE. Pipeline dùng `colmap_runner.py` (COLMAP 4.2.1, có `image_undistorter`), và mọi chế độ dùng cùng một đầu vào.
- **Lọc frame:** phần chú thích ghi bỏ 5% frame mờ, code lại dùng phân vị 10. Lọc theo phân vị cũng luôn bỏ 10% frame dù video nét hay mờ. Pipeline dùng `frames.py`: chọn frame nét nhất trong mỗi cửa sổ, chỉ loại frame quá mờ so với trung vị.
- **Xuất cho web:** file `.ply` 500 MB quá nặng; viewer phụ thuộc CDN nên cần mạng khi bảo vệ. Hai việc này để lại cho Bước 6.
- **Phụ thuộc Colab:** `drive.mount`, `files.upload`, đường dẫn `/content`, và tìm trainer bằng `glob` (lấy kết quả đầu tiên) không đưa vào pipeline.

## 4. SpatialScene3D — những điểm liên quan

- **Mục tiêu khác:** gán nhãn ngữ nghĩa (Grounded-SAM 2, CLIP) và lớp QA tự đánh giá. Phần liên quan tới đồ án chỉ là COLMAP → gsplat.
- **3DGS ở đây không còn là Default Strategy gốc.** Notebook `notebook_v10_5.ipynb` (cell 17) sửa trực tiếp file `gsplat/strategy/default.py` của thư viện đã cài:
  - `reset_every=99999`: tắt reset opacity;
  - `refine_stop_iter=99999`;
  - `prune_opa=0.001`, `grow_grad2d=0.00005`, `prune_scale3d=0.5`, `grow_scale3d=0.005`, `refine_start_iter=200`, `absgrad=True`.

  Không dùng cấu hình này làm baseline.
- **Cùng kiểu trộn phiên bản** như demo: parser từ `9b9f98a`, `exif.py` từ `main`, lệnh `sed`.
- **Đánh giá tự viết:**
  - lọc `images.bin` và `points3D.bin` của model COLMAP chung để tạo model chỉ có ảnh train (giữ mỗi ảnh thứ 10, bắt đầu từ chỉ số 9, làm test);
  - huấn luyện gsplat trên model đó với `test_every=8` mặc định, nên thêm 1/8 số ảnh train không được dùng để huấn luyện;
  - tự viết bước đánh giá, có ước lượng phép biến đổi pose sang hệ chuẩn hóa của gsplat;
  - LPIPS-AlexNet; ảnh đánh giá 291×518 px (theo output của cell 21).
- **License:** README ghi "Code under Apache 2.0", nhưng repo không có file LICENSE ở commit `64e9e88`. Không chép code; chỉ tham khảo ý tưởng, và trích dẫn nếu dùng.
- **Ý tưởng tham khảo:** lưu bằng chứng cho mỗi lần chạy (JSON chỉ số, bảng, ảnh so sánh) và cổng kiểm tra tự động trước khi công bố kết quả. Pipeline áp dụng ý tưởng này qua `summary.json`, `run_meta.json` và các mục kiểm tra điều kiện trong `compare_modes`.

## 5. Phân loại

| Thành phần của demo | Quyết định | Thay bằng / cách dùng |
|---|---|---|
| Chiến lược MCMC và bộ tham số (`mcmc` preset, `cap_max`, `refine_stop_iter`, regularization) | **Dùng lại** | Chế độ `mcmc`: protocol chung, ngân sách bằng baseline. Chế độ `mcmc_demo`: giữ đúng tham số của notebook |
| `simple_trainer.py` của gsplat | **Dùng lại**, không sửa file | Ghim `6e8c837`; mở rộng nhỏ (seed, tập test, ghi cấu hình) qua `gsplat_launcher.py` |
| Cách cài gsplat (trộn phiên bản, `sed`, `pycolmap` không ghim) | **Bỏ** | `scripts/setup_gsplat.sh`, `environment/gs-gsplat.conf`, `environment/gs-gsplat.lock.txt` |
| Trích frame, lọc mờ theo phân vị | **Bỏ** | `src/indoor3d/data/frames.py` |
| COLMAP từ apt, tên tùy chọn cũ, không undistort | **Bỏ** | `src/indoor3d/sfm/colmap_runner.py` |
| Chỉ số held-out trong output (39,97 dB) | **Không dùng làm kết quả** | Chạy lại theo giao thức ở `docs/design/training_modes.md` |
| Lọc Gaussian và ghi `.ply` / `.splat` | **Sửa rồi dùng** ở Bước 6 | Ngưỡng lọc thành tham số; có kiểm thử chuyển đổi; đo dung lượng và thời gian tải |
| Viewer gsplat.js qua CDN | **Tham khảo** cho Bước 6 | Ghim phiên bản, chạy được khi không có mạng |
| Cell riêng của Colab | **Bỏ** | Script chạy trên Runpod |

## 6. Kết luận

Demo cho thấy chuỗi video → COLMAP → gsplat MCMC → viewer web chạy được trọn vẹn trên một cảnh nội thất.

Demo chưa đủ làm bằng chứng MCMC tốt hơn baseline, vì bốn lý do:

- LPIPS tính bằng mạng khác;
- tập test là các frame xen kẽ trong cùng video;
- chỉ có một lần chạy, không có Default trên cùng dữ liệu;
- số vòng lặp, ngân sách Gaussian và antialiased khác giao thức của paper.

Pipeline giữ nguyên chiến lược MCMC và bộ tham số của demo (chế độ `mcmc_demo`). Bên cạnh đó, pipeline thêm chế độ `mcmc` chạy cùng điều kiện với Default, trên môi trường ghim phiên bản và cùng một đầu vào.
