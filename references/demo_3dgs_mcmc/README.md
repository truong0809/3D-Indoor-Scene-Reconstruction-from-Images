# Notebook demo 3DGS-MCMC của nhóm (tài liệu tham khảo)

| | |
|---|---|
| File | `3D_demo.ipynb` — giữ nguyên bản nhận được, kể cả output của lần chạy trên Colab |
| Nhận ngày | 09/10/2026 |
| SHA-256 | `618277f58b878a940500b5a49f5d51fa911d23862effbd17683a6efd87d22380` |
| Nguồn tham khảo của demo | [Ziro21/SpatialScene3D](https://github.com/Ziro21/SpatialScene3D) và paper 3DGS-MCMC ([arXiv 2404.09591](https://arxiv.org/abs/2404.09591)) |
| Môi trường đã chạy (theo output) | Google Colab, NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu130, gsplat 1.3.0, COLMAP 3.9.1 (apt) |

Notebook này **không** thuộc pipeline và không được chạy trong repo. Nó được lưu để đối chiếu.

- Kết quả kiểm tra và phân loại tái sử dụng / sửa / bỏ: [docs/research/demo_review.md](../../docs/research/demo_review.md).
- Phần huấn luyện MCMC của demo được tích hợp thành hai chế độ `mcmc` và `mcmc_demo`: [docs/design/training_modes.md](../../docs/design/training_modes.md).
- Chỉ số trong output của notebook (PSNR 39,97 dB, SSIM 0,982, LPIPS 0,037) **không** được dùng làm kết quả của đồ án: LPIPS tính bằng AlexNet, ảnh test xen kẽ trong cùng video và chỉ có một lần chạy, không có baseline trên cùng dữ liệu.
