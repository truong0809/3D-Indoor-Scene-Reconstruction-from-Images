"""Kiểm tra nhanh các CUDA extension của 3DGS gốc (Inria) sau khi build.

Chạy trên pod có GPU, trong conda env đã cài:
    python scripts/smoke_test_3dgs.py

Các kiểm tra:
  1. PyTorch nhận GPU.
  2. simple_knn: tính khoảng cách láng giềng gần nhất trên GPU.
  3. fused_ssim: SSIM(x, x) xấp xỉ 1 và có gradient.
  4. diff_gaussian_rasterization: render vài trăm Gaussian ngẫu nhiên, có điểm
     hiển thị và gradient hữu hạn khi lan truyền ngược.

Thoát với mã 0 nếu tất cả đạt, mã 1 nếu có kiểm tra thất bại.
"""

import math
import sys

import torch


def check_cuda():
    assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
    props = torch.cuda.get_device_properties(0)
    return f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {props.name}"


def check_simple_knn():
    from simple_knn._C import distCUDA2

    points = torch.rand(1000, 3, device="cuda")
    dist2 = distCUDA2(points)
    assert dist2.shape == (1000,), f"unexpected shape {tuple(dist2.shape)}"
    assert torch.isfinite(dist2).all() and (dist2 >= 0).all(), "invalid distances"
    return f"mean squared NN distance {dist2.mean().item():.3e}"


def check_fused_ssim():
    from fused_ssim import fused_ssim

    target = torch.rand(1, 3, 64, 64, device="cuda")
    same = fused_ssim(target.clone().requires_grad_(True), target).item()
    assert abs(same - 1.0) < 1e-3, f"SSIM(x, x) = {same}"

    pred = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
    fused_ssim(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "invalid SSIM gradient"
    return f"SSIM(x, x) = {same:.4f}"


def projection_matrix(znear, zfar, fov_x, fov_y):
    """Same formula as utils/graphics_utils.getProjectionMatrix in the Inria repo."""
    top = math.tan(fov_y / 2) * znear
    right = math.tan(fov_x / 2) * znear
    bottom, left = -top, -right
    proj = torch.zeros(4, 4)
    proj[0, 0] = 2.0 * znear / (right - left)
    proj[1, 1] = 2.0 * znear / (top - bottom)
    proj[0, 2] = (right + left) / (right - left)
    proj[1, 2] = (top + bottom) / (top - bottom)
    proj[3, 2] = 1.0
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    return proj


def check_rasterizer():
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    height = width = 128
    fov = math.radians(60.0)
    # Camera at the origin looking along +z (COLMAP convention used by 3DGS).
    # 3DGS stores matrices transposed, as in scene/cameras.py.
    world_view = torch.eye(4, device="cuda")
    proj_t = projection_matrix(0.01, 100.0, fov, fov).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(proj_t.unsqueeze(0)).squeeze(0)

    settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fov / 2),
        tanfovy=math.tan(fov / 2),
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=full_proj,
        sh_degree=0,
        campos=torch.zeros(3, device="cuda"),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    torch.manual_seed(0)
    n = 500
    means3d = torch.empty(n, 3, device="cuda").uniform_(-1.0, 1.0)
    means3d[:, 2] = 2.0 + 2.0 * torch.rand(n, device="cuda")  # depth in [2, 4]
    means3d.requires_grad_(True)
    means2d = torch.zeros_like(means3d, requires_grad=True)
    opacities = torch.full((n, 1), 0.8, device="cuda", requires_grad=True)
    colors = torch.rand(n, 3, device="cuda", requires_grad=True)
    scales = torch.full((n, 3), 0.05, device="cuda", requires_grad=True)
    rotations = torch.zeros(n, 4, device="cuda")
    rotations[:, 0] = 1.0  # identity quaternion (w, x, y, z)
    rotations.requires_grad_(True)

    image, radii, invdepth = rasterizer(
        means3D=means3d,
        means2D=means2d,
        opacities=opacities,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
    )
    assert image.shape == (3, height, width), f"unexpected image shape {tuple(image.shape)}"
    visible = int((radii > 0).sum().item())
    assert visible > 0, "no Gaussian is visible"
    assert torch.isfinite(image).all(), "image contains NaN/Inf"

    (image.mean() + invdepth.mean()).backward()
    for name, tensor in (("means3D", means3d), ("colors", colors),
                         ("opacities", opacities), ("scales", scales)):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), f"invalid gradient for {name}"
    assert means3d.grad.abs().sum().item() > 0, "zero gradient for means3D"
    return f"{visible}/{n} Gaussians visible, mean pixel value {image.mean().item():.4f}"


def main():
    checks = [
        ("PyTorch sees the GPU", check_cuda),
        ("simple_knn (distCUDA2)", check_simple_knn),
        ("fused_ssim", check_fused_ssim),
        ("diff_gaussian_rasterization forward/backward", check_rasterizer),
    ]
    failed = 0
    for name, fn in checks:
        try:
            detail = fn()
            print(f"[PASS] {name}: {detail}")
        except Exception as exc:  # report every failure, do not stop at the first
            failed += 1
            print(f"[FAIL] {name}: {exc!r}")
    print("SMOKE TEST:", "PASSED" if failed == 0 else f"FAILED ({failed} check(s))")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
