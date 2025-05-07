#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from skimage.draw import polygon

render_cube = False
# 定义立方体的最小和最大顶点坐标（超参数）
minxyz = torch.tensor([-0.1, -0.1, -0.1], device="cuda")  # 最小顶点
maxxyz = torch.tensor([0.1, 0.1, 0.1], device="cuda")      # 最大顶点

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建零张量，用于让PyTorch返回2D（屏幕空间）均值的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果提供了预计算的3D协方差，则使用它。否则，将由光栅化器根据缩放/旋转计算。
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预计算的颜色，则使用它们。否则，如果需要在Python中从SH预计算颜色，则执行。否则，SH -> RGB转换将由光栅化器完成。
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 将可见的高斯分布光栅化为图像，获取它们的半径（在屏幕上）
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    if render_cube:
        # 根据 minxyz 和 maxxyz 生成立方体的8个顶点
        cube_vertices = torch.tensor([
            [minxyz[0], minxyz[1], minxyz[2]],  # 底面左下后
            [maxxyz[0], minxyz[1], minxyz[2]],  # 底面右下后
            [maxxyz[0], maxxyz[1], minxyz[2]],  # 底面右上后
            [minxyz[0], maxxyz[1], minxyz[2]],  # 底面左上后
            [minxyz[0], minxyz[1], maxxyz[2]],  # 顶面左下前
            [maxxyz[0], minxyz[1], maxxyz[2]],  # 顶面右下前
            [maxxyz[0], maxxyz[1], maxxyz[2]],  # 顶面右上前
            [minxyz[0], maxxyz[1], maxxyz[2]]   # 顶面左上前
        ], device="cuda")

        # 将立方体顶点投影到屏幕空间
        homogenous_coords = torch.cat([cube_vertices, torch.ones(cube_vertices.shape[0], 1, device="cuda")], dim=1)
        screen_coords = (viewpoint_camera.full_proj_transform @ homogenous_coords.T).T
        screen_coords = screen_coords[:, :2] / screen_coords[:, 3:4]

        # 将屏幕坐标转换为像素坐标
        # 假设屏幕坐标范围是 [-1, 1]，需要映射到 [0, image_width] 和 [0, image_height]
        screen_coords[:, 0] = (screen_coords[:, 0] + 1) * 0.5 * rendered_image.shape[2]  # x 坐标
        screen_coords[:, 1] = (1 - screen_coords[:, 1]) * 0.5 * rendered_image.shape[1]  # y 坐标
        # print(screen_coords)
        

        # print(screen_coords)

        # 定义立方体的6个面（四边形）
        cube_faces = [
            [0, 1, 2, 3],  # 底面
            [4, 5, 6, 7],  # 顶面
            [0, 1, 5, 4],  # 前面
            [2, 3, 7, 6],  # 后面
            [0, 3, 7, 4],  # 左面
            [1, 2, 6, 5]   # 右面
        ]

        # 创建一个掩码，用于标记立方体在屏幕上的区域
        cube_mask = torch.zeros(rendered_image.shape[1:], device="cuda")

        for face in cube_faces:
            # 获取当前面的4个顶点在屏幕上的2D坐标
            face_coords = screen_coords[face]

            # 创建一个当前面的掩码
            face_mask = torch.zeros(rendered_image.shape[1:], device="cuda")

            # 使用多边形填充算法填充当前面的区域
            rr, cc = polygon(face_coords[:, 1].cpu().numpy(), face_coords[:, 0].cpu().numpy(), shape=rendered_image.shape[1:])
            face_mask[rr, cc] = 1

            # 将当前面的掩码合并到立方体掩码中
            cube_mask = torch.max(cube_mask, face_mask)

        # if torch.all(cube_mask == 0):
        #     print("警告: cube_mask 全为 0，立方体区域未正确投影或未覆盖任何像素。")
        # else:
        #     # 统计 cube_mask 中非零元素的数量
        #     non_zero_count = torch.sum(cube_mask != 0).item()
        #     print(f"cube_mask 中有 {non_zero_count} 个非零像素。")
        # 在立方体区域内应用红色滤镜
        red_filter = torch.tensor([1.0, 0.0, 0.0], device="cuda")  # 红色滤镜
        alpha = 0.5  # 透明度

        # 将红色滤镜应用到立方体区域，同时保留原始图像的细节
        rendered_image[:, cube_mask == 1] = alpha * red_filter[:, None] + (1 - alpha) * rendered_image[:, cube_mask == 1]

    
    # 测试代码：将整个屏幕加上红色滤镜
    # red_filter = torch.tensor([1.0, 0.0, 0.0], device="cuda")  # 红色滤镜 [3]
    # red_filter = red_filter[:, None, None]  # 扩展为 [3, 1, 1]
    # alpha = 0.5  # 透明度
    # rendered_image[:] = alpha * red_filter + (1 - alpha) * rendered_image[:]

    # print(rendered_image.shape)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # 那些被视锥剔除或半径为0的高斯分布不可见。
    # They will be excluded from value updates used in the splitting criteria.
    # 它们将被排除在用于分割标准的更新值之外。
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
