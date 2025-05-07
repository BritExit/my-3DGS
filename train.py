
# 导入必要的库
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# 导入必要的库
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 导入绘图库
import matplotlib.pyplot as plt

# 函数：绘制损失历史、移动平均损失和标准差图像并保存
def plot_and_save_loss_history(loss_history, loss_ma_history, loss_std_history, model_path, filename="loss_history.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Loss History", color="blue", alpha=0.7)
    plt.plot(loss_ma_history, label="Loss Moving Average", color="red", linestyle="--", alpha=0.7)
    
    # 绘制标准差曲线
    plt.plot(loss_std_history, label="Loss Standard Deviation", color="green", linestyle=":", alpha=0.7)
    
    # 获取MA和STD数组最末尾的值
    final_ma = loss_ma_history[-1]
    final_std = loss_std_history[-1]
    
    # 在图像上添加MA和STD最末尾的值
    plt.text(0.95, 0.95, f"Final MA: {final_ma:.5f}\nFinal STD: {final_std:.5f}", 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss History, Moving Average, and Standard Deviation")
    plt.legend()
    plt.grid(True)
    
    # 保存图像到模型路径
    save_path = os.path.join(model_path, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss history plot saved to {save_path}")

progressive_training = False
min_resolution_scale = 0.2
max_resolution_scale = 1.0
window_size = int(50)

def save_history(history, name, file_path):
    with open(file_path, 'a') as f:  # 以追加模式打开文件
        f.write(name + ":\n")
        for loss in history:
            f.write(f"{loss}\n")

def already_convergence():
    pass

def print_memory_stats(prefix=""):
    print(f"{prefix} Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB | "
          f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB | "
          f"Max Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

# 训练函数，负责整个训练过程的执行
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    # 准备输出和日志记录器
    tb_writer = prepare_output_and_logger(dataset)
    # 初始化高斯模型
    gaussians = GaussianModel(dataset.sh_degree)
    # 初始化场景
    scene = Scene(dataset, gaussians)
    # 设置训练参数
    gaussians.training_setup(opt)
    # 如果有检查点，加载模型参数
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 初始化CUDA事件，用于计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # 初始化进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    # 添加：初始化分辨率相关变量
    current_resolution_scale = min_resolution_scale  # 当前分辨率比例
    # 添加：初始化损失移动平均相关变量
    loss_history = []  # 记录损失历史
    loss_ma_history = []
    loss_min_history = []
    loss_std_history = []  # 新增：记录损失标准差
    # window_size = window_size  # 移动平均窗口大小
    
    # 添加：初始化延迟计数器
    
    delay_after_resolution_increase = window_size * 2  # 分辨率提升后的延迟轮数
    delay_counter = delay_after_resolution_increase

    # 开始训练循环
    for iteration in range(first_iter, opt.iterations + 1):        
        # 检查网络GUI连接
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # 接收网络GUI的数据
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # 渲染图像
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                # 发送渲染图像到网络GUI
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 记录迭代开始时间
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加SH的级别
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 渲染图像
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 设置背景
        bg = torch.rand((3), device="cuda") if opt.random_background else background


        # 渲染图像并获取相关数据
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 计算损失
        gt_image = viewpoint_cam.original_image.cuda()
        # 添加：根据当前分辨率比例调整图像分辨率
        if progressive_training and max_resolution_scale - current_resolution_scale > 0.001:
            # 应用高斯模糊
            # gt_image = F.gaussian_blur(gt_image, kernel_size=15, sigma=5.0)  # 调整参数增强模糊效果
            # 计算模糊程度，current_resolution_scale 越小，模糊程度越大
            blur_sigma = 10 * (1 - current_resolution_scale)  # 调整模糊强度
            kernel_size = int(blur_sigma) * 2 + 1
            gt_image = gaussian_blur(gt_image, kernel_size=kernel_size, sigma=blur_sigma)

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        # 记录迭代结束时间
        iter_end.record()

        with torch.no_grad():
            # 更新进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


            # 添加：更新损失历史并计算移动平均
            loss_history.append(loss.item())
            if len(loss_history) >= window_size:
                window_losses = loss_history[-window_size:]
                current_ma_loss = np.mean(window_losses)
                current_std_loss = np.std(window_losses)  # 新增：计算标准差
                loss_ma_history.append(current_ma_loss)
                loss_std_history.append(current_std_loss)  # 新增：记录标准差
            else:
                # 如果数据不足，计算当前所有损失值的平均值和标准差
                current_ma_loss = np.mean(loss_history)
                current_std_loss = np.std(loss_history)  # 新增：计算标准差
                loss_ma_history.append(current_ma_loss)
                loss_std_history.append(current_std_loss)  # 新增：记录标准差

            # 计算最小损失历史
            if len(loss_min_history) > 0:
                loss_min_history.append(min(loss_min_history[-1], loss_history[-1]))
            else:
                loss_min_history.append(loss_history[-1])

            # 添加：动态调整分辨率逻辑
            # 倒计时-1
            if delay_counter > 0:
                delay_counter -= 1
            if progressive_training and len(loss_ma_history) >= 2 * window_size and max_resolution_scale - current_resolution_scale > 0.001 and delay_counter == 0:  # 确保有足够的数据
                delay_counter = delay_after_resolution_increase
                
                # current_ma = loss_ma_history[-1]  # 当前100轮移动平均
                # prev_ma = loss_ma_history[-(1 + window_size)]  # 100轮前的移动平均
                # prev_prev_ma = loss_ma_history[-(1 + 2 * window_size)]  # 200轮前的移动平均
                current_min_ma = loss_min_history[-1]
                prev_min_ma = loss_min_history[-(1 + window_size)]

                # 如果当前MA < 100轮前MA < 200轮前MA，认为仍需训练
                # if current_ma < prev_ma < prev_prev_ma:
                if prev_min_ma - current_min_ma > 0.0001:
                    pass  # 继续训练
                else:
                    # 否则，提高分辨率
                    loss_min_history[-1] = float('inf')

                    current_resolution_scale = min(max_resolution_scale, current_resolution_scale + 0.4)  # 分辨率翻倍
                    print(f"\n[ITER {iteration}] Increasing resolution to {current_resolution_scale:.2f}")  # 打印分辨率提升信息

                    # 添加：保存模型
                    print(f"[ITER {iteration}] Saving Gaussians after resolution increase")
                    scene.save(iteration)  # 保存模型
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")  # 保存检查点

            # 记录日志和保存模型
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密度化处理 -----------------------------------
            if iteration < opt.densify_until_iter:
                # 跟踪图像空间中的最大半径以进行剪枝
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # 添加合并环节
                    # gaussians.merge_high_density_leaf()
                    # gaussians.add_random_gaussian(20)
                    if iteration % (opt.densification_interval * 3) == 0:
                        gaussians.merge_high_density()

                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.reset_octree()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 密度化处理 -----------------------------------

            # 优化器步骤
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    # 将八叉树数据保存到文件
    gaussians.dump_octree_to_file("octree.json")

    # loss_history
    save_history(loss_history, "loss_history", "loss_log")
    save_history(loss_ma_history, "loss_ma_history", "loss_log")
    save_history(loss_min_history, "loss_min_history", "loss_log")
    save_history(loss_std_history, "loss_std_history", "loss_log")

        # 新增：绘制损失历史图像并保存
    # plot_and_save_loss_history(loss_history, loss_ma_history, scene.model_path)  # 新增
    plot_and_save_loss_history(loss_history, loss_ma_history, loss_std_history, scene.model_path)  # 新增

    print("conbine times = ", gaussians.conbine_cnt)
    

# 准备输出和日志记录器
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 设置输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建Tensorboard记录器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# 训练报告函数，记录训练过程中的各项指标
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 报告测试集和训练集的样本
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# 主函数，程序入口
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_000, 10_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # 初始化系统状态（随机数生成器）
    safe_state(args.quiet)

    # 启动GUI服务器，配置并运行训练
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # 训练完成
    print("\nTraining complete.")
