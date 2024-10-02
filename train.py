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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, HaLoss
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from render import render_set
from metrics import evaluate as evaluate_metrics
import torchvision

import random
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(sh_degree=dataset.sh_degree, mlp_depth=opt.mlp_depth, mlp_width=opt.mlp_width, frgb=opt.frgb, type=dataset.source_path, mask=opt.mask_unet, a_use=opt.a_enc)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_temp_path=os.path.join(dataset.model_path,"train_temp_rendering")
    gt_temp_path=os.path.join(dataset.model_path,"train_temp_gt")
    os.makedirs(render_temp_path,exist_ok=True)
    os.makedirs(gt_temp_path,exist_ok=True)
    if opt.mask_unet:
        render_temp_mask_path=os.path.join(dataset.model_path,"train_mask_temp_rendering")
        os.makedirs(render_temp_mask_path,exist_ok=True) 

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    if opt.a_enc:
        embedding_a_list = [None] * 2000
    haloss = HaLoss(opt)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        input = {}
        gt_image = viewpoint_cam.original_image.cuda()
        if opt.mask_unet:
            input['out_mask'] = gaussians.get_mask(gt_image.unsqueeze(0)*2-1).squeeze(0) # (1, h, w)
        if opt.a_enc:
            enc_a = gaussians.a_encoder(viewpoint_cam.original_image_8.cuda().unsqueeze(0))
        else:
            enc_a = None
        # render img from enc_a
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, a_use=opt.a_enc, enc_a_r=enc_a)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        input['rgb_fine'] = image
        if opt.a_enc:
            enc_a = render_pkg["enc_a"]
            input['a_embedded'] = enc_a
            # input['re_a_embedded'] = gaussians.a_encoder(image.unsqueeze(0) * 2 - 1)
            if opt.a_random:
                # render img from random enc_a
                idexlist = [k for k,v in enumerate(embedding_a_list) if v != None]
                if len(idexlist) == 0:
                    enc_a_r = enc_a
                else:
                    enc_a_r = embedding_a_list[random.choice(idexlist)]
                image_random = render(viewpoint_cam, gaussians, pipe, bg, enc_a_r = enc_a_r, a_use=opt.a_enc)["render"]
                enc_a_r_res = gaussians.a_encoder(image_random.unsqueeze(0) * 2 - 1)
                embedding_a_list[viewpoint_cam.uid] = enc_a.clone().detach()
                input['a_embedded_random'] = enc_a_r
                input['a_embedded_random_rec'] = enc_a_r_res
                input['rgb_fine_random'] = image_random       
            
        h_l = haloss(input, opt, gt_image, iteration, opt.mask_unet)

        # Loss
        # Ll1 = l1_loss(gt_image, image)
        loss = sum(l for l in h_l.values()) # + 0.0005*torch.mean((torch.sigmoid(gaussians._mask)))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration%1000==0 or iteration==1:
                torchvision.utils.save_image(image, os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam.image_name + ".png"))
                torchvision.utils.save_image(gt_image, os.path.join(gt_temp_path, f"iter{iteration}_"+viewpoint_cam.image_name + ".png"))
                if opt.mask_unet:
                    torchvision.utils.save_image(input['out_mask'], os.path.join(render_temp_mask_path, f"iter{iteration}_"+viewpoint_cam.image_name + ".png"))
            if iteration % 10 == 0:            
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                           "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()    

            # if iteration == opt.iterations:
            #     gaussians.mask_prune()

            # Log and save
            training_report(tb_writer, iteration, h_l['f_l'], loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), opt.a_enc)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # specular_mlp.save_weights(dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            # else:
            #     if iteration % opt.mask_prune_iter == 0:
            #         gaussians.mask_prune()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if opt.a_enc:
                    gaussians.optimizer_2.step()
                    gaussians.optimizer_2.zero_grad(set_to_none = True)
                if opt.mask_unet:
                    gaussians.optimizer_3.step()
                    gaussians.optimizer_3.zero_grad(set_to_none = True)
                gaussians.update_learning_rate(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        print(f"Rendering testing set [{len(scene.getTestCameras())}]...")
        render_set(dataset.model_path, "test", opt.iterations + 1, scene.getTestCameras(), gaussians, pipe, background, a_use=opt.a_enc, enc_a_r=None)
        print("Evaluating metrics on testing set...")
        evaluate_metrics([dataset.model_path])

def prepare_output_and_logger(args, opt):    
    if not args.model_path:
        args.model_path = os.path.join("./output1/", args.exp_name)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        cfg_log_f.write(str(Namespace(**vars(opt))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, a_use):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpipss_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if a_use:
                        enc_a = scene.gaussians.a_encoder(viewpoint.original_image_8.cuda().unsqueeze(0))
                    else:
                        enc_a = None
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, a_use = a_use, enc_a_r = enc_a)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    res = torch.cat((image[None], gt_image[None]), dim=0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images("{}_view/render_{}".format(config['name'], viewpoint.image_name), res, global_step=iteration)
                        # if iteration == testing_iterations[0]:
                        #     tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpipss_test += lpips(image.unsqueeze(0), gt_image.unsqueeze(0), net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])        
                ssim_test /= len(config['cameras']) 
                lpipss_test /= len(config['cameras'])   
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} lpipss {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpipss_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpipss', lpipss_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 21_000, 24_000, 27_000, 30_000, 33_000, 36_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
