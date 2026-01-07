#!/usr/bin/env python3
"""
Automated generation script for WonderWorld.
Bypasses the interactive interface and generates scenes automatically using
predefined camera paths and prompts.

Usage:
    python run_automated.py --example_config config/example.yaml --num_scenes 16
    python run_automated.py --example_config config/example.yaml --prompts "scene1,scene2,scene3"
"""

import gc
import random
import time
import copy
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from datetime import datetime
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
from models.models import KeyframeGen, save_point_cloud_as_ply
from util.gs_utils import save_pc_as_3dgs, convert_pc_to_splat
from util.chatGPT4 import TextpromptGen
from util.general_utils import apply_depth_colormap, save_video
from util.utils import save_depth_map, prepare_scheduler, soft_stitching, load_example_yaml, convert_pt3d_cam_to_3dgs_cam
from util.segment_utils import create_mask_generator_repvit
from util.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from arguments import GSParams, CameraParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.loss import l1_loss, ssim
from scene.cameras import Camera
from random import randint
import cv2
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from kornia.morphology import dilation
import warnings
import os
import json
warnings.filterwarnings("ignore")

xyz_scale = 1000
background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda')


def camera_to_view_matrix(camera, xyz_scale=1000):
    """
    Convert PyTorch3D camera to view matrix (16-element list).
    This is the inverse of get_camera_by_js_view_matrix.
    
    The process reverses:
    1. view_matrix -> apply xy_negate -> extract R, T
    2. So: R, T -> build matrix -> apply xy_negate -> view_matrix
    """
    R = camera.R[0]  # (3, 3)
    T = camera.T[0] * xyz_scale  # (3,) - scale back up
    
    # Build 4x4 matrix from R and T
    view_mat_negate_xy = torch.eye(4, device=R.device)
    view_mat_negate_xy[:3, :3] = R
    view_mat_negate_xy[3, :3] = T
    
    # Apply xy_negate_matrix to get original view_matrix
    # (same as in get_camera_by_js_view_matrix, but in reverse)
    xy_negate_matrix = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 
                                     device=R.device, dtype=torch.float)
    view_mat = view_mat_negate_xy @ xy_negate_matrix.inverse()
    
    return view_mat.flatten().tolist()


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Running with seed: {seed}.")


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def train_gaussian(gaussians: GaussianModel, scene: Scene, opt: GSParams, save_dir: Path, initialize_scaling=True):
    iterable_gauss = range(1, opt.iterations + 1)
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras, initialize_scaling=initialize_scaling)

    for iteration in iterable_gauss:
        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        render_pkg = render(viewpoint_cam, gaussians, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        if iteration == opt.iterations:
            time.sleep(0.1)
            print(f'Iteration {iteration}, Loss: {loss.item()}')
        
        loss.backward()
        if iteration == opt.iterations:
            print(f'Final loss: {loss.item()}')

        # Use variables that related to the trainable GS
        n_trainable = gaussians.get_xyz.shape[0]
        viewspace_point_tensor_grad, visibility_filter, radii = viewspace_point_tensor.grad[:n_trainable], visibility_filter[:n_trainable], radii[:n_trainable]

        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    max_screen_size = opt.max_screen_size if iteration >= opt.prune_from_iter else None
                    camera_height = 0.0003 * xyz_scale
                    scene_extent = camera_height * 2 if opt.scene_extent is None else opt.scene_extent
                    opacity_lowest = 0.05
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opacity_lowest, scene_extent, max_screen_size)
                    gaussians.compute_3D_filter(cameras=trainCameras)
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def run_automated(config, camera_indices=None, prompts=None):
    """
    Run automated generation without interactive interface.
    
    Args:
        config: Configuration dict
        camera_indices: List of camera indices to use (None = use all from rotation_path)
        prompts: List of scene prompts (None = use GPT or rotation_path)
    """
    global xyz_scale, background
    
    seeding(config["seed"])
    example = config['example_name']

    # Load modules (same as run.py)
    segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to('cuda')

    mask_generator = create_mask_generator_repvit()

    inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config["stable_diffusion_checkpoint"],
            safety_checker=None,
            torch_dtype=torch.bfloat16,
        ).to(config["device"])
    inpainter_pipeline.scheduler = DDIMScheduler.from_config(inpainter_pipeline.scheduler.config)
    inpainter_pipeline.unet.set_attn_processor(AttnProcessor2_0())
    inpainter_pipeline.vae.set_attn_processor(AttnProcessor2_0())
    
    rotation_path = config['rotation_path'][:config['num_scenes']]
    assert len(rotation_path) == config['num_scenes']
    
    depth_model = MarigoldPipeline.from_pretrained("prs-eth/marigold-depth-v1-0", torch_dtype=torch.bfloat16).to(config["device"])
    depth_model.scheduler = EulerDiscreteScheduler.from_config(depth_model.scheduler.config)
    depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

    normal_estimator = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v0-1", torch_dtype=torch.bfloat16).to(config["device"])
    
    print('###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######') 
    kf_gen = KeyframeGen(config=config, inpainter_pipeline=inpainter_pipeline, mask_generator=mask_generator, depth_model=depth_model,
                            segment_model=segment_model, segment_processor=segment_processor, normal_estimator=normal_estimator,
                            rotation_path=rotation_path, inpainting_resolution=config['inpainting_resolution_gen']).to(config["device"])

    yaml_data = load_example_yaml(config["example_name"], 'examples/examples.yaml')
    content_prompt, style_prompt, adaptive_negative_prompt, background_prompt, control_text, outdoor = yaml_data['content_prompt'], yaml_data['style_prompt'], yaml_data['negative_prompt'], yaml_data.get('background', None), yaml_data.get('control_text', None), yaml_data.get('outdoor', False)
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "

    start_keyframe = Image.open(yaml_data['image_filepath']).convert('RGB').resize((512, 512))
    kf_gen.image_latest = ToTensor()(start_keyframe).unsqueeze(0).to(config['device'])
    
    # Sky generation
    if config['gen_sky_image'] or (not os.path.exists(f'examples/sky_images/{example}/sky_0.png') and not os.path.exists(f'examples/sky_images/{example}/sky_1.png')):
        syncdiffusion_model = SyncDiffusion(config['device'], sd_version='2.0-inpaint')
    else:
        syncdiffusion_model = None
    sky_mask = kf_gen.generate_sky_mask().float()
    kf_gen.generate_sky_pointcloud(syncdiffusion_model, image=kf_gen.image_latest, mask=sky_mask, gen_sky=config['gen_sky_image'], style=style_prompt)

    kf_gen.recompose_image_latest_and_set_current_pc(scene_name=None)
    
    pt_gen = TextpromptGen(kf_gen.run_dir, isinstance(control_text, list))
    
    content_list = content_prompt.split(',')
    scene_name = content_list[0]
    entities = content_list[1:]
    scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}

    kf_gen.increment_kf_idx()
    
    # Sky 3DGS training
    if config['gen_sky'] or not os.path.exists(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply'):
        traindatas = kf_gen.convert_to_3dgs_traindata(xyz_scale=xyz_scale, remove_threshold=None, use_no_loss_mask=False)
        if config['gen_layer']:
            traindata, traindata_sky, traindata_layer = traindatas
        else:
            traindata, traindata_sky = traindatas
        gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
        opt = GSParams()
        opt.max_screen_size = 100
        opt.scene_extent = 1.5
        opt.densify_from_iter = 200
        opt.prune_from_iter = 200
        opt.densify_grad_threshold = 1.0
        opt.iterations = 399
        scene = Scene(traindata_sky, gaussians, opt, is_sky=True)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_sky"
        train_gaussian(gaussians, scene, opt, save_dir, initialize_scaling=False)
        gaussians.save_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply')
    else:
        gaussians = GaussianModel(sh_degree=0)
        gaussians.load_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply')

    gaussians.visibility_filter_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    gaussians.delete_mask_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    gaussians.is_sky_filter = torch.ones(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    
    if config['load_gen'] and os.path.exists(f'examples/sky_images/{example}/finished_3dgs.ply') and os.path.exists(f'examples/sky_images/{example}/visibility_filter_all.pth') and os.path.exists(f'examples/sky_images/{example}/is_sky_filter.pth') and os.path.exists(f'examples/sky_images/{example}/delete_mask_all.pth'):
        print("Loading existing 3DGS...")
        gaussians = GaussianModel(sh_degree=0)
        gaussians.load_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs.ply')
        gaussians.visibility_filter_all = torch.load(f'examples/sky_images/{example}/visibility_filter_all.pth').to('cuda')
        gaussians.is_sky_filter = torch.load(f'examples/sky_images/{example}/is_sky_filter.pth').to('cuda')
        gaussians.delete_mask_all = torch.load(f'examples/sky_images/{example}/delete_mask_all.pth').to('cuda')
    opt = GSParams()

    # First scene 3DGS
    if config['gen_layer']:
        traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata_layer, gaussians, opt)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_layer{0:02d}"
        train_gaussian(gaussians, scene, opt, save_dir)
    else:
        traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)

    gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
    scene = Scene(traindata, gaussians, opt)
    dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{0:02d}"
    train_gaussian(gaussians, scene, opt, save_dir)

    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale)
    gaussians.set_inscreen_points_to_visible(tdgs_cam)
    
    # Determine camera indices to use
    if camera_indices is None:
        # Use cameras from rotation_path (skip first one as it's the origin)
        camera_indices = list(range(1, len(kf_gen.cameras)))
        camera_indices = camera_indices[:config['num_scenes']]
    
    print(f"\n{'='*60}")
    print(f"AUTOMATED GENERATION MODE")
    print(f"{'='*60}")
    print(f"Total scenes to generate: {len(camera_indices)}")
    print(f"Using cameras: {camera_indices}")
    if prompts:
        print(f"Using {len(prompts)} predefined prompts")
    elif config['use_gpt']:
        print("Using GPT for prompt generation")
    else:
        print("Using rotation_path-based generation")
    print(f"{'='*60}\n")
    
    # Initialize logging for scene generation
    generation_log = {
        "example_name": example,
        "style_prompt": style_prompt,
        "initial_scene": {
            "scene_name": scene_name,
            "entities": entities,
            "background": background_prompt
        },
        "scenes": []
    }
    log_file_path = Path(config['runs_dir']) / "generation_log.json"
    
    gaussians_tmp = copy.deepcopy(gaussians)
    
    # Automated generation loop
    for scene_idx, cam_idx in enumerate(tqdm(camera_indices, desc="Generating scenes")):
        if cam_idx >= len(kf_gen.cameras):
            print(f"Warning: Camera index {cam_idx} out of range (max: {len(kf_gen.cameras)-1}). Skipping.")
            continue
        
        print(f"\n--- Generating Scene {scene_idx + 1}/{len(camera_indices)} (Camera {cam_idx}) ---")
        
        # Get camera and convert to view matrix
        camera = kf_gen.cameras[cam_idx]
        view_matrix = camera_to_view_matrix(camera, xyz_scale)
        
        # Handle prompts
        change_scene_name_by_user = False
        if prompts and scene_idx < len(prompts):
            scene_name = prompts[scene_idx]
            change_scene_name_by_user = True
            print(f"Using predefined prompt: {scene_name}")
        elif config['use_gpt']:
            # Generate prompt using GPT
            print("Generating prompt with GPT...")
            scene_dict = pt_gen.wonder_next_scene(
                scene_name=scene_name, 
                entities=scene_dict['entities'], 
                style=style_prompt, 
                background=scene_dict['background'], 
                change_scene_name_by_user=change_scene_name_by_user,
                maintain_continuity=False,  # Allow scene name changes
                contextual_continuity=True    # But ensure they're contextually aware
            )
            change_scene_name_by_user = False
        else:
            # Use rotation_path-based generation (default)
            # Allow scene names to vary but ensure contextual continuity
            scene_dict = pt_gen.wonder_next_scene(
                scene_name=scene_name, 
                entities=scene_dict['entities'], 
                style=style_prompt, 
                background=scene_dict['background'], 
                change_scene_name_by_user=False,
                maintain_continuity=False,  # Allow scene name changes
                contextual_continuity=True    # But ensure they're contextually aware
            )
        
        inpainting_prompt = pt_gen.generate_prompt(
            style=style_prompt, 
            entities=scene_dict['entities'], 
            background=scene_dict['background'], 
            scene_name=scene_dict['scene_name']
        )
        scene_name = scene_dict['scene_name'] if isinstance(scene_dict['scene_name'], str) else scene_dict['scene_name'][0]
        
        # Extract background text for logging
        background_text = scene_dict['background']
        if isinstance(background_text, list):
            background_text = background_text[0] if background_text else ""
        
        # Log this scene's information
        scene_log_entry = {
            "scene_index": scene_idx + 1,
            "camera_index": cam_idx,
            "scene_name": scene_name,
            "entities": scene_dict['entities'] if isinstance(scene_dict['entities'], list) else [scene_dict['entities']],
            "background": background_text,
            "inpainting_prompt": inpainting_prompt,
            "style_prompt": style_prompt,
            "negative_prompt": adaptive_negative_prompt
        }
        generation_log["scenes"].append(scene_log_entry)
        
        # Print detailed scene information
        print(f"\n{'='*60}")
        print(f"SCENE {scene_idx + 1}/{len(camera_indices)} - Camera {cam_idx}")
        print(f"{'='*60}")
        print(f"Scene Name: {scene_name}")
        print(f"Entities: {', '.join(scene_dict['entities'] if isinstance(scene_dict['entities'], list) else [scene_dict['entities']])}")
        print(f"Background: {background_text}")
        print(f"Inpainting Prompt: {inpainting_prompt}")
        print(f"{'='*60}\n")
        
        # Generate scene (same as run.py)
        kf_gen.set_kf_param(
            inpainting_resolution=config['inpainting_resolution_gen'],
            inpainting_prompt=inpainting_prompt, 
            adaptive_negative_prompt=adaptive_negative_prompt
        )
        current_pt3d_cam = kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale)
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
        kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)
        
        # Render and generate
        with torch.no_grad():
            render_pkg = render(tdgs_cam, gaussians, opt, background)
            render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)
        
        side_sky_height = 128
        sky_cond_width = 40

        inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"]<0.6)
        inpaint_mask_0p0_nosky = (render_pkg_nosky["final_opacity"]<0.01)
        inpaint_mask_0p5 = (render_pkg["final_opacity"]<0.6)
        inpaint_mask_0p0 = (render_pkg["final_opacity"]<0.01)
        fg_mask_0p5_nosky = ~inpaint_mask_0p5_nosky.clone()
        foreground_cols = torch.sum(fg_mask_0p5_nosky == 1, dim=1)>150
        foreground_cols_idx = torch.nonzero(foreground_cols, as_tuple=True)[1]

        mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
        if foreground_cols_idx.numel() > 0:
            min_index = foreground_cols_idx.min().item()
            max_index = foreground_cols_idx.max().item()
            mask_using_full_render[:, :, :, min_index:max_index+1] = 1
        mask_using_full_render[:, :, :sky_cond_width, :] = 1
        mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
        mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1
        
        mask_using_nosky_render = 1 - mask_using_full_render
        outpaint_condition_image = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
        fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
        outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
        outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())

        inpaint_output = kf_gen.inpaint(
            outpaint_condition_image, 
            inpaint_mask=outpaint_mask, 
            fill_mask=fill_mask, 
            inpainting_prompt=inpainting_prompt, 
            mask_strategy=np.max, 
            diffusion_steps=50
        )

        sem_seg = kf_gen.update_sky_mask()
        recomposed = soft_stitching(render_pkg["render"], kf_gen.image_latest, kf_gen.sky_mask_latest)

        depth_should_be = render_pkg['median_depth'][0:1].unsqueeze(0) / xyz_scale
        mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (depth_should_be > 0.001)

        ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
        depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (depth_should_be_ground < 0.006 * 0.8)

        joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
        depth_should_be_joint = torch.where(mask_to_align_depth, depth_should_be, depth_should_be_ground)

        with torch.no_grad():
            depth_guide_joint, _ = kf_gen.get_depth(
                kf_gen.image_latest, 
                target_depth=depth_should_be_joint, 
                mask_align=joint_mask, 
                archive_output=True, 
                diffusion_steps=30, 
                guidance_steps=8
            )

        kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())

        kf_gen.image_latest = recomposed
        if config['gen_layer']:
            kf_gen.generate_layer(pred_semantic_map=sem_seg, scene_name=scene_name)

            depth_should_be = kf_gen.depth_latest_init
            mask_to_align_depth = ~(kf_gen.mask_disocclusion.bool()) & (depth_should_be < 0.006 * 0.8)
            mask_to_farther_depth = kf_gen.mask_disocclusion.bool() & (depth_should_be < 0.006 * 0.8)
            with torch.no_grad():
                kf_gen.depth, kf_gen.disparity = kf_gen.get_depth(
                    kf_gen.image_latest, 
                    archive_output=True, 
                    target_depth=depth_should_be, 
                    mask_align=mask_to_align_depth, 
                    mask_farther=mask_to_farther_depth,
                    diffusion_steps=30, 
                    guidance_steps=8
                )
            kf_gen.refine_disp_with_segments(
                no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                existing_mask=~(kf_gen.mask_disocclusion).bool().squeeze().cpu().numpy(),
                existing_disp=kf_gen.disparity_latest_init.squeeze().cpu().numpy()
            )
            wrong_depth_mask = kf_gen.depth_latest<kf_gen.depth_latest_init
            kf_gen.depth_latest[wrong_depth_mask] = kf_gen.depth_latest_init[wrong_depth_mask] + 0.0001
            kf_gen.depth_latest = kf_gen.mask_disocclusion * kf_gen.depth_latest + (1-kf_gen.mask_disocclusion) * kf_gen.depth_latest_init
            kf_gen.update_sky_mask()
            valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask)
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest_init, depth=kf_gen.depth_latest_init, valid_mask=kf_gen.mask_disocclusion*outpaint_mask, gen_layer=True)
        else:
            valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask)
        kf_gen.archive_latest()

        # Train 3DGS
        if config['gen_layer']:
            traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer, gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_layer{scene_idx+1:02d}"
            train_gaussian(gaussians, scene, opt, save_dir)
        else:
            traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)

        if traindata['pcd_points'].shape[-1] == 0:
            gaussians.set_inscreen_points_to_visible(tdgs_cam)
            kf_gen.increment_kf_idx()
            continue
        
        mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
        x = torch.sum(fg_mask_0p5_nosky == 1, dim=2)>0
        x_idx = torch.nonzero(x, as_tuple=True)[1]
        if foreground_cols_idx.numel() > 0:
            min_index = foreground_cols_idx.min().item()
            max_index = foreground_cols_idx.max().item()
            mask_using_full_render[:, :, :x_idx.max().item(), min_index:max_index+1] = 1
        
        mask_using_nosky_render = 1 - mask_using_full_render
        image_tmp = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
        
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata, gaussians, opt)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{scene_idx+1:02d}"
        train_gaussian(gaussians, scene, opt, save_dir)
        
        gaussians.set_inscreen_points_to_visible(tdgs_cam)
        kf_gen.increment_kf_idx()
        gaussians_tmp = copy.deepcopy(gaussians)
        empty_cache()
        
        # Save log incrementally after each scene (in case of crash)
        generation_log["total_scenes"] = len(camera_indices)
        generation_log["completed_scenes"] = scene_idx + 1
        with open(log_file_path, 'w') as f:
            json.dump(generation_log, f, indent=2)
        
        print(f"✓ Completed scene {scene_idx + 1}/{len(camera_indices)}")
    
    # Final save - save the complete Gaussian splat model in the run directory
    print("\n" + "="*60)
    print("Saving final Gaussian splat model...")
    print("="*60)
    
    # Save PLY file
    final_ply_path = kf_gen.run_dir / "finished_3dgs.ply"
    gaussians.save_ply_all_with_filter(str(final_ply_path))
    print(f"✓ PLY file saved: {final_ply_path}")
    
    # Save filter masks
    visibility_path = kf_gen.run_dir / "visibility_filter_all.pth"
    sky_filter_path = kf_gen.run_dir / "is_sky_filter.pth"
    delete_mask_path = kf_gen.run_dir / "delete_mask_all.pth"
    
    torch.save(gaussians.visibility_filter_all, str(visibility_path))
    torch.save(gaussians.is_sky_filter, str(sky_filter_path))
    torch.save(gaussians.delete_mask_all, str(delete_mask_path))
    print(f"✓ Filter masks saved: {visibility_path.name}, {sky_filter_path.name}, {delete_mask_path.name}")
    
    # Save splat file (binary format for real-time viewers)
    splat_path = kf_gen.run_dir / f"{example}_finished_3dgs.splat"
    gaussians.yield_splat_data(str(splat_path))
    print(f"✓ Splat file saved: {splat_path}")
    
    # Finalize and save generation log
    generation_log["total_scenes"] = len(camera_indices)
    generation_log["completed_scenes"] = len(camera_indices)
    generation_log["completion_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, 'w') as f:
        json.dump(generation_log, f, indent=2)
    
    print("\n" + "="*60)
    print("AUTOMATED GENERATION COMPLETE")
    print("="*60)
    print(f"Generated {len(camera_indices)} scenes")
    print(f"Results saved to: {kf_gen.run_dir}")
    print(f"  - Final PLY: {final_ply_path.name}")
    print(f"  - Splat file: {splat_path.name}")
    print(f"  - Scene images: images/frames/")
    print(f"  - Generation log: {log_file_path.name}")
    print("="*60)
    
    # Print summary of scene names
    print("\n" + "="*60)
    print("SCENE GENERATION SUMMARY")
    print("="*60)
    print(f"Style: {style_prompt}")
    print(f"\nScene Progression:")
    for i, scene in enumerate(generation_log["scenes"], 1):
        print(f"  {i}. {scene['scene_name']} (Camera {scene['camera_index']})")
        print(f"     Entities: {', '.join(scene['entities'])}")
    print("="*60)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base-config", default="./config/base-config.yaml", help="Base config path")
    parser.add_argument("--example-config", required=True, help="Example config path")
    parser.add_argument("--camera-indices", type=str, help="Comma-separated camera indices (e.g., '1,2,3,4') or 'all' for all")
    parser.add_argument("--prompts", type=str, help="Comma-separated scene prompts (e.g., 'campus,library,quad')")
    parser.add_argument("--use-gpt", action="store_true", help="Use GPT for prompt generation")
    args = parser.parse_args()
    
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)
    
    # Parse camera indices
    camera_indices = None
    if args.camera_indices:
        if args.camera_indices.lower() == 'all':
            camera_indices = None  # Will use all cameras
        else:
            camera_indices = [int(x.strip()) for x in args.camera_indices.split(',')]
    
    # Parse prompts
    prompts = None
    if args.prompts:
        prompts = [x.strip() for x in args.prompts.split(',')]
    
    # Override use_gpt if prompts provided
    if prompts:
        config['use_gpt'] = False
    
    if args.use_gpt:
        config['use_gpt'] = True
    
    run_automated(config, camera_indices=camera_indices, prompts=prompts)

