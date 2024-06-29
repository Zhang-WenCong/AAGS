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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torchvision.transforms as T
import torchvision

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    cam_img = cam_info.image

    resized_image_rgb = PILtoTorch(cam_img, resolution)
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    resolution_8 = (orig_w//(args.resolution*8), orig_h//(args.resolution*8))
    resized_image_rgb_8 = PILtoTorch(cam_img, resolution_8)
    gt_image_8 = resized_image_rgb_8[:3, ...]
    gt_image_8 = normalize(gt_image_8)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, image_8=gt_image_8, image_pil = cam_img.resize(resolution), data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def JSON_to_camera(cam_json, root_dir):
    W2C = np.zeros((4, 4))
    W2C[:3, 3] = cam_json['position']
    W2C[:3, :3] = cam_json['rotation']
    W2C[3, 3] = 1.0
    Rt = np.linalg.inv(W2C)
    R = Rt[:3, :3].transpose()
    T = Rt[:3, 3]
    fov_x = focal2fov(cam_json['fx'], cam_json['width'])
    fov_y = focal2fov(cam_json['fy'], cam_json['height'])
    
    if root_dir is not None:
        from PIL import Image
        image = Image.open(root_dir +  cam_json['img_name'])
        resolution = (int(cam_json['width']), int(cam_json['height']))
        gt_image = PILtoTorch(image, resolution)

        normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        resolution_8 = (int(cam_json['width'] // 16), int(cam_json['height']) // 16)
        resized_image_rgb_8 = PILtoTorch(image, resolution_8)
        gt_image_8 = resized_image_rgb_8[:3, ...]
        gt_image_8 = normalize(gt_image_8)

    return Camera(colmap_id=cam_json['id'], R=R, T=T, FoVx=fov_x, FoVy=fov_y, image = gt_image, image_8=gt_image_8,
                    gt_alpha_mask=None, image_name=cam_json['img_name'], uid=cam_json['id'])