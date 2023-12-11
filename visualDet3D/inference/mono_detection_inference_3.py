import sys
sys.path.append('/home/dimitris/PhD/PhD/visualDet3D')
sys.path.append('/home/dimitris/PhD/PhD/nuscenes')
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from devkit_dataloader.Dataloader import *
from visualDet3D.data.kitti.utils import write_result_to_file
from visualDet3D.utils.utils import LossLogger, cfg_from_file
from visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
from visualDet3D.networks.utils import BBox3dProjector, BackProjection
from visualDet3D.data.pipeline import build_augmentator

cfg = cfg_from_file('/home/dimitris/PhD/PhD/visualDet3D/config/config.py')
checkpoint_name = '/home/dimitris/PhD/PhD/visualDet3D/workdirs/Mono3D/checkpoint/GroundAware_pretrained.pth'
weight_path = os.path.join(cfg.path.checkpoint_path, checkpoint_name)

detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
detector = detector.cuda()

state_dict = torch.load(weight_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
new_dict = state_dict.copy()
for key in state_dict:
    if 'focalLoss' in key:
        new_dict.pop(key)
detector.load_state_dict(new_dict, strict=False)
detector.eval().cuda() #turns off behaviors specific to training such as dropout and batch normalization layers.

# testing pipeline
test_func = PIPELINE_DICT[cfg.trainer.test_func]

projector = BBox3dProjector().cuda()
backprojector = BackProjection().cuda()

nusc_dataset = NuscenesDataset(nusc)

def process_index(index, nusc_dataset, detector, cfg, test_func, backprojector, projector):
    # Fetch data
    input_image, _ = nusc_dataset.__getitem__(index)
    image = np.array(input_image)
    calibr = nusc_dataset.get_calib(index)

    def collate_fn(batch):
        rgb_images = np.array([item["image"] for item in batch]) #[batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])
        calib = np.array([item["calib"] for item in batch])
        return torch.from_numpy(rgb_images).float(), torch.from_numpy(calib).float()

    transform = build_augmentator(cfg.data.test_augmentation)
    transformed_image, transformed_P2 = transform(image.copy(), p2=calibr.copy())
    data = {'calib': transformed_P2,
                        'image': transformed_image,
                        'original_shape':image.shape,
                        'original_P':calibr.copy()}

    original_height = data['original_shape'][0]
    collated_data = collate_fn([data])

    transformed_image = collated_data[0]
    print(f"input image to detector shape : {transformed_image.shape}")
    transformed_P2 = collated_data[1]

    with torch.no_grad():
        scores, bbox, obj_names = test_func(collated_data, detector, None, cfg=cfg)
        transformed_P2 = transformed_P2[0]
        bbox_2d = bbox[:, 0:4]
        bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha]
        bbox_3d_state_3d = backprojector(bbox_3d_state, transformed_P2.cuda()) #[x, y, z, w,h ,l, alpha]
        abs_bbox, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, transformed_P2.cuda()) 

        objects = []
        N = len(bbox)
        for i in range(N):
            obj = {}
            obj['whl'] = bbox_3d_state_3d[i, 3:6]
            obj['theta'] = thetas[i]
            obj['score'] = scores[i]
            obj['type_name'] = obj_names[i]
            obj['xyz'] = bbox_3d_state_3d[i, 0:3]
            objects.append(obj)

    def is_inside_box(points, obj):
        cx, cy, z, w, h, l, theta = obj['xyz'][0], obj['xyz'][1], obj['xyz'][2], obj['whl'][0], obj['whl'][1], obj['whl'][2], obj['theta']

        # Create the 3D bounding box vertices in object local coordinate
        z_corners = torch.tensor([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], device=points.device) + z
        y_corners = torch.tensor([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], device=points.device) + cy
        x_corners = torch.tensor([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], device=points.device) + cx

        # Get the coordinates of the points.
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Check if each coordinate of each point lies within the bounding box limits.
        x_inside = torch.logical_and(x_corners.min() <= x, x <= x_corners.max())
        y_inside = torch.logical_and(y_corners.min() <= y, y <= y_corners.max())
        z_inside = torch.logical_and(z_corners.min() <= z, z <= z_corners.max())

        # Combine the results for x, y, and z.
        inside = torch.logical_and(x_inside, torch.logical_and(y_inside, z_inside))

        return inside

    z_local = []
    z_dim = []
    z_class = []
    points, lidar_pc, masked_pc = nusc_dataset.get_points(index)
    lidar_points = torch.tensor(masked_pc)
    lidar_points = lidar_points.to('cuda:0')
    inside_any_box = torch.zeros(lidar_points.shape[1], dtype=torch.bool, device=lidar_points.device)

    for obj in objects: 
        # obj_detector_length = math.sqrt(obj['xyz'][0]**2 + obj['xyz'][1]**2 + obj['xyz'][2]**2)
        # object_info_list = nusc_dataset.get_object_pose(0)
        # # print(obj_detector_length)
        # for idx, info in enumerate(object_info_list):
        #     if (abs(info[0] - obj_detector_length) <= 2 and info[1] == "vehicle.car"):
        #         print("Condition is true for index:", idx)
        #         print(obj_detector_length)
        #         print(info)
        inside_this_box = is_inside_box(lidar_points.T, obj)
        inside_any_box |= inside_this_box  
        if inside_this_box.any(): 
            l = lidar_points[:, inside_this_box]
            offset = l - obj['xyz']
            z_local.append(offset)
            whl = obj['whl'].unsqueeze(0).repeat(offset.shape[0], 1)  
            z_dim.append(whl)
            z_class.append(torch.ones(offset.shape[0], device=lidar_points.device))

    outside_all_boxes = ~inside_any_box 
    if outside_all_boxes.any():  
        l = torch.zeros((torch.sum(outside_all_boxes), 3), device=lidar_points.device)
        z_local.append(l)  
        z_dim.append(torch.zeros((l.shape[0], 3), device=lidar_points.device))  
        z_class.append(torch.zeros(l.shape[0], device=lidar_points.device))  

    z_local = torch.cat(z_local, dim=0)
    z_dim = torch.cat(z_dim, dim=0)
    z_class = torch.cat(z_class, dim=0)

    z_class = z_class.unsqueeze(1)  
    z_box = torch.cat((z_local, z_dim, z_class), dim=1)

    return z_box

sys.path.append('/home/dimitris/PhD/PhD/vrn_encoder')
from test_vrn import process_data

index = 0
z_box = process_index(index, nusc_dataset, detector, cfg, test_func, backprojector, projector)
z_image = process_data(index)
z = torch.cat((z_box, z_image), 1)
# print(z_box)
# print(z_box.shape)
# print(z_image.shape)
# print(z.shape)

