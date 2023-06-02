import sys
sys.path.append('D:/Python_Projects/PhD_project')

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nuscenes.devkit_dataloader.Dataloader import *
from visualDet3D.visualDet3D.data.kitti.utils import write_result_to_file
from visualDet3D.visualDet3D.utils.utils import LossLogger, cfg_from_file
from visualDet3D.visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
from visualDet3D.visualDet3D.networks.utils import BBox3dProjector, BackProjection
from visualDet3D.visualDet3D.data.pipeline import build_augmentator

cfg = cfg_from_file('D:/Python_Projects/PhD_project/visualDet3D/config/config.py')
is_test_train = True

checkpoint_name = 'D:/Python_Projects/PhD_project/visualDet3D/workdirs/Mono3D/checkpoint/GroundAware_pretrained.pth'

detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
detector = detector.cuda()

weight_path = os.path.join(cfg.path.checkpoint_path, checkpoint_name)
state_dict = torch.load(weight_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
new_dict = state_dict.copy()
for key in state_dict:
    if 'focalLoss' in key:
        new_dict.pop(key)
detector.load_state_dict(new_dict, strict=False)
detector.eval().cuda()

# testing pipeline
test_func = PIPELINE_DICT[cfg.trainer.test_func]

projector = BBox3dProjector().cuda()
backprojector = BackProjection().cuda()

input_image = Image.open('D:/Python_Projects/self_driving_car/nuscenes-devkit/python-sdk/nuscenes/visualDet3D/data/testing/image_2/000001.png')
image = np.array(input_image)
calib = dict()
path = 'D:/Python_Projects/self_driving_car/nuscenes-devkit/python-sdk/nuscenes/visualDet3D/data/testing/calib/000001.txt'
with open(path) as f:
    str_list = f.readlines()
str_list = [itm.rstrip() for itm in str_list if itm != '\n']
for itm in str_list:
    calib[itm.split(':')[0]] = itm.split(':')[1]
for k, v in calib.items():
    calib[k] = [float(itm) for itm in v.split()]
calibr = np.array(calib['P2']).reshape(3,4)


def collate_fn(batch):
    rgb_images = np.array([item["image"] for item in batch]) #[batch, H, W, 3]
    rgb_images = rgb_images.transpose([0, 3, 1, 2])
    calib = np.array([item["calib"] for item in batch])
    return torch.from_numpy(rgb_images).float(), torch.from_numpy(calib).float()


is_test_train=True
transform = build_augmentator(cfg.data.test_augmentation)
transformed_image, transformed_P2 = transform(image.copy(), p2=calibr.copy())
data = {'calib': transformed_P2,
                    'image': transformed_image,
                    'original_shape':image.shape,
                    'original_P':calibr.copy()}

original_height = data['original_shape'][0]
collated_data = collate_fn([data])

transformed_image = collated_data[0]
transformed_P2 = collated_data[1]

height = collated_data[0].shape[2]
scale_2d = (original_height - cfg.data.augmentation.crop_top) / height

test_func = PIPELINE_DICT[cfg.trainer.test_func]
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
#print(objects)
# [{'whl': tensor([1.6292, 1.5660, 4.5354], device='cuda:0'), 'theta': tensor(-0.2489, device='cuda:0'), 
# 'score': tensor(0.9961, device='cuda:0'), 'type_name': 'Car', 'xyz': tensor([-9.1681,  0.9484, 18.3366], device='cuda:0')}, 
# {'whl': tensor([1.7288, 1.6186, 4.1517], device='cuda:0'), 'theta': tensor(0.7109, device='cuda:0'), 'score': tensor(0.9540, device='cuda:0'), 
# 'type_name': 'Car', 'xyz': tensor([-8.2275,  0.7891, 
# 11.9172], device='cuda:0')}]

# from scipy.spatial.transform import Rotation 
# whl = objects[0]['whl']
# xyz = objects[0]['xyz']
# theta = objects[0]['theta']

# # 1. Create the rotation matrix Ry
# Ry = torch.tensor([
#     [torch.cos(theta), 0, torch.sin(theta)],
#     [0, 1, 0],
#     [-torch.sin(theta), 0, torch.cos(theta)]
# ]).cuda()

# # 2. Create the translation vector
# T = xyz

# # 3. Create the scale matrix
# S = torch.diag(whl)

# # 4. Now construct the 4x4 transformation matrix Qn
# Qn = torch.eye(4).cuda()
# Qn[:3, :3] = torch.mm(Ry, S)
# Qn[:3, 3] = T
# Qn_inv = torch.inverse(Qn)
# print(Qn_inv)
# tensor([[ 5.9488e-01,  0.0000e+00,  1.5123e-01,  2.6809e+00],
#         [ 4.2521e-09,  6.3859e-01, -4.8019e-10, -6.0567e-01],
#         [-5.4325e-02, -0.0000e+00,  2.1369e-01, -4.4164e+00],
#         [ 4.1738e-09,  0.0000e+00, -4.9048e-10,  1.0000e+00]], device='cuda:0')

# # Create the rotation matrix Ry around the y-axis
# tx, ty, tz = objects[0]['xyz']
# h, w, l = objects[0]['whl']
# theta = objects[0]['theta']

# cos, sin = torch.cos(theta), torch.sin(theta)
# Ry = torch.tensor([
#     [cos, 0, sin],
#     [0, 1, 0],
#     [-sin, 0, cos]
# ], device='cuda:0')

# # Create the translation vector
# t = torch.tensor([tx, ty, tz], device='cuda:0')

# # Create the scale matrix based on the object's dimensions
# S = torch.diag(torch.tensor([l, h, w], device='cuda:0'))

# # Now construct the 4x4 transformation matrix Qn
# Qn = torch.eye(4, device='cuda:0')
# Qn[:3, :3] = torch.mm(Ry, S)
# Qn[:3, 3] = t
# Qn_inv = torch.inverse(Qn)
# print(Qn_inv)
# tensor([[ 2.1369e-01,  0.0000e+00,  5.4325e-02,  9.6301e-01],
#         [ 2.9464e-09,  6.1380e-01,  1.3783e-09, -5.8216e-01],
#         [-1.5734e-01, -0.0000e+00,  6.1890e-01, -1.2791e+01],
#         [ 5.7101e-10,  0.0000e+00, -3.4257e-09,  1.0000e+00]], device='cuda:0')

# dn = objects[0]['whl'].cpu().numpy()
# xyz = objects[0]['xyz'].cpu().numpy()
# theta = objects[0]['theta'].cpu().numpy()

# # Step 1: Convert orientation angle to rotation matrix
# R = Rotation.from_euler('z', theta).as_matrix()

# # Step 2: Compute translation vector
# t = xyz.reshape(3, 1)

# # Step 3: Combine R and t into a homogeneous transformation matrix T
# T = np.hstack((R, t))
# T = np.vstack((T, [0, 0, 0, 1]))

# # Step 4: Invert T
# T_inv = np.linalg.inv(T)

# # Step 5: Compute scaled object dimensions
# d_prime = T_inv @ np.hstack((dn, 1))

# # Step 6: Compute object pose
# Qnt = np.eye(4)  # Example object pose
# diag_scale = np.diag([1/d_prime[0], 1/d_prime[1], 1/d_prime[2], 1])
# Qn = Qnt @ diag_scale @ T_inv
# Qn_inv = np.linalg.inv(Qn)
# # Output result
# print(Qn_inv)
# [[  9.99439782   0.80291432   0.          -9.16812992]
#  [ -2.5408022    3.15831162   0.           0.94844323]
#  [ -0.          -0.         -13.80119133  18.33657837]
#  [  0.           0.           0.           1.        ]]

# # Start with an identity matrix
# Qn = np.eye(4)

# # Set the translation (object position)
# Qn[0:3, 3] = xyz

# # Set the rotation
# # Assuming that theta is the yaw angle (rotation around the up axis)
# cos_theta = np.cos(theta)
# sin_theta = np.sin(theta)

# # Assuming that the up axis is Z
# Qn[0, 0] = cos_theta
# Qn[0, 1] = -sin_theta
# Qn[1, 0] = sin_theta
# Qn[1, 1] = cos_theta

# # Set the scale (object size)
# # Assuming that the object's local axes align with the bounding box sides
# # This is not always the case in reality, but without more information, it's a reasonable assumption
# Qn[0, 0] *= dn[0]
# Qn[1, 1] *= dn[1]
# Qn[2, 2] *= dn[2]
# Qn_inv = np.linalg.inv(Qn)
# # Output result
# print(Qn_inv)
# [[ 0.6176817  -0.10027676  0.          5.75809285]
#  [ 0.10027676  0.64262226  0.          0.30985967]
#  [ 0.          0.          0.22048835 -4.04300189]
#  [ 0.          0.          0.          1.        ]]

nusc_dataset = NuscenesDataset(nusc)

dn = objects[0]['whl'].cpu().numpy()
xyz = objects[0]['xyz'].cpu().numpy()
theta = objects[0]['theta'].cpu().numpy()

points, lidar_points = nusc_dataset.get_points(0)
lidar_point = np.array([ lidar_points[0][0], lidar_points[1][0], lidar_points[2][0] ])
# print("bb", objects[0]['xyz'].cpu().numpy())
# print("lidar", lidar_point)
# print(points[0][0])
# print(points[0][1])
# print(points[0][2])

z_local = xyz - lidar_point
#print(z_local)
z_dim = dn
#print(z_dim)
if(objects[0]['type_name']=='Car'):
    z_class = 1
else:
    z_class = 0
#print(z_class)
z_dim = z_dim.reshape(1, 3)
z_local = z_local.reshape(1, 3)
# Concatenate the arrays horizontally
z = np.concatenate((z_local, z_dim, np.array([[z_class]])), axis=1)
print(z)

# lidar_point_homog = np.append(lidar_point, 1)
# # Multiply x_homog by Qn_inv
# z_local_homog = np.dot(Qn_inv, lidar_point_homog)
# # Convert x_local_homog back to 3D coordinates
# z_local = z_local_homog[:3] / z_local_homog[3]







