import sys
sys.path.append('D:/Python_Projects/PhD_project')
import os
import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import cv2

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
from nuscenes.devkit_dataloader.Dataloader import *

from visualDet3D.visualDet3D.utils.utils import cfg_from_file
from visualDet3D.visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
from visualDet3D.visualDet3D.networks.utils import BBox3dProjector, BackProjection
from visualDet3D.visualDet3D.utils.utils import convertAlpha2Rot, convertRot2Alpha, draw_3D_box, compound_annotation
import visualDet3D.visualDet3D.data.kitti.dataset
from visualDet3D.visualDet3D.data.pipeline import build_augmentator
from visualDet3D.visualDet3D.networks.pipelines.testers import test_mono_detection 
from visualDet3D.visualDet3D.data.kitti.utils import write_result_to_file

# cfg = cfg_from_file('D:/Python_Projects/PhD_project/visualDet3D/config/config.py')
# weight_path = 'D:/Python_Projects/PhD_project/visualDet3D/workdirs/Mono3D/checkpoint/GroundAware_pretrained.pth'

# detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
# detector = detector.cuda()

# state_dict = torch.load(weight_path)
# detector.load_state_dict(state_dict, strict=False)

cfg = cfg_from_file('D:/Python_Projects/PhD_project/visualDet3D/config/config.py')
checkpoint_name = 'D:/Python_Projects/PhD_project/visualDet3D/workdirs/Mono3D/checkpoint/GroundAware_pretrained.pth'
weight_path = os.path.join(cfg.path.checkpoint_path, checkpoint_name)

detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
detector = detector.cuda()

state_dict = torch.load(weight_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
new_dict = state_dict.copy()
for key in state_dict:
    if 'focalLoss' in key:
        new_dict.pop(key)
detector.load_state_dict(new_dict, strict=False)
detector.eval().cuda()

# nusc_dataset = NuscenesDataset(nusc)
# input_image = nusc_dataset.__getitem__(0)
# image = np.array(input_image)
# calibr = nusc_dataset.get_calib(0)
# print(image.shape)
#calibr = np.concatenate((calibration, np.zeros((3,1))), axis=1)
input_image = Image.open('D:/Python_Projects/self_driving_car/nuscenes-devkit/python-sdk/nuscenes/visualDet3D/data/testing/image_2/000001.png')
#input_image.show()
input_image = np.array(input_image)
#print(input_image)
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
# print(P2)

def collate_fn(batch):
    rgb_images = np.array([item["image"] for item in batch]) #[batch, H, W, 3]
    rgb_images = rgb_images.transpose([0, 3, 1, 2])
    calib = np.array([item["calib"] for item in batch])
    return torch.from_numpy(rgb_images).float(), torch.from_numpy(calib).float()

projector = BBox3dProjector().cuda()
backprojector = BackProjection().cuda()

transform = build_augmentator(cfg.data.test_augmentation)
test_func = PIPELINE_DICT[cfg.trainer.test_func]

transformed_image, transformed_P2 = transform(input_image.copy(), p2=calibr.copy())
data = {'calib': transformed_P2,
                       'image': transformed_image,
                       'original_shape':input_image.shape,
                       'original_P':calibr.copy()}
result_path = 'D:/Python_Projects/self_driving_car/nuscenes-devkit/python-sdk/nuscenes/visualDet3D/workdirs/Mono3D/checkpoint'   
original_height = data['original_shape'][0]
collated_data = collate_fn([data])
#height = collated_data[0].shape[2]
#print(type(transformed_P2))
transformed_image = collated_data[0]
transformed_P2 = collated_data[1]
# print(transformed_image.shape)
print(collated_data)
print("Shape of image tensor: ", transformed_image.shape)
print("Data type: ", transformed_image.dtype)
print("Max value: ", torch.max(transformed_image))
print("Min value: ", torch.min(transformed_image))
print("Mean value: ", torch.mean(transformed_image))
with torch.no_grad():       
    scores, bbox, obj_names = test_func(collated_data, detector, None, cfg=cfg)
    transformed_P2 = transformed_P2[0] 
    bbox_2d = bbox[:, 0:4]
    bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha]
    bbox_3d_state_3d = backprojector(bbox_3d_state, transformed_P2.cuda()) #[x, y, z, w,h ,l, alpha]
    abs_bbox, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, transformed_P2.cuda()) 
    # print("scores", scores)
    # print("bbox", bbox)
    # print("obj_names", obj_names)
    # bbox_2d = bbox[:, 0:4]
    # bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha]
    # bbox_3d_state[:, 2] *= 1.0
    # bbox_3d_state_3d = backprojector(bbox_3d_state, transformed_P2) #[x, y, z, w,h ,l, alpha]
    # abs_bbox, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, bbox_3d_state_3d.new(transformed_P2))
    # original_P = data['original_P']
    # scale_x = original_P[0, 0] / transformed_P2[0, 0]
    # scale_y = original_P[1, 1] / transformed_P2[1, 1]          
    # shift_left = original_P[0, 2] / scale_x - transformed_P2[0, 2]
    # shift_top  = original_P[1, 2] / scale_y - transformed_P2[1, 2]
    # bbox_2d[:, 0:4:2] += shift_left
    # bbox_2d[:, 1:4:2] += shift_top
    # bbox_2d[:, 0:4:2] *= scale_x
    # bbox_2d[:, 1:4:2] *= scale_y
    # bbox_2d = bbox_2d.cpu().numpy()
    # bbox_3d_state_3d = bbox_3d_state_3d.cpu().numpy()
    # thetas = thetas.cpu().numpy()
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
print(objects) 