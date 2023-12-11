import os
import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import cv2
import matplotlib.pyplot as plt
from visualDet3D.visualDet3D.networks.utils import BBox3dProjector, BackProjection
from visualDet3D.visualDet3D.networks.utils import BBox3dProjector, BackProjection
from visualDet3D.visualDet3D.data.pipeline import build_augmentator
from visualDet3D.visualDet3D.utils.utils import cfg_from_file
from visualDet3D.visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT

cfg_file = 'D:/Python_Projects/self_driving_car/nuscenes-devkit/python-sdk/nuscenes/visualDet3D/config/config.py'
cfg = cfg_from_file(cfg_file)

def draw_3D_box(img, corners, color = (255, 255, 0)):
    """
        draw 3D box in image with OpenCV,
        the order of the corners should be the same with BBox3dProjector
    """
    points = np.array(corners[0:2], dtype=np.int32) #[2, 8]
    points = [tuple(points[:,i]) for i in range(8)]
    for i in range(1, 5):
        cv2.line(img, points[i], points[(i%4+1)], color, 2)
        cv2.line(img, points[(i + 4)%8], points[((i)%4 + 5)%8], color, 2)
    cv2.line(img, points[2], points[7], color)
    cv2.line(img, points[3], points[6], color)
    cv2.line(img, points[4], points[5],color)
    cv2.line(img, points[0], points[1], color)
    return img

def denorm(image):
    new_image = np.array((image * cfg.data.augmentation.rgb_std +  cfg.data.augmentation.rgb_mean) * 255, dtype=np.uint8)
    return new_image

projector = BBox3dProjector().cuda()
backprojector = BackProjection().cuda()

input_image = Image.open('D:/Python_Projects/self_driving_car/nuscenes-devkit/python-sdk/nuscenes/visualDet3D/data/testing/image_2/000001.png')
input_image = np.array(input_image)
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

transform = build_augmentator(cfg.data.test_augmentation)
transformed_image, transformed_P2 = transform(input_image.copy(), p2=calibr.copy())
data = {'calib': transformed_P2,
                       'image': transformed_image,
                       'original_shape':input_image.shape,
                       'original_P':calibr.copy()}
collated_data = collate_fn([data])

image, P2 = collated_data[0], collated_data[1]
img_batch = image.cuda().float().contiguous() #[1, 3, H, W] Augmented
P2 = torch.tensor(P2).cuda().float() # [1, 3, 4] Augmented

weight_path = os.path.abspath('D:/Python_Projects/self_driving_car/nuscenes-devkit/python-sdk/nuscenes/visualDet3D/workdirs/Mono3D/checkpoint/GroundAware_pretrained.pth')
detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
detector = detector.cuda()
state_dict = torch.load(weight_path)
detector.load_state_dict(state_dict, strict=False)

scores, bbox, obj_index = detector([img_batch, P2]) # test_forward

img = image.squeeze().permute(1, 2, 0).numpy()
rgb_image = denorm(img) # np.uint8 

bbox_2d = bbox[:, 0:4]
bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha]
bbox_3d_state_3d = backprojector(bbox_3d_state, P2.cuda()) #[x, y, z, w,h ,l, alpha]
abs_bbox, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, P2[0]) # check the docstring

for box in bbox_3d_corner_homo:
    box = box.cpu().numpy().T
    rgb_image = draw_3D_box(rgb_image, box)

plt.imshow(rgb_image) # the bboxes are drawed on the augmented image