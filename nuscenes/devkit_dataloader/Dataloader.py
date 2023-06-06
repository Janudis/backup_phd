import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from nuscenes.devkit_dataloader.nuscenes import NuScenes
from nuscenes.devkit_dataloader.nuscenes import NuScenesExplorer

nusc = NuScenes(version='v1.0-mini', dataroot='/home/dimitris/PhD/PhD/nuscenes/data/sets/nuscenes', verbose=True)
nusc = NuScenes(version='v1.0-mini', dataroot='D:/Python_Projects/self_driving_car/nuscenes-devkit/python-sdk/nuscenes/data/sets/nuscenes', verbose=True)
#nusc = NuScenes(version='v1.0-mini', dataroot='home/dimitris/PhD/PhD/nuscenes/data/sets/nuscenes', verbose=True)

nusc2 = NuScenesExplorer(nusc)
from scipy.spatial.transform import Rotation as R

class NuscenesDataset(Dataset):
    def __init__(self, nusc):
        self.nusc = nusc
        self.transform = transforms.Compose([
            #transforms.Resize((375, 1242)),
            transforms.Resize((512, 512)),
            #transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.permute(1, 2, 0)), # Permute the dimensions from CHW to HWC
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.nusc.sample)

    def __getitem__(self, idx):
        my_sample = self.nusc.sample[idx]
        cam_front_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        x = cam_front_data['filename']
        image_path = '/home/dimitris/PhD/PhD/nuscenes/' + x
        image = Image.open(image_path)
        #image = self.transform(image)
        #image = image.transpose(0, 1).transpose(1, 2)
        return image
    
    def get_item_vrn(self,idx):
        my_sample = self.nusc.sample[idx]
        cam_front_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        x = cam_front_data['filename']
        image_path = '/home/dimitris/PhD/PhD/nuscenes/' + x
        image = Image.open(image_path)
        image = self.transform(image)
        return image

    def get_points(self,idx):
        points = []
        my_sample = self.nusc.sample[idx]
        point, coloring, im, lidar_points = nusc2.map_pointcloud_to_image(pointsensor_token=my_sample['data']['LIDAR_TOP'],
                                                            camera_token=my_sample['data']['CAM_FRONT'])
        points.append(point)  # mia lista me pinakes
        return points, lidar_points

    def get_calib(self,idx):
        my_sample = self.nusc.sample[idx]
        cam_front_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        for i in range(0,119):  
            a = nusc.calibrated_sensor[i]
            if a['token'] == cam_front_data['calibrated_sensor_token']:
                K = np.array(a['camera_intrinsic'])
                t = np.array(a['translation'])
                # Convert quaternion to rotation matrix
                quaternion = np.array(a['rotation'])
                r = R.from_quat(quaternion)
                rotation = r.as_matrix()
                # Concatenate rotation matrix and translation vector to form a 3x4 extrinsic matrix
                RT = np.hstack((rotation, t.reshape(-1, 1)))
                # Compute projection matrix P by multiplying K and [R|t]
                P = np.dot(K, RT)
                return P
            
    