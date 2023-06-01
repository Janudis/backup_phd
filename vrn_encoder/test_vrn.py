import torch
import torchvision.transforms as transforms
from PIL import Image
from vrn_encoder.vrn_model import vrn_unguided
from nuscenes.devkit_dataloader.nuscenes import NuScenes
from nuscenes.devkit_dataloader.nuscenes import NuScenesExplorer
from nuscenes.devkit_dataloader.Dataloader import *

nusc_dataset = NuscenesDataset(nusc)
input_image = nusc_dataset.get_item_vrn(0)
input_image = input_image.unsqueeze(0)
points, lidar_points = nusc_dataset.get_points(0)
# print(points)
# print(points[0][0])
# Load model
model = vrn_unguided

# Set model to evaluation mode
model.eval()

# Forward pass
with torch.no_grad():
    output = model(input_image)

z_list = []
for i in range(len(points[0][0])):
    out = output[0]
    z = out[:, :, int(points[0][0][i]), int(points[0][1][i])]
    z_list.append(z)
    z_final = torch.cat(z_list, dim=0)
print(z_final.shape)
print(z_final[3030].shape)

# # Define device (CPU or GPU)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Create DataLoader object
# batch_size = 32
# nusc_dataset = NuscenesDataset(nusc)
# nusc_dataloader = torch.utils.data.DataLoader(nusc_dataset, batch_size=batch_size, shuffle=True)
#
# model = vrn_unguided
# # Test model on a batch of images
# for batch in nusc_dataloader:
#     # Move batch to device
#     batch = batch.to(device)
#
#     # Forward pass
#     with torch.no_grad():
#         output = model(batch)



# Load image
#image_path = 'D:/data/sets/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg'
#image = Image.open(image_path)

# my_sample = nusc.sample[0]
# cam_front_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
# x = cam_front_data['filename']
# print(f'x= {x}')
# image_path = 'D:/data/sets/nuscenes/' + x
# print(image_path)
# image = Image.open(image_path)

# # Define image transformation pipeline
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),#RuntimeError: The size of tensor a (225) must match the size of tensor b (224) at non-singleton dimension 2
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Apply transformation pipeline
# input_image = transform(image).unsqueeze(0)

# # Load model
# model = vrn_unguided

# # Set model to evaluation mode
# model.eval()

# # Forward pass
# with torch.no_grad():
#     output = model(input_image)

# # Print the output shape of the predictions
# out = output[0]
# z = out[:, :, 10, 10]
# print(z.shape)