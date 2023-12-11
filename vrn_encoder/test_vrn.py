import sys
sys.path.append('/home/dimitris/PhD/PhD/visualDet3D')
sys.path.append('/home/dimitris/PhD/PhD/nuscenes')
import torch
import torchvision.transforms as transforms
from PIL import Image
from vrn_model import vrn_unguided
from devkit_dataloader.nuscenes import NuScenes
from devkit_dataloader.nuscenes import NuScenesExplorer
from devkit_dataloader.Dataloader import *

def process_data(item_index):
    # Load dataset
    nusc_dataset = NuscenesDataset(nusc)

    # Get data
    input_image = nusc_dataset.get_item_vrn(item_index)
    input_image = input_image.unsqueeze(0)
    points, lidar_points, masked_pc = nusc_dataset.get_points(item_index)
    # print(points) #2996 2d points - projected sto image plane
    # print(points[0][0])
    
    # Load model
    model = vrn_unguided
    # Remember to load the model weights if necessary with something like:
    # model.load_state_dict(torch.load('model_weights.pth'))

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_image = input_image.to(device)
    print(f"input image to vrn shape : {input_image.shape}")
    # Set model to evaluation mode
    model.eval()
    # Forward pass
    with torch.no_grad():
        output = model(input_image)
    #print(output[0].shape) #torch.Size([1, 256, 512, 512])

    z_list = []
    for i in range(len(points[0][0])):
        out = output[0]
        z = out[:, :, int(points[0][0][i]), int(points[0][1][i])]
        z_list.append(z)
    z_image = torch.cat(z_list, dim=0).to(device)  # Move the tensor to the GPU

    return z_image

if __name__ == "__main__":
    z_image = process_data(52)
    print(z_image.shape)  # torch.Size([2996, 256])
    # #print(z_image[2995].shape)

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
