from MPL import *

sys.path.append('/home/dimitris/PhD/PhD/visualDet3D')
sys.path.append('/home/dimitris/PhD/PhD/nuscenes')
from inference.mono_detection_inference_3 import *
# from devkit_dataloader.Dataloader import *

nusc_dataset = NuscenesDataset(nusc)

def test_mlp(index, nusc_dataset):
    #lidar points from nuscenes
    points, lidar_pc, masked_pc = nusc_dataset.get_points(index)
    lidar_points = torch.tensor(masked_pc)
    lidar_points = lidar_points.to('cuda:0')
    lidar_points = lidar_points.permute(1, 0)
    #obtain the z_box from detector, the z_image from encoder
    z_box = process_index(index, nusc_dataset, detector, cfg, test_func, backprojector, projector)
    z_image = process_data(index)
    z = torch.cat((z_box, z_image), 1)
    # lidar_points = lidar_points.unsqueeze(-1)  # Shape: [3044, 3, 1]
    # z = z.unsqueeze(-1)  # Shape: [3044, 263, 1]
    print(lidar_points.shape)
    print(z.shape)

    # Input dimensions
    D_in = lidar_points.shape[1]  # for x, y, z of the 3D point
    D_z = z.shape[1]  # for the conditioning vector

    # Define the MLP structure
    filter_channels = [D_in + D_z, 64, 128, 256, 512, 1]
    res_layers = [1, 2, 3]  # Skip connections before 2nd, 3rd, and 4th layers

    mlp = MLP(filter_channels, res_layers=res_layers, norm='group')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp = mlp.to(device)

    # Forward pass
    input_features = torch.cat([lidar_points, z], dim=1)  # concatenate x and z
    input_features = input_features.unsqueeze(0)  # Shape becomes [1, 3044, 266]
    input_features = input_features.transpose(1, 2)  # Swap the 2nd and 3rd dimensions
    input_features = input_features.to(device)
    print(input_features.shape)

    output, phi = mlp(input_features)
    """
    This phi tensor essentially gives you the activations at this layer, 
    which can be useful for various reasons depending on your application (e.g., feature visualization, additional processing, etc.)
    """
    print(output.shape)
    print(phi.shape)
    tenth_point_output = output[0, 0, 9].item()
    print(tenth_point_output)
    
test_mlp(0,nusc_dataset)
