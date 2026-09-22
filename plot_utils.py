import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
    
    
def plot_cliff_map_with_weight(cliff_map_data, mod='ours', version='max'):
    if mod == 'ours':
        weight_ind = -1
    elif mod == 'cliff':
        weight_ind = 8
    
    
    max_index_list = []
    
    location = cliff_map_data[0, :2]
    weight = cliff_map_data[0, weight_ind]
    speed = cliff_map_data[:, 2]
    orientation = cliff_map_data[:, 3]
    max_weight_index = 0

    for i in range(1, len(cliff_map_data)):
        tmp_location = cliff_map_data[i, :2]
        if (tmp_location == location).all():
            tmp_weight = cliff_map_data[i, weight_ind]
            if tmp_weight > weight:
                max_weight_index = i
                weight = tmp_weight
        else:
            max_index_list.append(max_weight_index)
            location = cliff_map_data[i, :2]
            weight = cliff_map_data[i, weight_ind]
            max_weight_index = i

    max_index_list.append(max_weight_index)

    (u, v) = pol2cart(speed, orientation)
    weight = cliff_map_data[:, weight_ind]

    colors = orientation  * 180 / np.pi
    colors = np.append(colors, [0, 360])
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.hsv

    for i in range(len(cliff_map_data)):
        if version == 'max':
            if i in max_index_list:
                plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=1, cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.004)
        elif version == 'all':
            plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.004)


    # sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    # cbar = plt.colorbar(sm, shrink = 0.5, ticks=[0, 90, 180, 270, 360], fraction=0.05)
    # cbar.ax.tick_params(labelsize=10)
    # plt.text(100, -17,"Orientation [deg]", rotation='vertical')
    
def plot_cliff_map_with_weight_cliff(cliff_map_data, mode="cliff", all_alpha1=False):
    if mode == "cliff":
        weight_index = 8
        speed_index = 3
        orientation_index = 2
    elif mode == "online":
        weight_index = 8
        speed_index = 2
        orientation_index = 3
    elif mode == "nemo":
        weight_index = -1
        speed_index = 2
        orientation_index = 3
    
    ## Only leave the SWND with largest weight
    max_index_list = []
    
    location = cliff_map_data[0, :2]
    weight = cliff_map_data[0, weight_index]
    speed = cliff_map_data[:, speed_index]
    orientation = cliff_map_data[:, orientation_index]
    max_weight_index = 0

    for i in range(1, len(cliff_map_data)):
        tmp_location = cliff_map_data[i, :2]
        if (tmp_location == location).all():
            tmp_weight = cliff_map_data[i, weight_index]
            if tmp_weight > weight:
                max_weight_index = i
                weight = tmp_weight
        else:
            max_index_list.append(max_weight_index)
            location = cliff_map_data[i, :2]
            weight = cliff_map_data[i, weight_index]
            max_weight_index = i

    max_index_list.append(max_weight_index)

    (u, v) = pol2cart(speed, orientation)
    weight = cliff_map_data[:, weight_index]
    colors = orientation  * 180 / np.pi
    colors = np.append(colors, [0, 360])
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.hsv

    for i in range(len(cliff_map_data)):
    # for i in range(200):
        ## For only plot max weight:
        # if i in max_index_list:
            # plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=1, cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.004)
        ## For only plot one point:
        # if cliff_map_data[i, 0] == 20 and cliff_map_data[i, 1] == -13:
        if all_alpha1:
            plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=1, cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.004)
        else:
            # plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.004)
            # plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=0.9)
            plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.01)


    # sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    # cbar = plt.colorbar(sm, shrink = 0.7, ticks=[0, 90, 180, 270, 360], fraction=0.05)
    # cbar.ax.tick_params(labelsize=15)
    # # plt.text(100, -17,"Orientation [deg]", rotation='vertical', fontsize=20)
    # cbar.set_label('Orientation [deg]', rotation=90, fontsize=20)


def plot_cliff_map_ethucy(cliff_map_data, transformer, mode="cliff", version="eth"):
    if mode in ["cliff", "online"]:
        weight_index = 8
        speed_index = 2
        orientation_index = 3
    elif mode == "nemo":
        weight_index = -1
        speed_index = 2
        orientation_index = 3

    ## Only leave the SWND with largest weight
    max_index_list = []
    
    location = cliff_map_data[0, :2]
    weight = cliff_map_data[0, weight_index]
    speed = cliff_map_data[:, speed_index]
    orientation = cliff_map_data[:, orientation_index]
    max_weight_index = 0

    #### filter out the cliffmap_data where speed is zero
    # non_zero_speed_indices = np.where(speed > 0.5)[0]
    # cliff_map_data = cliff_map_data[non_zero_speed_indices]


    for i in range(1, len(cliff_map_data)):
        tmp_location = cliff_map_data[i, :2]
        if (tmp_location == location).all():
            tmp_weight = cliff_map_data[i, weight_index]
            if tmp_weight > weight:
                max_weight_index = i
                weight = tmp_weight
        else:
            max_index_list.append(max_weight_index)
            location = cliff_map_data[i, :2]
            weight = cliff_map_data[i, weight_index]
            max_weight_index = i

    max_index_list.append(max_weight_index)

    (u, v) = pol2cart(speed, orientation)
    
    # combine u,v as one array
    uv = np.column_stack((u, v))
    # transform to pixel coordinates
    uv_pixel = transformer.world_to_image(uv)
    if version in ["eth", "hotel"]:
        u = uv_pixel[:, 1]
        v = uv_pixel[:, 0]
    elif version in ["zara01", "students003"]:
        u = uv_pixel[:, 0]
        v = uv_pixel[:, 1]
    ## move uv back to the original 
    
    ## original (0,0) to transform value
    zero = np.array([0, 0]).reshape(1, -1)
    zero_pixel = transformer.world_to_image(zero)
    if version in ["eth", "hotel"]:
        zero_pixel_u = zero_pixel[0,1]
        zero_pixel_v = zero_pixel[0,0]
    elif version in ["zara01", "students003"]:
        zero_pixel_u = zero_pixel[0,0]
        zero_pixel_v = zero_pixel[0,1]
    
    u = u - zero_pixel_u
    v = v - zero_pixel_v    
    
    weight = cliff_map_data[:, weight_index]

    if version in ["eth", "hotel"]:
        orientation_plot = np.degrees(np.arctan2(v, u))
        orientation_plot = (orientation_plot + 360) % 360   # wrap to [0, 360)
        orientation_plot = (360 - orientation_plot) % 360
        colors = orientation_plot
        # anchor the norm to [0, 360] so ticks match degrees
        colors_for_norm = np.append(colors, [0, 360])
        norm = Normalize(vmin=0, vmax=360)
        norm.autoscale(colors_for_norm)
        colormap = cm.hsv
    elif version in ["zara01", "students003"]:
        colors = orientation  * 180 / np.pi
        colors = np.append(colors, [0, 360])
        norm = Normalize()
        norm.autoscale(colors)
        colormap = cm.hsv

    for i in range(len(cliff_map_data)):
    # for i in range(200):
        ## For only plot max weight:
        # if i in max_index_list:
            # plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=1, cmap="hsv",angles='xy', scale_units='xy', scale=0.7)
        ## For only plot one point:
        # if cliff_map_data[i, 0] == 20 and cliff_map_data[i, 1] == -13:
        # if u[i]**2 + v[i]**2 > 0.01:
        plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=2, width=0.005)

    # sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    # sm.set_array([])  # optional but avoids some warnings

    # ax = plt.gca()    # get current axes created by plt.quiver
    # cbar = plt.colorbar(
    #     sm,
    #     ax=ax,
    #     shrink=0.7,
    #     ticks=[0, 90, 180, 270, 360],
    #     fraction=0.05,
    # )
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('Orientation [deg]', rotation=90, fontsize=20)
    
class HomographyTransformer:
    def __init__(self, homography_file):
        """Initialize by loading the homography matrix from file."""
        self.H = np.loadtxt(homography_file)  # Load the homography matrix from H.txt
        
        
        self.H_inv = np.linalg.inv(self.H)  # Compute the inverse matrix for world-to-pixel conversion
        
        self.scale = 1 / self.H[2, 2]

    def world_to_image(self, traj_w):
        traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T  
        # to camera frame
        traj_cam = np.matmul(self.H_inv, traj_homog)  
        # to pixel coords
        traj_uvz = np.transpose(traj_cam/traj_cam[2])
        
        return traj_uvz[:, :2].astype(int)

    def pixel_to_world(self, x_pixel, y_pixel):
        """Convert pixel coordinates (x, y) to world coordinates (meters)."""
        pixels = np.vstack((x_pixel, y_pixel, np.ones_like(x_pixel)))  # Create homogeneous coordinates (3, N)
        world_coords = self.H @ pixels  # Matrix multiplication (3, N)
        X_world, Y_world, scale = world_coords  # Extract results
        return X_world / scale, Y_world / scale  # Normalize

    def world_to_pixel(self, X_world, Y_world):
        """Convert world coordinates (meters) to pixel coordinates."""
        world_coords = np.vstack((X_world, Y_world, np.ones_like(X_world)))  # Homogeneous coordinates (3, N)
        pixel_coords = self.H_inv @ world_coords  # Matrix multiplication (3, N)
        x_pixel, y_pixel, scale = pixel_coords  # Extract results
        return y_pixel / scale, x_pixel / scale  # Normalize

    def scale_to_pixel_speed(self, speed_world):
        """Scale speed using a fixed approximation from H[2,2]."""
        return speed_world * self.scale  # Apply stored scale factor

from typing import Union
import torch
class PixWorldConverter:
    """Pixel to world converter"""

    def __init__(self, info: dict) -> None:
        self.resolution = info["resolution_pm"]  # 1pix -> m
        self.offset = np.array(info["offset"])

    def convert2pixels(
        self, world_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:

        if world_locations.ndim == 2:
            return (world_locations / self.resolution) - self.offset
        new_world_locations = [
            self.convert2pixels(world_location) for world_location in world_locations
        ]
        return (
            torch.stack(new_world_locations)
            if isinstance(world_locations, torch.Tensor)
            else np.stack(new_world_locations)
        )

    def convert2world(
        self, pix_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:
        return (pix_locations + self.offset) * self.resolution