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