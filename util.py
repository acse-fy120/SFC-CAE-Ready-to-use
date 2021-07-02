import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import space_filling_decomp_new as sfc
import sys
import vtk
import numpy as np
import time
import glob
import progressbar
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.tri as tri
import meshio

# create an animation
from matplotlib import animation
from IPython.display import HTML
import cmocean

import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torchvision import transforms  # Transforms from torchvision

def read_in_files(data_path, file_format='vtu', vtu_field=None):
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[0].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    # file_format = data[0].split('.')[-1]
    file_format = '.' + file_format
    print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    cnt_progress = 0
    if (file_format == ".vtu"):
        print("Read in vtu Data......\n")
        bar=progressbar.ProgressBar(maxval=num_data)
        bar.start()
        data = []
        coords = None
        for i in range(num_data):
            # vtu_file = vtktools.vtu(F'{file_prefix}%d{file_format}' % i)
            vtu_file = meshio.read(F'{file_prefix}%d{file_format}' % i)
            if not (coords == vtu_file.points).all():
               coords = vtu_file.points
               print('mesh adapted at snapshot %d' % i)
            if not vtu_field in vtu_file.point_data.keys():
               raise ValueError(F'{vtu_field} not avaliable in {vtu_file.point_data.keys()} for {file_prefix} %d {file_format}' % i)
            data.append(vtu_file.point_data[vtu_field])
            cnt_progress +=1
            bar.update(cnt_progress)
        bar.finish()
        whole_data = np.array(data)
        if whole_data[..., whole_data.ndim - 1].max() - whole_data[..., whole_data.ndim - 1].min() < 1e-6: 
            whole_data = whole_data[..., :whole_data.ndim - 1]
        if coords[..., -1].max() - coords[..., -1].min() < 1e-6: coords = coords[..., :-1]
        # print(F'{vtu_field} has %d dimensions.'% whole_data.ndim)
        return whole_data, coords      

    elif (file_format == ".txt" or file_format == ".dat"):
        print("Read in txt/dat Data......")
        bar=progressbar.ProgressBar(maxval=num_data)
        data = []
        for i in range(num_data):
            data[i] = torch.from_numpy(np.loadtxt('{file_prefix} %d {file_format}' % i))
            cnt_progress +=1
            bar.update(cnt_progress)
        bar.finish()
        return torch.cat(data, -1)

def normalize_tensor(tensor):
    t_mean = torch.mean(tensor)
    t_std = torch.std(tensor)
    return (tensor - t_mean) / t_std, t_mean, t_std

def denormalize_tensor(normalized_tensor, t_mean, t_std):
    return normalized_tensor * t_std + t_mean

def get_sfc_curves_from_coords(coords, num):
    findm, colm, ncolm = sfc.form_spare_matric_from_pts(coords, coords.shape[0])
    colm = colm[:ncolm]
    curve_lists = []
    inv_lists = []
    ncurve = num
    graph_trim = -10  # has always been set at -10
    starting_node = 0 # =0 do not specifiy a starting node, otherwise, specify the starting node
    whichd, space_filling_curve_numbering = sfc.ncurve_python_subdomain_space_filling_curve(colm, findm, starting_node, graph_trim, ncurve, coords.shape[0], ncolm)
    for i in range(space_filling_curve_numbering.shape[-1]):
        curve_lists.append(np.argsort(space_filling_curve_numbering[:,i]))
        inv_lists.append(np.argsort(np.argsort(space_filling_curve_numbering[:,i])))

    return curve_lists, inv_lists

def plot_trace_vtu_2D(coords, levels):
    x_left = coords[:, 0].min()
    x_right = coords[:, 0].max()
    y_bottom = coords[:, 1].min()
    y_top = coords[:, 1].max()
    fig, ax = plt.subplots(figsize=(40,8))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    cuts = np.linspace(0, coords.shape[0], levels + 1).astype(np.int32)
    for i in range(levels):
        ax.plot(coords[cuts[i]:cuts[i+1], 0], coords[cuts[i]:cuts[i+1], 1], '-')
    plt.axis('off')
    plt.show() 

def countour_plot_vtu_2D(coords, levels, values=None, cmap = None):
    x = coords[:, 0]
    y = coords[:, 1]
    x_left = x.min()
    x_right = x.max()
    y_bottom = y.min()
    y_top = y.max()
    fig, ax = plt.subplots(figsize=(40,8))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    triang = tri.Triangulation(x, y)
    if values == None:
        values=np.arange(coords.shape[0])
    
    min_radius = 0.05
    # Mask off unwanted triangles.
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where((xmid - 0.2)**2 + (ymid - 0.2)**2 <= min_radius*min_radius, 1, 0)
    triang.set_mask(mask)
    plt.tricontourf(triang, values, levels = levels, cmap = cmap)    
    plt.axis('off')
    plt.show()  

class anim_vtu_fields_2D():
    def __init__(self, coords, values=None, levels = 15, cmap = None, steps = None, min_radius = 0.05, mask_x = 0.2, mask_y = 0.2):
       # initialize location of mesh
       self.x = coords[:, 0]
       self.y = coords[:, 1]
       self.x_left = self.x.min()
       self.x_right = self.x.max()
       self.y_bottom = self.y.min()
       self.y_top = self.y.max()
       self.fig, self.ax = plt.subplots(figsize=(40,8))
       self.ax.set_xlim(self.x_left, self.x_right)
       self.ax.set_ylim(self.y_bottom, self.y_top)
       self.triang = tri.Triangulation(self.x, self.y)
       self.cmap = cmap
       self.levels = levels
       self.values = np.array(values)

       if steps is None: 
           self.steps = self.values.shape[0]
       else: self.steps = steps
    
       self.min_radius = min_radius
       self.mask_x = mask_x
       self.mask_y = mask_y
       # Mask off unwanted triangles.
       xmid = self.x[self.triang.triangles].mean(axis=1)
       ymid = self.y[self.triang.triangles].mean(axis=1)
       mask = np.where((xmid - self.mask_x)**2 + (ymid - self.mask_y)**2 <= self.min_radius**2, 1, 0)
       self.triang.set_mask(mask)
       self.image = self.ax.tricontourf(self.triang, self.values[0], levels = self.levels, cmap = self.cmap)    
       self.ax.axis('off')
       # plt.show() 

    def update_grid(self, n_step: int):
       # self.image.set_array(self.values[n_step])
       self.image = self.ax.tricontourf(self.triang, self.values[n_step], levels = self.levels, cmap = self.cmap)
    #    print(n_step, self.values[n_step])
       print('frame %d saved.' % n_step)
       return self.image,
   
    def generate_anime(self):
       # anim = animation.FuncAnimation(self.fig, self.update_grid, frames = np.arange(0, self.steps))
       return animation.FuncAnimation(self.fig, self.update_grid, frames = np.arange(0, self.steps))

def find_plus_neigh(ordering):
    plus_neigh = np.zeros_like(ordering)
    plus_neigh[:-1] = ordering[1:]
    plus_neigh[-1] = ordering[-1]
    return plus_neigh

def find_minus_neigh(ordering):
    minus_neigh = np.zeros_like(ordering)
    minus_neigh[1:] = ordering[:-1]
    minus_neigh[0] = ordering[0]
    return minus_neigh

def ordering_tensor(tensor, ordering):
    return tensor[:, ordering]

class NearestNeighbouring(nn.Module):
    def __init__(self, size, initial_weight, num_neigh = 3):
        super(NearestNeighbouring, self).__init__()
        self.size = size
        self.num_neigh = num_neigh
        self.weights = nn.Parameter(torch.ones(size, num_neigh) * initial_weight)
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, tensor_list):
        return torch.mul(tensor_list, self.weights).sum(-1) + self.bias

def expend_SFC_NUM(sfc_ordering, partitions):
    size = len(sfc_ordering)
    sfc_ext = np.zeros(size * partitions, dtype = 'int')
    for i in range(partitions):
        sfc_ext[i * size : (i+1) * size] = i * size + sfc_ordering
    return sfc_ext

def find_size_conv_layers_and_fc_layers(size, stride, dims_latent, sfc_nums, input_channel, increase_multi, num_final_channels):
       channels = [input_channel]
       output_paddings = [size % stride]
       while size * num_final_channels  > 1200:
          size = size // stride + 1
          if num_final_channels >= input_channel * increase_multi: 
              input_channel *= increase_multi
              output_paddings.append(size % stride)
              channels.append(input_channel)
          else: 
              channels.append(num_final_channels)
              output_paddings.append(size % stride)
    
       inv_conv_start = size
       size *= sfc_nums * num_final_channels
       size_fc = [size]
       while size // (stride ** 1.5) > dims_latent: 
          size //= stride
          size_fc.append(size)
       size_fc.append(dims_latent)

       return len(channels) - 1, size_fc, channels, inv_conv_start, np.array(output_paddings[::-1][1:])



