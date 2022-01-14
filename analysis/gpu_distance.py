from numba import cuda, types, jitclass
from tqdm import tqdm
import math as m
import argparse
import numpy as np

@cuda.jit()
def calc_distances(point_list, distance_mat):
    tid_x, tid_y = cuda.grid(2)
    if tid_x >= point_list.shape[0] or \
       tid_y >= point_list.shape[0] or \
       tid_x == tid_y:
       return
    poi =  point_list[tid_y]
    tgt = point_list[tid_x]
    xs = tgt[1] - poi[1]
    ys = tgt[2] - poi[2]
    distance = m.sqrt(float(xs**2 + ys**2))
    distance_mat[tid_x, tid_y] = int(distance)

@cuda.jit()
def memset_k(array, val):
    if len(array.shape) >= 2:
        #2d
        i, j = cuda.grid(2)
        if i < array.shape[0] and j < array.shape[1]:
            array[i, j] = val
    else:
        #1d
        i = cuda.grid(1)
        if i < array.shape[0]:
            array[i] = val


def set_memory(array, val):
    """
    Set a 2d or 1d ndarray to a given value using the custom kernel

    This function calls a kernel that set the memory space to a given value,
    should be fixed by using memset intead of the viewshed_kernel

    Parameters:
    array (ndarray): numba cuda array to be setted
    val (): value to set
    """

    if(len(array.shape)>=2 and array.shape[1] >= 16): #2df
        threadsperblock = (16, 16)
        blockspergrid_y = int(m.ceil(array.shape[1] / threadsperblock[1]))
    else:                     #1d or 2d smaller than 16
        threadsperblock = (16, 1)
        blockspergrid_y = 1
    blockspergrid_x = int(m.ceil(array.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    memset_k[blockspergrid, threadsperblock](array, val)

def gpu_distance(path, datasets, pe, te):
    for d in datasets:
        print("GPU distance for %s"%(d))
        point_list = np.genfromtxt("%s/%s_%d_%d/best_p.csv"%(path,d,pe,te), delimiter=',',dtype=np.uint32)
        building_n = len(point_list)
        distance_mat = cuda.device_array(shape=(building_n,
                                                building_n),
                                                dtype=np.uint16)
        set_memory(distance_mat, 0)
        threadsperblock = 16
        blockspergrid_x = int(m.ceil(building_n/ threadsperblock))
        cuda_pl = cuda.to_device(point_list)
        calc_distances[(blockspergrid_x, blockspergrid_x),
                       (threadsperblock, threadsperblock)](cuda_pl,
                                                           distance_mat)
        distance_mat.copy_to_host()
        print("Binning %s"%(d))
        hist = np.histogram(distance_mat[:][:], bins=range(0,int(np.amax(distance_mat)), 100))
        h = np.zeros(shape=(len(hist[0]), 2))
        h[:, 1] = hist[0]
        h[:, 0] = hist[1][:-1]
        np.savetxt("%s/%s_%d_%d/link_disthist.csv"%(path,d,pe,te), h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyize graphs')
    parser.add_argument('-b', "--base_path", help="path of the graph",
                             type=str, required=True)
    parser.add_argument('-pe', "--pole_elev", help="heigh of the pole in meters (2,4,10)", type=int, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    pe = te = args.pole_elev
    #datasets = ['mezzolombardo', 'barberino', 'sorrento', 'pontremoli', 'visciano', 'predaia','firenze', 'trento','napoli']
    datasets = ['predaia']
    gpu_distance(base_path, datasets, pe, te)
