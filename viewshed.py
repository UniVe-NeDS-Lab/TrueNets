from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np
from numba import cuda
import math
from kernels import viewshed_k, threads_n, sum_k, memset_k, take_values, knife_k, los_k, dist_k, CORNER_OVERLAP
from tqdm import tqdm


class Viewshed():
    def __init__(self,  max_dist: int = 0):
        self.max_dist = max_dist

    def prepare_cumulative_viewshed(self, raster_np: np.ndarray):
        """
        Prepare the cuda device for the upcoming cumulative viewshed computation.

        Allocate and copy the raster memory space, allocate the global memory
        for the computation of the single viewshed and the global memory for the
        cumulative adding

        Parameters:
        raster_np ([:][:]): 2d matrix containing the DSM of the area

        """
        self.raster = raster_np
        self.dsm_global_mem = cuda.to_device(raster_np)  # transfer dsm
        self.out_global_mem = cuda.device_array(shape=raster_np.shape, dtype=np.uint8)
        self.cumulative_global_mem = cuda.device_array(shape=raster_np.shape, dtype=np.uint16)  # with 16bit we can have up to 65k buildings
        self.set_memory(self.cumulative_global_mem, 0)
        self.set_memory(self.out_global_mem, 0)

    def cumulative_viewsheds(self,
                             coordinates_list: 'list[np.ndarray]',
                             poi_elev:
                             int, tgt_elev: int
                             ) -> np.ndarray:
        """
        Run the cumulative viewshed

        For each coordinate pair in the coordinates list it run the viewshed and sum the result

        Parameters:
        coordinates_list ([:][2]): n*2 matrix containing the coordinates of the
        point in the local projection

        """
        for c in tqdm(coordinates_list):
            self._run_viewshed(c, poi_elev, tgt_elev)
            self.sum_results()
        return self.cumulative_global_mem.copy_to_host()

    def single_viewshed(self,
                        raster_np: np.ndarray,
                        poi_coord: np.ndarray,
                        poi_elev: int,
                        tgt_elev: int,
                        poi_elev_type=0
                        ) -> np.ndarray:
        self.raster = raster_np
        self.dsm_global_mem = cuda.to_device(raster_np)  # transfer dsm
        self.out_global_mem = cuda.device_array(shape=raster_np.shape, dtype=np.uint8)
        self.set_memory(self.out_global_mem, 0)
        self._run_viewshed(poi_coord, poi_elev, tgt_elev, poi_elev_type)
        return self.out_global_mem.copy_to_host()

    def _run_viewshed(self,
                      poi_coord: np.ndarray,
                      poi_elev: int,
                      tgt_elev: int,
                      poi_elev_type=0
                      ) -> None:
        """
        Run the  single viewshed kernel

        Calculate the viewshed using Osterman algorithm from a point to the whole DSM.

        Parameters:
        poi_elev (int): height of the observer above the DSM
        tgt_elev (int): height of the target above the DSM
        """
        # calculate block size and thread number
        blocks_landscape = (self.raster.shape[0] +
                            2*self.raster.shape[0]/threads_n +
                            CORNER_OVERLAP +
                            threads_n-1)/threads_n
        block_upright = (self.raster.shape[1] +
                         2*self.raster.shape[1]/threads_n +
                         CORNER_OVERLAP +
                         threads_n-1)/threads_n
        blocks_n = max(block_upright, blocks_landscape)
        blockspergrid = (int(blocks_n), 4)
        threadsperblock = (threads_n, 1)
        viewshed_k[blockspergrid, threadsperblock](self.dsm_global_mem,
                                                   self.out_global_mem,
                                                   np.int32(poi_coord),
                                                   np.int16(self.max_dist),
                                                   np.int16(1),
                                                   np.int16(1),
                                                   np.float32(poi_elev),
                                                   np.float32(tgt_elev),
                                                   poi_elev_type)

    def set_memory(self,
                   array: DeviceNDArray,
                   val: int
                   ) -> None:
        """
        Set a 2d or 1d ndarray to a given value using the custom kernel

        This function calls a kernel that set the memory space to a given value,
        should be fixed by using memset intead of the viewshed_kernel

        Parameters:
        array (ndarray): numba cuda array to be setted
        val (): value to set
        """

        if(len(array.shape) >= 2 and array.shape[1] >= 16):  # 2df
            threadsperblock = (16, 16)
            blockspergrid_y = int(math.ceil(array.shape[1] / threadsperblock[1]))
        else:  # 1d or 2d smaller than 16
            threadsperblock = (16, 1)
            blockspergrid_y = 1
        blockspergrid_x = int(math.ceil(array.shape[0] / threadsperblock[0]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        memset_k[blockspergrid, threadsperblock](array, val)

    def sum_results(self) -> None:
        """
        Sum the results of the viewshed computation on another memory space and
        set the original one to 0

        This function calls a kernel that set the memory space to a given value

        Parameters:
        array (ndarray): numba cuda array to be setted
        val (): value to set
        """

        threadsperblock = (16, 16)
        blockspergrid_x = int(math.ceil(self.out_global_mem.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(self.out_global_mem.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        sum_k[blockspergrid, threadsperblock](self.out_global_mem, self.cumulative_global_mem)


# TrueNets (intervisibility graphs)

    def _extract_values(self, i: int) -> None:
        """
        Extract the values of the i-th computation and write them to the i-th line
        of the output matrix

        This function calls a kernel that takes all the points in the ordered list
        "building_list" , fetch their value and write it to the corresponding
        colum of the i-th line


        Parameters:
        array (ndarray): numba cuda array to be setted
        val (): value to set
        """
        threadsperblock = 16
        blockspergrid_x = int(math.ceil(self.building_n / threadsperblock))
        take_values[blockspergrid_x, threadsperblock](self.out_global_mem,
                                                      self.building_list,
                                                      self.intervisibility_mat,
                                                      i)

    def prepare_intervisibility(self, raster: np.ndarray) -> None:
        self.raster = raster
        self.dsm_global_mem = cuda.to_device(raster)  # transfer dsm
        self.out_global_mem = cuda.device_array(shape=raster.shape,
                                                dtype=np.uint8)
        self.set_memory(self.out_global_mem, 0)

    def generate_intervisibility_fast(self,
                                      raster: np.ndarray,
                                      coordinates: np.ndarray,
                                      poi_elev: int,
                                      tgt_elev: int
                                      ) -> np.ndarray:
        self.raster = raster
        self.dsm_global_mem = cuda.to_device(raster)
        self.building_n = coordinates.shape[0]
        self.building_list = cuda.to_device(coordinates)
        self.intervisibility_mat = cuda.device_array(shape=(self.building_n,
                                                            self.building_n),
                                                     dtype=np.uint8)
        self.set_memory(self.intervisibility_mat, 0)
        threadsperblock = 16
        blockspergrid_x = int(math.ceil(self.building_n / threadsperblock))
        los_k[(blockspergrid_x, blockspergrid_x),
              (threadsperblock, threadsperblock)](self.dsm_global_mem,
                                                  self.building_list,
                                                  self.intervisibility_mat,
                                                  np.uint16(1),
                                                  np.uint16(1),
                                                  np.float32(poi_elev),
                                                  np.float32(tgt_elev))
        return self.intervisibility_mat.copy_to_host()

    def calculate_distance(self,
                           adj_list: 'list[list]',
                           coordinates_dict: dict,
                           n_buildings: int
                           ) -> np.ndarray:

        self.intervisibility_mat = cuda.device_array(shape=(n_buildings, n_buildings),
                                                     dtype=np.float32)
        self.set_memory(self.intervisibility_mat, 0)
        for l in tqdm(adj_list):
            n_tgt = len(l)
            # create 2d array with id, y, x (id must be the index of the ordered array)
            adj_array = np.zeros(shape=(n_tgt, 3), dtype=np.int32)
            for j in range(0, n_tgt):
                adj_array[j] = coordinates_dict[l[j]]
            threadsperblock = 32
            blockspergrid_x = int(math.ceil(n_tgt / threadsperblock))
            dist_k[blockspergrid_x, threadsperblock](adj_array,
                                                     self.intervisibility_mat,
                                                     np.uint16(1),
                                                     np.uint16(1))
        return self.intervisibility_mat.copy_to_host()

    def knife_edge(self,
                   raster: np.ndarray,
                   adj_list: 'list[list]',
                   coordinates_dict: dict,
                   n_buildings: int,
                   poi_elev: int,
                   tgt_elev: int,
                   ple: float = 2.0,
                   f: float = 5
                   ) -> 'tuple[np.ndarray, np.ndarray]':
        self.dsm_global_mem = cuda.to_device(raster)
        lmb = 0.299792458/f
        self.intervisibility_mat = cuda.device_array(shape=(n_buildings, n_buildings),
                                                     dtype=np.uint16)  # TODO: check memory problems with more bit
        self.angles_mat = cuda.device_array(shape=(n_buildings, n_buildings, 2),
                                            dtype=np.int16)
        self.set_memory(self.intervisibility_mat, 0)
        for l in tqdm(adj_list):
            n_tgt = len(l)
            # create 2d array with id, y, x (id must be the index of the ordered array)
            adj_array = np.zeros(shape=(n_tgt, 3), dtype=np.int32)
            for j in range(0, n_tgt):
                adj_array[j] = coordinates_dict[l[j]]
            threadsperblock = 32  # a big number will slow down since we have heterogeneous links per block
            blockspergrid_x = int(math.ceil(n_tgt / threadsperblock))
            knife_k[blockspergrid_x, threadsperblock](self.dsm_global_mem,
                                                      adj_array,
                                                      self.intervisibility_mat,
                                                      self.angles_mat,
                                                      np.uint16(1),
                                                      np.uint16(1),
                                                      np.float32(poi_elev),
                                                      np.float32(tgt_elev),
                                                      np.float32(ple),
                                                      np.float32(lmb))
        return (self.intervisibility_mat.copy_to_host(), self.angles_mat.copy_to_host())
