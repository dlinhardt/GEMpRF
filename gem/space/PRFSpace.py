# -*- coding: utf-8 -*-
"""

Created on Mon Feb 19 16:08:11 2024

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Siddharth Mittal",
"@Desc    :   None",
        
"""

import itertools
import time
import numpy as np
import cupy as cp
# from sklearn.neighbors import KDTree
from numba_kdtree import KDTree
from numba import njit
from typing import Callable
from joblib import Parallel, delayed, parallel_backend, effective_n_jobs
from gem.utils.gem_gpu_manager import GemGpuManager as ggm
from gem.utils.logger import Logger

#################################---------------------------------NUMBA Compatible Functions---------------------------------#################################
@njit(cache=True, nogil=True)
def combination_generator_jit(arrays):

    """
    NOTE: This function is used instead of itertools.product to generate a cartesian product of input arrays.
    Taken from: https://gist.github.com/hernamesbarbara/68d073f551565de02ac5
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list or tuple of arrays
        1-D arrays to form the cartesian product of.


    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
        [1, 4, 7],
        [1, 5, 6],
        [1, 5, 7],
        [2, 4, 6],
        [2, 4, 7],
        [2, 5, 6],
        [2, 5, 7],
        [3, 4, 6],
        [3, 4, 7],
        [3, 5, 6],
        [3, 5, 7]])

    """

    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)))


    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size

    n = arrays[-1].size
    for k in range(len(arrays)-2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j*m:(j+1)*m,k+1:] = out[0:m,k+1:]
    return out

@njit(nogil=True)
def multidim2flatIdx(point : list, shape : list)->int:
    """
    Compute the flat index for a multi-dimensional point.
    NOTE: The first dimensions are contiguous, rather than in the common order where the last dimensions are contiguous.
    NOTE: Considered order (Col, Row, dim3, dim4, ...), rather than "(dim4, dim3, Row, Col) or (row, col, dim3, dim4)"

    Args:
    - point (list): A list representing the coordinates of the point in each dimension.
    - shape (list): A list representing the shape of the multi-dimensional space.

    Returns:
    - int: The flat index corresponding to the given point.
    """
    flat_index = 0
    multiplier = 1

    # Iterate through each dimension in regular order
    for dim, coord in zip(shape, point):
        flat_index += coord * multiplier
        multiplier *= dim

    return flat_index


def compute_multidim_points_neighbours_multithreaded(all_multi_dim_points_arr_cpu: np.ndarray, num_spatial_dimensions: int, all_neighbours_indices_xy: np.ndarray, points_xy: np.ndarray,
                                             num_total_dimensions: int, num_extra_dimensions: int, extra_dimensions: list, num_neighbors: int = 9,
                                             validated_multidim_indices: np.ndarray = None, validated_mask: np.ndarray = None, n_jobs: int = -1):
    """
    Computes the multidimensional points neighbours (in terms of indices and Visual Field values) using multiprocessing.

    NOTE: `validated_multidim_indices` and `validated_mask` carry the same information but serve
    two different roles and are both required. The index array selects WHICH points get a
    neighbour list (fixing the output length and its alignment with `multi_dim_points_cpu`);
    the mask filters candidate neighbours WITHIN each list.

    NOTE: the work is dispatched in chunks rather than one task per point. Each point needs only
    ~100us of kernel time, so a task per point spends most of its life in joblib dispatch.

    Returns:
        list: A list of multidimensional points neighbours.
    """
    validated_points = all_multi_dim_points_arr_cpu[validated_multidim_indices]

    # Every replication of an xy point shares its xy neighbours, so the KD-tree was queried once per
    # unique xy point; recover which row of that result each multi-dimensional point should read.
    # Holds for any number of extra dimensions: flat = xy_idx * prod(extra dims) + extra_dim_flat,
    # with extra_dim_flat < prod(extra dims).
    product_shape_extra_dimensions = 1
    for dim in extra_dimensions:
        product_shape_extra_dimensions *= len(dim)
    xy_row_ids = np.asarray(validated_multidim_indices, dtype=np.int64) // product_shape_extra_dimensions

    num_workers = effective_n_jobs(n_jobs)
    num_chunks = max(1, min(len(validated_points), num_workers * 4))
    chunk_slices = np.array_split(np.arange(len(validated_points)), num_chunks)

    # Wrapper to perform the neighbor computation for a chunk of points
    def process_chunk(chunk):
        return get_points_neighbours_chunk_Numba(validated_points[chunk], xy_row_ids[chunk], num_spatial_dimensions, all_neighbours_indices_xy, points_xy, num_total_dimensions, num_extra_dimensions, extra_dimensions, num_neighbors, validated_mask)

    # Use threading backend because our "get_points_neighbours_chunk_Numba()"  uses @njit(nogil=True) decorator
    with parallel_backend('threading', n_jobs=n_jobs):
        results = Parallel()(delayed(process_chunk)(chunk) for chunk in chunk_slices)

    # Unpacking the results; chunks come back in dispatch order, so flattening restores point order
    multi_dim_points_neighbours_flat_indices_list = [arr for chunk_indices, _ in results for arr in chunk_indices]
    multi_dim_points_neighbours_vf_values_list = [arr for _, chunk_values in results for arr in chunk_values]

    return multi_dim_points_neighbours_flat_indices_list, multi_dim_points_neighbours_vf_values_list


@njit(nogil=True)
def get_points_neighbours_chunk_Numba(chunk_multi_dim_points: np.ndarray, chunk_xy_row_ids: np.ndarray, num_spatial_dimensions: int, all_neighbours_indices_xy: np.ndarray, points_xy: np.ndarray, num_total_dimensions: int, num_extra_dimensions: int, extra_dimensions: list, num_neighbors: int, validated_mask : np.ndarray):
    """
    Get neighbors of a chunk of multi-dimensional points.

    Args:
        chunk_multi_dim_points (ndarray): The multi-dimensional points for which to find neighbors.
        chunk_xy_row_ids (ndarray): For each point, the row of `all_neighbours_indices_xy` holding
            its xy neighbours. The xy neighbours depend only on the point's xy coordinates, so they
            are looked up here rather than re-queried once per replication of the same xy point.
        all_neighbours_indices_xy (ndarray): (len(points_xy), num_neighbors) KD-tree query result,
            computed once by the caller over the unique xy points.
        num_neighbors (int): The number of neighbors to find.
        validated_mask (ndarray): Boolean mask over the FULL multi-dimensional grid; a candidate
            neighbour is kept only where the mask is True. All-True when the caller does not
            validate. NOTE: `x in ndarray` is an O(N) linear scan in numba, so a mask is used
            here rather than an array of validated indices.

    Returns:
        tuple: two lists, holding per point the flat indices and the Visual Field values of its
        neighbours. Both are ragged: points near an edge keep fewer neighbours.
    """
    chunk_neighbours_flat_indices = []
    chunk_neighbours_values = []

    for point_idx in range(len(chunk_multi_dim_points)):
        multi_dim_point = chunk_multi_dim_points[point_idx]
        neighbours_indices_xy = all_neighbours_indices_xy[chunk_xy_row_ids[point_idx]]
        neighbours_xy = points_xy[neighbours_indices_xy]
        additional_dimensions = multi_dim_point[num_spatial_dimensions:]

        # gather the values and indices of the extra dimensions
        shape_extra_dimensions = [len(dim) for dim in extra_dimensions]
        extra_dimensions_neighbours = []
        extra_dimensions_neighbours_indices = []

        for dim_idx in range(num_extra_dimensions):
            dim_value = additional_dimensions[dim_idx]
            dim_value_idx = np.argmin(np.abs(extra_dimensions[dim_idx] - dim_value))
            dim_neighbours_indices = np.arange(max(0, dim_value_idx - 1), min(len(extra_dimensions[dim_idx]), dim_value_idx + 2))
            # # # # dim_neighbours_indices = np.array(np.arange(max(0, dim_value_idx - 1), min(len(extra_dimensions[dim_idx]), dim_value_idx + 2))) # NOTE: Earlier I used range() instead of np.arange()
            dim_neighbours = (extra_dimensions[dim_idx])[dim_neighbours_indices]
            extra_dimensions_neighbours_indices.append(dim_neighbours_indices) # reqired to get error gradients neighbours
            extra_dimensions_neighbours.append(dim_neighbours) # required to compute M-matrix

        # convert flat indices for 2D points to flat indices for our multi-dimensional points (to generate multi-dimensional points, we replicate the 2D points in all possible combinations of extra dimensions)
        product_shape_extra_dimensions = 1
        for dim in shape_extra_dimensions:
            product_shape_extra_dimensions *= dim

        # neighbours_indices_multi_dim_points = neighbours_indices_xy * np.prod(shape_extra_dimensions)
        neighbours_indices_multi_dim_points = neighbours_indices_xy * product_shape_extra_dimensions

        # generate all combinations of extra dimension neighbours
        num_neighbours_per_extra_dimension = [len(extra_dimensions_neighbours[dim_idx]) for dim_idx in range(num_extra_dimensions)]
        product_num_neighbours_per_extra_dimension = 1
        for dim in num_neighbours_per_extra_dimension:
            product_num_neighbours_per_extra_dimension *= dim
        total_neighbours = num_neighbors * product_num_neighbours_per_extra_dimension ##np.prod(num_neighbours_per_extra_dimension)

        current_point_neighbours_values = np.zeros((total_neighbours, num_total_dimensions))
        current_point_neighbours_flat_indices = np.zeros((total_neighbours, 1), dtype=np.int64)

        # both combinations are invariant in xy_idx, so build them once instead of per xy neighbour
        extra_dim_coordinates_combinations = combination_generator_jit(extra_dimensions_neighbours_indices)
        extra_dim_values_combinations = combination_generator_jit(extra_dimensions_neighbours)

        counter = 0
        for xy_idx in range(len(neighbours_xy)):
            for combination_idx in range(len(extra_dim_coordinates_combinations)):
            # # # # # # for dim_combination in itertools.product(*extra_dimensions_neighbours): ### NOTE: this is not NUMBA compatible

                # neighbours indices
                multi_dim_flat_idx = neighbours_indices_multi_dim_points[xy_idx]
                extra_dim_coordinates = extra_dim_coordinates_combinations[combination_idx]
                extra_dim_flat = multidim2flatIdx(point=list(extra_dim_coordinates), shape=list(shape_extra_dimensions))

                # current_point_neighbours_flat_indices[counter] = multi_dim_flat_idx + extra_dim_flat #* len(points_xy) # NOTE: original
                multi_dim_flat_idx = multi_dim_flat_idx + extra_dim_flat #* len(points_xy) # add extra-dimensions offset

                # keep the index and neighburs values only if the multi-dimensional point is validated
                if not validated_mask[int(multi_dim_flat_idx)]:
                    continue
                current_point_neighbours_flat_indices[counter] = multi_dim_flat_idx

                # neighbours values
                extra_dim_values = extra_dim_values_combinations[combination_idx]
                current_point_neighbours_values[counter, :num_spatial_dimensions] = neighbours_xy[xy_idx]
                current_point_neighbours_values[counter, num_spatial_dimensions:] = extra_dim_values

                # update counter
                counter += 1

        # keeping only the filled values (i.e. valid neighbours)
        chunk_neighbours_flat_indices.append(current_point_neighbours_flat_indices[: counter, :])
        chunk_neighbours_values.append(current_point_neighbours_values[: counter, :])

    return chunk_neighbours_flat_indices, chunk_neighbours_values



################################---------------------------------PRFSPACE CLASS---------------------------------################################
class PRFSpace:
    """
    Represents a class for handling PRF (Population Receptive Field) points.

    Args:
        points_xy (numpy.ndarray): Array of XY points.
        additional_dimensions (list): List of additional dimensions.

    Attributes:
        kdtree (scipy.spatial.KDTree): KDTree for efficient nearest neighbor search.
        extra_dimensions (list): List of additional dimensions.
        num_extra_dimensions (int): Number of additional dimensions.
        num_total_dimensions (int): Total number of dimensions.
        num_spatial_dimensions (int): Number of spatial dimensions (i.e., dimensions of XY plane).
        __multi_dim_points_arr (list): List of multidimensional points.
        all_xy_points_vs_2d_nearest_neighbours (numpy.ndarray): Array of XY points vs 2D nearest neighbors.
        _multi_dim_points_list (numpy.ndarray): Array of multidimensional points.

    Properties:
        points_xy (numpy.ndarray): Array of XY points.

    Methods:
        compute_all_xy_points_neighbours(num_neighbors): Computes all XY points neighbors.
        convert_spatial_to_multidim(): Converts spatial points to multidimensional points.
        compute_xy_to_multi_dimensional_neighbours(): Computes 3D neighbors for each 2D point.
        get_single_point_neighbours(multi_dim_point, num_neighbors): Gets neighbors of a single multi-dimensional point.
        _3d_model_test_visualization(query_point, nearest_points): Visualizes the 3D model test.
    """
    
    def __init__(self, points_xy : np.ndarray, additional_dimensions : tuple):
        self._points_xy = points_xy
        # # self.kdtree = KDTree(points_xy, leaf_size=30, metric='euclidean') # NOTE: orginal KDTree
        self.kdtree = KDTree(points_xy, leafsize=30) # Numba compatible KDTree
        self.extra_dimensions = additional_dimensions
        self.num_extra_dimensions = len(self.extra_dimensions)
        self.num_total_dimensions = points_xy.shape[1] + len(additional_dimensions)
        self.num_spatial_dimensions = points_xy.shape[1] # i.e. the dimensions of XY plane i.e. 2
        # self.__multi_dim_points_arr_cpu = []
        self.all_xy_points_vs_2d_nearest_neighbours = None
        self.__all_multi_dim_points_arr_cpu = None # these are possible multi-dimensional points WITHOUT validation
        self.__multi_dim_points_arr_gpu = None
        self.__multi_dim_points_neighbours_vf_values_list = []
        self.__multi_dim_points_neighbours_flat_indices_list = []

        #NOTE: Validation of multi-dimensional points
        # NOTE: !!!!! if the validated pRF points are not computed until the function "compute_multidim_points_neighbours()" is called, ....
        # ...it means the user does not want to validate the points. In that case, we consider all multi-dimensional points are validated inside the function "compute_multidim_points_neighbours()" !!!!!
        self.__validated_multidim_indices = None
        self.__full_2_validated_indices_mapping_dict = None

        self.__grid_steps = None  # cached per-dimension grid spacing (see get_grid_steps)

    @property
    def points_xy(self):
        """
        Array of XY points.

        Returns:
            numpy.ndarray: Array of XY points.
        """
        return self._points_xy
    
    @property
    def multi_dim_points_cpu(self)->np.ndarray:
        """
        Get the multi-dimensional points array allocated on CPU.

        Returns:
            numpy.ndarray: The multi-dimensional points array.
        """
        if self.__validated_multidim_indices is None:
            return self.__all_multi_dim_points_arr_cpu

        return self.__all_multi_dim_points_arr_cpu[self.__validated_multidim_indices]
    
    @property
    def multi_dim_points_gpu(self)->cp.ndarray:
        """
        Get the multi-dimensional points array allocated on GPU.

        Returns:
            cupy.ndarray: The multi-dimensional points array.
        """
        if self.__multi_dim_points_arr_gpu is None:
            self.__multi_dim_points_arr_gpu = ggm.get_instance().execute_cupy_func_on_default(cp.asarray, cupy_func_args=(self.__all_multi_dim_points_arr_cpu,))

        if self.__validated_multidim_indices is None:
            return self.__multi_dim_points_arr_gpu

        return self.__multi_dim_points_arr_gpu[self.__validated_multidim_indices]    
    
    @property
    def multi_dim_points_vf_neighbours(self):
        """
        Returns the multi-dimensional points neighbours in Visual Field values.
        
        Returns:
            list: A list of multi-dimensional points neighbours.
        """
        return self.__multi_dim_points_neighbours_vf_values_list
    
    @property
    def multi_dim_points_neighbours_flat_indices(self):
        """
        Returns the multi-dimensional points neighbours indices. 
        NOTE: The first value in each row is the flat index of the XY point, and rest are the coordinates of extra dimensions.
        
        Returns:
            list: A list of multi-dimensional points neighbours indices.
        """
        return self.__multi_dim_points_neighbours_flat_indices_list

    def get_grid_steps(self)->np.ndarray:
        """
        Per-dimension spacing between adjacent grid points, in the column order of
        ``multi_dim_points_cpu`` (i.e. [x, y, sigma, ...]).

        Computed once (cached) as the median of the diffs between the sorted unique
        values of each column. This is exact for the default uniform ``linspace`` grids
        and robust to a non-uniform / file-defined sigma grid.

        Returns:
            numpy.ndarray: 1-D array of length ``num_total_dimensions`` with the grid step
            of each dimension. A dimension with a single unique value yields a step of 0.
        """
        if self.__grid_steps is not None:
            return self.__grid_steps

        def _spacing(values):
            unique_values = np.unique(np.asarray(values).ravel())
            return float(np.median(np.diff(unique_values))) if unique_values.size > 1 else 0.0

        # Derived from the small 1-D generators (spatial x/y columns + each extra dimension),
        # NOT from a full copy of the multi-dim point cloud, so this stays cheap.
        steps = [_spacing(self.points_xy[:, col]) for col in range(self.points_xy.shape[1])]
        steps += [_spacing(dim) for dim in self.extra_dimensions]

        self.__grid_steps = np.array(steps, dtype=np.float64)
        return self.__grid_steps

    @staticmethod
    def make_extra_dimensions(*args):
        """
        Create extra dimensions based on the given arguments.

        Args:
            *args: Variable number of arguments.

        Returns:
            Tuple: A tuple containing the input arguments.

        """
        return args

    def convert_spatial_to_multidim(self):
        """
        Converts spatial points to multidimensional points.

        Returns:
            numpy.ndarray: Array of multidimensional points.
        """
        num_replications_per_extra_dimension = np.array([len(dim) for dim in self.extra_dimensions])
        num_multi_dim_points_per_xy_point = np.prod(num_replications_per_extra_dimension)
        self.__all_multi_dim_points_arr_cpu = np.zeros((len(self.points_xy) * num_multi_dim_points_per_xy_point, self.num_total_dimensions), dtype=np.float64)

        for point_idx in range(len(self.points_xy)):
            current_point_xy = self.points_xy[point_idx]
            total_replications = np.prod(num_replications_per_extra_dimension)
            current_point_replications = np.zeros((total_replications, self.num_total_dimensions))
            counter = 0

            # Nested loops to generate all combinations of extra dimension indices
            dim_indices = [0] * len(self.extra_dimensions)
            for i in range(total_replications):
                for j in range(len(self.extra_dimensions)):
                    dim_indices[j] = (i // np.prod(num_replications_per_extra_dimension[j+1:])) % num_replications_per_extra_dimension[j]

                # Fill the replicated points
                current_point_replications[counter, :self.num_spatial_dimensions] = current_point_xy
                current_point_replications[counter, self.num_spatial_dimensions:] = [self.extra_dimensions[k][dim_indices[k]] for k in range(len(self.extra_dimensions))]
                counter += 1
            self.__all_multi_dim_points_arr_cpu[point_idx * num_multi_dim_points_per_xy_point : (point_idx + 1) * num_multi_dim_points_per_xy_point] = current_point_replications
                    
        return self.__all_multi_dim_points_arr_cpu     

    @classmethod
    def combination_generator(cls, arrays):
        result = combination_generator_jit(arrays)
        return result

    # NOTE: This is just a Wrapper function to call the NUMBA compatible function
    def compute_multidim_points_neighbours(self):
        """
        Computes the multidimensional points neighbours (in terms of indices and Visual Field values)

        Returns:
            list: A list of multidimensional points neighbours.            
        """
        if self.__multi_dim_points_neighbours_vf_values_list:
            return self.__multi_dim_points_neighbours_flat_indices_list, self.__multi_dim_points_neighbours_vf_values_list
        
        # NOTE: !!!!! if the validated pRF points are not computed yet, it means the user does not want to validate the points. In this case, we consider all multi-dimensional points are validated !!!!!
        if self.__validated_multidim_indices is None:
            self.__validated_multidim_indices = np.arange(len(self.__all_multi_dim_points_arr_cpu))

        # Boolean mask over the full grid, for the O(1) membership test inside the numba kernel.
        # All-True when the caller did not validate (indices are then a full arange).
        validated_mask = np.zeros(len(self.__all_multi_dim_points_arr_cpu), dtype=np.bool_)
        validated_mask[self.__validated_multidim_indices] = True

        num_neighbors = 9
        _num_validated = len(self.__validated_multidim_indices)

        # The xy neighbours of a multi-dimensional point depend only on its xy coordinates, and every
        # xy point is replicated across all combinations of the extra dimensions. So query the tree
        # once per unique xy point rather than once per replication.
        _t0 = time.time()
        _, all_neighbours_indices_xy, _ = self.kdtree.query(self.points_xy, k=num_neighbors)
        Logger.print_timing_message(f"kdtree.query alone ({len(self.points_xy)} xy points, k={num_neighbors}): {time.time() - _t0:.2f}s")

        _t_neighbours_start = time.time()
        # Call the NUMBA compatible function to compute the multidimensional points neighbours
        self.__multi_dim_points_neighbours_flat_indices_list, self.__multi_dim_points_neighbours_vf_values_list = compute_multidim_points_neighbours_multithreaded(
            self.__all_multi_dim_points_arr_cpu,
            self.num_spatial_dimensions,
            np.ascontiguousarray(all_neighbours_indices_xy),
            self.points_xy,
            self.num_total_dimensions,
            self.num_extra_dimensions,
            list(self.extra_dimensions),
            num_neighbors=num_neighbors,
            validated_multidim_indices = self.__validated_multidim_indices,
            validated_mask = validated_mask)
        Logger.print_timing_message(f"neighbour search total ({_num_validated} validated points, k={num_neighbors}): {time.time() - _t_neighbours_start:.2f}s")

        return self.__multi_dim_points_neighbours_flat_indices_list, self.__multi_dim_points_neighbours_vf_values_list

    # validate the multi-dimensional points (e.g. in case of DoG model, sigma2 should be greater than sigma1)
    def keep_validated_sampling_points(self, validation_function: Callable)->None:
        """
        Extracts validated sampling points and updates the original multi-dimensional points array to keep ONLY the validated points 
        (i.e., checks that the pRF point fulfills the requirement of the selected model).

        Args:
            validation_function (Callable): A function that validates the sampling points. This is varies based on the selected pRF model.

        Returns:
            numpy.ndarray: Array of validated sampling points.
        """
        #self.__multi_dim_points_arr_cpu = validation_function(self.__multi_dim_points_arr_cpu) # update the computed multi-dimensional points array
        self.__validated_multidim_indices = validation_function(self.__all_multi_dim_points_arr_cpu) # update the computed multi-dimensional points array

        return self.__validated_multidim_indices

    def get_full_2_validated_indices(self, *indices, invalid_key_value: int = -1)->np.ndarray:
        if self.__full_2_validated_indices_mapping_dict is None:
            self.__full_2_validated_indices_mapping_dict = {idx: i for i, idx in enumerate(self.__validated_multidim_indices)}

        # Flatten the input numpy array and convert to a list of individual indices
        indices_list = np.ravel(indices[0]).tolist()
        mapped_indices = [self.__full_2_validated_indices_mapping_dict.get(idx, invalid_key_value) for idx in indices_list]

        return np.array(mapped_indices, dtype=np.int64).reshape(len(mapped_indices), 1)

    def _3d_model_test_visualization(self, multi_dim_query_point, nearest_points):
        """
        Visualizes the 3D model test by plotting the query point, all points, and nearest points in a 3D scatter plot.

        Args:
            query_point (list): The coordinates of the query point in the form [x, y, z].
            nearest_points (list): A list of nearest points, where each point is represented as [x, y, z].

        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all points
        x_points = self.__all_multi_dim_points_arr_cpu[:, 0] # first column containing all X
        y_points = self.__all_multi_dim_points_arr_cpu[:, 1] # second column containing all Y
        z_points = self.__all_multi_dim_points_arr_cpu[:, 2] # in case of 3D model, there is going to be only one extra dimenion (i.e. sigma)
        ax.scatter(x_points, y_points, z_points, c='b', label='All Points')

        # Plot query point
        ax.scatter(multi_dim_query_point[0], multi_dim_query_point[1], multi_dim_query_point[2], c='r', label='Query Point')

        # Plot nearest points and draw lines
        x_nearest = [point[0] for point in nearest_points]
        y_nearest = [point[1] for point in nearest_points]
        z_nearest = [point[2] for point in nearest_points]
        ax.scatter(x_nearest, y_nearest, z_nearest, c='g', label='Nearest Points')

        for i in range(len(nearest_points)):
            ax.plot([multi_dim_query_point[0], x_nearest[i]], [multi_dim_query_point[1], y_nearest[i]], [multi_dim_query_point[2], z_nearest[i]], c='g')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Queries Point and Nearest Neighbors')
        plt.legend()
        plt.show()




######################################-----------------TESTING-----------------######################################
      # NOTE: This is a test code to check the functionality of the PRFSpace class.
#       The test code is not part of the actual implementation and is only used for testing purposes.
#       Used to visualize the neighbors of a single point in 3D space.
if __name__ == "__main__":
    # Test combination generator
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5])
    result = PRFSpace.combination_generator((arr1, arr2)) # takes a bit long time for first run
    result = PRFSpace.combination_generator((arr2, arr1))  

    for i in zip(PRFSpace.combination_generator((arr1, arr2)), PRFSpace.combination_generator((arr2, arr1))):
        print(i)

    import matplotlib.pyplot as plt
    from enum import Enum            
    class QueryPointLocation(Enum):
        BOUNDARY_3D = 0
        CENTER_3D = 1
        BOUNDARY_4D = 2
        CENTER_4D = 3

    # NOTE: change the query point location here
    # query_point_location = QueryPointLocation.CENTER_4D 
    query_point_location = QueryPointLocation.BOUNDARY_3D 

    numX = 5
    numY = 3
    numSigma = 3
    num_neighbours_per_xy_position = 5
    x_values = np.linspace(-8, +8, numX, dtype=np.float64)
    y_values = np.linspace(-9, +9, numY, dtype=np.float64)
    sigma_values = np.linspace(0.5, 3, numSigma, dtype=np.float64)
    sigma_dummy = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    y, x = np.meshgrid(y_values, x_values) # row, column
    points_xy = np.column_stack((y.ravel(), x.ravel()))


    #NOTE: add each extra dimension in the tuple for Numba compatibility
    if query_point_location == QueryPointLocation.BOUNDARY_3D or query_point_location == QueryPointLocation.CENTER_3D:
        additional_dimensions = PRFSpace.make_extra_dimensions(sigma_values)
    elif query_point_location == QueryPointLocation.BOUNDARY_4D or query_point_location == QueryPointLocation.CENTER_4D:
        additional_dimensions = PRFSpace.make_extra_dimensions(sigma_values, sigma_dummy)

    prf_space = PRFSpace(points_xy, additional_dimensions=additional_dimensions)
    multi_dim_points = prf_space.convert_spatial_to_multidim()

    # NOTE: test extracting validated sampling points based on the set visual_field_radius
    import sys
    sys.path.append(r'/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/smittal/dgx-versions/fmri/')
    sys.path.append(r'/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/smittal/dgx-versions/fmri/codebase')
    sys.path.append(r'D:/code/sid-git/fmri')
    sys.path.append(r'D:\code\sid-git\fmri\codebase')
    from gem.model.prf_gaussian_model import PRFGaussianModel
    prf_model = PRFGaussianModel(visual_field_radius=9.2)
    validated_multdim_points_indices = prf_space.keep_validated_sampling_points(prf_model.get_validated_sampling_points_indices)

    # Compute multi-dimensional neigbours
    multi_dim_points_neighbours_list = prf_space.compute_multidim_points_neighbours(validated_multidim_indices=validated_multdim_points_indices)



    # test visualization    
    num_xy_points = len(prf_space.points_xy)
    num_points_in_one_xy_plane = len(prf_space.points_xy)
    dummy_query_point_z_idx = len(sigma_values)//2
    dummy_query_point_z = sigma_values[dummy_query_point_z_idx]            

    if query_point_location == QueryPointLocation.BOUNDARY_3D:        
        dummy_query_point_x_idx = 0 
        dummy_query_point_y_idx = 0
        dummy_query_point = np.array([prf_space.points_xy[dummy_query_point_x_idx, 0], prf_space.points_xy[dummy_query_point_y_idx, 1], sigma_values[dummy_query_point_z_idx]]) # 3D
        dummy_point_nearest_neighbours_indices, dummy_point_nearest_neighbours_values = prf_space.get_single_point_neighbours(dummy_query_point, num_neighbors=9)
        prf_space._3d_model_test_visualization(dummy_query_point, dummy_point_nearest_neighbours_values)
    elif query_point_location == QueryPointLocation.CENTER_3D:
        dummy_query_point_x_idx = num_xy_points//2
        dummy_query_point_y_idx = num_xy_points//2                
        dummy_query_point = np.array([prf_space.points_xy[dummy_query_point_x_idx, 0], prf_space.points_xy[dummy_query_point_y_idx, 1], sigma_values[dummy_query_point_z_idx]])     # 3D   
        dummy_point_nearest_neighbours_indices, dummy_point_nearest_neighbours_values = prf_space.get_single_point_neighbours(dummy_query_point, num_neighbors=9)
        prf_space._3d_model_test_visualization(dummy_query_point, dummy_point_nearest_neighbours_values)
    elif query_point_location == QueryPointLocation.CENTER_4D:
        dummy_query_point_x_idx = num_xy_points//2
        dummy_query_point_y_idx = num_xy_points//2                        
        dummy_query_point = np.array([prf_space.points_xy[dummy_query_point_x_idx, 0], prf_space.points_xy[dummy_query_point_y_idx, 1], sigma_values[dummy_query_point_z_idx], 2.2]) # 4D
        dummy_point_nearest_neighbours_indices, dummy_point_nearest_neighbours_values = prf_space.get_single_point_neighbours(dummy_query_point, num_neighbors=9)
        print(dummy_point_nearest_neighbours_values)

    print("Done!")

