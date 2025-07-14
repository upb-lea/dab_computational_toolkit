"""Sample the operating range with different strategies."""

# python libraries


# 3rd party libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import qmc

def check_user_points_in_min_max_region(dim_min_max_list: list[float], dim_points_list: list[float]) -> None:
    """
    Check if given operating point is within the limits of the operating area.

    :param dim_min_max_list: e.g. [175, 295]
    :type dim_points_list: list[float]
    :param dim_points_list: list of user-given operating points, e.g. [200, 230]
    :type dim_min_max_list: list[float]
    :raises ValueError: in case of list is out of min/max limits
    """
    for point in dim_points_list:
        if point < dim_min_max_list[0] or point > dim_min_max_list[1]:
            raise ValueError(f"Incorrect user-given operating point {point} not within {dim_min_max_list[0]} and {dim_points_list[1]}.")

def latin_hypercube(dim_1_min: float, dim_1_max: float, dim_2_min: float, dim_2_max: float, dim_3_min: float, dim_3_max: float, total_number_points: int,
                    dim_1_user_given_points_list: list[float], dim_2_user_given_points_list: list[float], dim_3_user_given_points_list: list[float]) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Latin hypercube sampling for a given 3-dimensional user input.

    :param dim_1_min: dimension 1 minimum, e.g. 690
    :param dim_1_max: dimension 1 maximum, e.g. 710
    :param dim_2_min: dimension 2 minimum, e.g. 175
    :param dim_2_max: dimension 2 maximum, e.g. 295
    :param dim_3_min: dimension 3 minimum, e.g. -2000
    :param dim_3_max: dimension 3 maximum, e.g. 2000
    :param total_number_points: point to sample by the sampler
    :param dim_1_user_given_points_list: user-given points for dimension 1, e.g. [695, 705]
    :param dim_2_user_given_points_list: user-given points for dimension 2, e.g. [289, 299]
    :param dim_3_user_given_points_list: user-given points for dimension 3, e.g. [-1300, 1530]
    :return:
    """
    # check if user points are within the given limits
    check_user_points_in_min_max_region([dim_1_min, dim_1_max], dim_1_user_given_points_list)
    check_user_points_in_min_max_region([dim_2_min, dim_2_max], dim_2_user_given_points_list)
    check_user_points_in_min_max_region([dim_3_min, dim_3_max], dim_3_user_given_points_list)

    lower_bounds = [dim_1_min, dim_2_min, dim_3_min]
    upper_bounds = [dim_1_max, dim_2_max, dim_3_max]

    # latin hypercube sampler
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=total_number_points)

    # scale latin hypercube samples to the three dimensions
    scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)

    # add user-given points to the sampling if available
    if dim_1_user_given_points_list:
        user_given_points = [[dim_1_user_given_points_list[count], dim_2_user_given_points_list[count], dim_3_user_given_points_list[count]]
                             for count, _ in enumerate(dim_1_user_given_points_list)]

        # add the user-given operating points to the hypercube sampling
        scaled_sample = np.vstack((scaled_sample, np.array(user_given_points)))

    dim_1_all_points = np.array([points[0] for points in scaled_sample])
    dim_2_all_points = np.array([points[1] for points in scaled_sample])
    dim_3_all_points = np.array([points[2] for points in scaled_sample])

    return dim_1_all_points, dim_2_all_points, dim_3_all_points


def plot_samples(dim_1_min_max_list: list[float], dim_2_min_max_list: list[float], dim_3_min_max_list: list[float], number_user_points: int,
                 dim_1_points_list: list[float], dim_2_points_list: list[float], dim_3_points_list: list[float]) -> None:
    """
    Plot sampling in a 3D-room.

    :param dim_1_min_max_list: dimension 1 as min-max-list, e.g. [690, 710]
    :type dim_1_min_max_list: list[float]
    :param dim_2_min_max_list: dimension 2 as min-max-list, e.g. [175, 295]
    :type dim_2_min_max_list: list[float]
    :param dim_3_min_max_list: dimension 3 as min-max-list, e.g. [-2000, 2000]
    :type dim_3_min_max_list: list[float]
    :param number_user_points: Number of user given points (color change)#
    :type number_user_points: int
    :param dim_1_points_list: sampled points in dimension 1 (e.g. by latin hypercube)
    :type dim_1_points_list: list[float]
    :param dim_2_points_list: sampled points in dimension 2 (e.g. by latin hypercube)
    :type dim_2_points_list: list[float]
    :param dim_3_points_list: sampled points in dimension 3 (e.g. by latin hypercube)
    :type dim_3_points_list: list[float]
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # user given points in red
    ax.scatter(dim_1_points_list[:number_user_points], dim_2_points_list[:number_user_points], dim_3_points_list[:number_user_points],
               color='red', s=50, label='user given points')  # type: ignore
    # by algorithm added points in blue
    ax.scatter(dim_1_points_list[number_user_points:], dim_2_points_list[number_user_points:], dim_3_points_list[number_user_points:],
               color='blue', s=50, label='algorithm added points')  # type: ignore

    ax.set_xlim(dim_1_min_max_list[0], dim_1_min_max_list[1])
    ax.set_ylim(dim_2_min_max_list[0], dim_2_min_max_list[1])
    ax.set_zlim(dim_3_min_max_list[0], dim_3_min_max_list[1])  # type: ignore
    ax.set_xlabel(r'V_\mathrm{in} / V')
    ax.set_ylabel(r'V_\mathrm{out} / V')
    ax.set_zlabel(r'P_\mathrm{out} / W')  # type: ignore
    ax.legend()
    plt.tight_layout()
    plt.show()
