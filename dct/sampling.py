"""Sample the operating range with different strategies."""

# python libraries


# 3rd party libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import qmc
from dessca import DesscaModel


def _normalize_single_point(x: np.ndarray, x_min: float, x_max: float, x_norm_min: float, x_norm_max: float) -> np.ndarray:
    """
    Normalize a single point x where x is in between x_min and x_max to a normalized point x_norm where x_norm is between x_norm_min and x_norm_max.

    :param x: point
    :type x: float
    :param x_min: minimum value which x can assume
    :type x_min: float
    :param x_max: maximum value which x can assume
    :type x_max: float
    :param x_norm_min: minimum value of the normalized area
    :type x_norm_min: float
    :param x_norm_max: maximum value of the normalized area
    :type x_norm_max: float
    :return: normalized point
    :rtype: float
    """
    x_norm_point = (x - x_min) * (x_norm_max - x_norm_min) / (x_max - x_min) + x_norm_min
    return x_norm_point


def _denormalize_single_point(x_norm: float | np.ndarray, x_min: float, x_max: float, x_norm_min: float, x_norm_max: float) \
        -> float | list[float] | np.ndarray:
    """
    Denormalize a single point x_norm where x_norm is in between x_norm_min and x_norm_max to a point x where x is between x_min and x_max.

    :param x_norm: point
    :type x_norm: float
    :param x_min: minimum value which x can assume
    :type x_min: float
    :param x_max: maximum value which x can assume
    :type x_max: float
    :param x_norm_min: minimum value of the normalized area
    :type x_norm_min: float
    :param x_norm_max: maximum value of the normalized area
    :type x_norm_max: float
    :return: denormalized point
    :rtype: float
    """
    x_denorm = (x_norm - x_norm_min) * (x_max - x_min) / (x_norm_max - x_norm_min) + x_min
    return x_denorm


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
            raise ValueError(f"Incorrect user-given operating point {point} not within {dim_min_max_list[0]} and {dim_min_max_list[1]}.")

def latin_hypercube(dim_1_min: float, dim_1_max: float, dim_2_min: float, dim_2_max: float, dim_3_min: float, dim_3_max: float, total_number_points: int,
                    dim_1_user_given_points_list: list[float], dim_2_user_given_points_list: list[float], dim_3_user_given_points_list: list[float],
                    sampling_random_seed: int | None) \
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
    :param sampling_random_seed: random seed for the hypercube sampling (reproducible)
    :type sampling_random_seed: int
    :return: dim_1_all_points, dim_2_all_points, dim_3_all_points
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # check if user points are within the given limits
    check_user_points_in_min_max_region([dim_1_min, dim_1_max], dim_1_user_given_points_list)
    check_user_points_in_min_max_region([dim_2_min, dim_2_max], dim_2_user_given_points_list)
    check_user_points_in_min_max_region([dim_3_min, dim_3_max], dim_3_user_given_points_list)

    if len(dim_1_user_given_points_list) != len(dim_2_user_given_points_list) or len(dim_1_user_given_points_list) != len(dim_3_user_given_points_list):
        raise ValueError("User-given points incomplete.")

    lower_bounds = [dim_1_min, dim_2_min, dim_3_min]
    upper_bounds = [dim_1_max, dim_2_max, dim_3_max]

    # latin hypercube sampler
    sampler = qmc.LatinHypercube(d=3, rng=sampling_random_seed)
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


def dessca(dim_1_min: float, dim_1_max: float, dim_2_min: float, dim_2_max: float, dim_3_min: float, dim_3_max: float, total_number_points: int,
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
    :return: dim_1_all_points, dim_2_all_points, dim_3_all_points
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # check if user points are within the given limits
    check_user_points_in_min_max_region([dim_1_min, dim_1_max], dim_1_user_given_points_list)
    check_user_points_in_min_max_region([dim_2_min, dim_2_max], dim_2_user_given_points_list)
    check_user_points_in_min_max_region([dim_3_min, dim_3_max], dim_3_user_given_points_list)

    if len(dim_1_user_given_points_list) != len(dim_2_user_given_points_list) or len(dim_1_user_given_points_list) != len(dim_3_user_given_points_list):
        raise ValueError("User-given points incomplete.")

    dim_1_user_given_points_list_norm = _normalize_single_point(np.array(dim_1_user_given_points_list), dim_1_min, dim_1_max, -1, 1)
    dim_2_user_given_points_list_norm = _normalize_single_point(np.array(dim_2_user_given_points_list), dim_2_min, dim_2_max, -1, 1)
    dim_3_user_given_points_list_norm = _normalize_single_point(np.array(dim_3_user_given_points_list), dim_3_min, dim_3_max, -1, 1)

    dessca_instance = DesscaModel(box_constraints=[[-1, 1],
                                                   [-1, 1],
                                                   [-1, 1]],
                                  state_names=["x1", "x2", "x3"],
                                  bandwidth=0.1)

    # transfer user-given points to the format needed by dessca
    if np.any(dim_1_user_given_points_list):
        scaled_points = [[dim_1_user_given_points_list_norm[count], dim_2_user_given_points_list_norm[count], dim_3_user_given_points_list_norm[count]]
                         for count, _ in enumerate(dim_1_user_given_points_list_norm)]

        dessca_instance.update_coverage_pdf(data=np.transpose(scaled_points))
    else:
        next_sample_suggest = dessca_instance.update_and_sample()
        scaled_points = [next_sample_suggest]

    next_sample_suggest = dessca_instance.update_and_sample()
    scaled_points = np.append(scaled_points, [next_sample_suggest], axis=0)  # type: ignore
    for _ in range(total_number_points - len(dim_1_user_given_points_list) - 1):
        next_sample_suggest = dessca_instance.update_and_sample(np.transpose([next_sample_suggest]))
        scaled_points = np.append(scaled_points, [next_sample_suggest], axis=0)  # type: ignore

    dim_1_all_points_scaled = np.array([points[0] for points in scaled_points])
    dim_2_all_points_scaled = np.array([points[1] for points in scaled_points])
    dim_3_all_points_scaled = np.array([points[2] for points in scaled_points])

    dim_1_all_points = _denormalize_single_point(dim_1_all_points_scaled, dim_1_min, dim_1_max, -1, 1)
    dim_2_all_points = _denormalize_single_point(dim_2_all_points_scaled, dim_2_min, dim_2_max, -1, 1)
    dim_3_all_points = _denormalize_single_point(dim_3_all_points_scaled, dim_3_min, dim_3_max, -1, 1)

    return dim_1_all_points, dim_2_all_points, dim_3_all_points  # type: ignore


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
