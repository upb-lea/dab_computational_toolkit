import numpy as np
from dct.topology.dab.dab_datasets import HandleDabDto
from dct.topology.dab import dab_datasets_dtos as d_dtos
import logging
import os
from dct.constant_path import GECKO_COMPONENT_MODELS_DIRECTORY

logger = logging.getLogger(__name__)


f_s_suggest = 200_000
l_s_suggest = 137e-6
l_1_suggest = 618e-6
l_2__suggest = 606e-6
n_suggest = 3.975
transistor_1_name_suggest = 'CREE_C3M0120100J'
transistor_2_name_suggest = 'CREE_C3M0060065J'
t_dead_1_max=1000e-9
t_dead_2_max=1000e-9
c_par_1=16e-12
c_par_2=16e-12
transistor_1_name_list = ['CREE_C3M0065100J', 'CREE_C3M0120100J']
transistor_2_name_list = ['CREE_C3M0060065J', 'CREE_C3M0120065J']

sampling_points=27
v1_min_max_list=[700,700]
v2_min_max_list=[175,295]
p_min_max_list=[500, 2000]

circuit_study_name="döner"
sampling="meshgrid"
optimization_directory='/home/nikolasf/Downloads/2026-03-02_test'
#optimization_directory='/home/nikolasf/Downloads/2026-03-02_test_new_algorithm'

if not os.path.exists(optimization_directory):
    os.makedirs(optimization_directory)

HandleDabDto.set_c_oss_storage_directory('')
transistor_1_dto_list = []
transistor_2_dto_list = []

for transistor in transistor_1_name_list:
    transistor_1_dto_list.append(HandleDabDto.tdb_to_transistor_dto(transistor))

for transistor in transistor_2_name_list:
    transistor_2_dto_list.append(HandleDabDto.tdb_to_transistor_dto(transistor))

# choose sampling method
steps_per_dimension = int(np.ceil(np.power(sampling_points, 1 / 3)))
logger.info(f"Number of sampling points has been updated from {sampling_points} to {steps_per_dimension ** 3}.")
logger.info("Note: meshgrid sampling does not take user-given operating points into account")
if v1_min_max_list[0] == v1_min_max_list[1]:
    v1_dimension = np.array([v1_min_max_list[0]])
    logger.info(f"Reducing v1 dimension to {v1_min_max_list[0]} V.")
else:
    v1_dimension = np.linspace(v1_min_max_list[0], v1_min_max_list[1], steps_per_dimension)
if v2_min_max_list[0] == v2_min_max_list[1]:
    v2_dimension = np.array(v2_min_max_list[0])
    logger.info(f"Reducing v2 dimension to {v2_min_max_list[0]} V.")
else:
    v2_dimension = np.linspace(v2_min_max_list[0], v2_min_max_list[1], steps_per_dimension)
if p_min_max_list[0] == p_min_max_list[1]:
    p_dimension = np.array(p_min_max_list[0])
    logger.info(f"Reducing p dimension to {p_min_max_list[0]} W.")
else:
    p_dimension = np.linspace(p_min_max_list[0], p_min_max_list[1], steps_per_dimension)
v1_operating_points, v2_operating_points, p_operating_points = np.meshgrid(v1_dimension, v2_dimension, p_dimension, sparse=False)


logger.debug(f"{v1_operating_points=}")

# calculate weighting

weight_sum = 0
given_user_points = 0


if weight_sum > 1 or weight_sum < 0:
    raise ValueError("Sum of weighting point list must be within 0 and 1.")
else:
    leftover_auto_weight = (1 - weight_sum) / (v1_operating_points.size - given_user_points)
    logger.info(f"Auto-weight given for all other {v1_operating_points.size - given_user_points} operating points: {leftover_auto_weight}")
    # default case, same weights for all points
    weights = np.full_like(v1_operating_points, leftover_auto_weight)

for _, transistor_dto in enumerate(transistor_1_dto_list):
    if transistor_dto.name == transistor_1_name_suggest:
        transistor_1_dto: d_dtos.TransistorDTO = transistor_dto

for _, transistor_dto in enumerate(transistor_2_dto_list):
    if transistor_dto.name == transistor_2_name_suggest:
        transistor_2_dto: d_dtos.TransistorDTO = transistor_dto

dab_calc = HandleDabDto.init_config(
    name=circuit_study_name,
    mesh_v1=v1_operating_points,
    mesh_v2=v2_operating_points,
    mesh_p=p_operating_points,
    sampling=sampling,
    n=n_suggest,
    ls=l_s_suggest,
    fs=f_s_suggest,
    lc1=l_1_suggest,
    lc2=l_2__suggest / n_suggest ** 2,
    lossfilepath=os.path.join(optimization_directory, GECKO_COMPONENT_MODELS_DIRECTORY),
    c_par_1=c_par_1,
    c_par_2=c_par_2,
    transistor_dto_1=transistor_1_dto,
    transistor_dto_2=transistor_2_dto,
    t_dead_1_max=t_dead_1_max,
    t_dead_2_max=t_dead_2_max
)

if (np.any(np.isnan(dab_calc.calc_modulation.phi)) or np.any(np.isnan(dab_calc.calc_modulation.tau1)) \
        or np.any(np.isnan(dab_calc.calc_modulation.tau2))):
    raise ValueError("not alls points are possible")

if dab_calc.calc_dead_time is None:
    raise ValueError("Incomplete calculation, as dead time is missing.")

# check if the minimum dead time in all operating points is less equal than the maximum allowed dead time
dead_time_1_less_maximum = np.less_equal(dab_calc.calc_dead_time.t_dead_1, dab_calc.input_config.t_dead_1_max)
dead_time_2_less_maximum = np.less_equal(dab_calc.calc_dead_time.t_dead_2, dab_calc.input_config.t_dead_2_max)

if not np.all(dead_time_1_less_maximum):
    raise ValueError(f"Needed dead time of bridge 1 exceeds maximum dead time of {dab_calc.input_config.t_dead_1_max}.")
if not np.all(dead_time_2_less_maximum):
    raise ValueError(f"Needed dead time of bridge 2 exceeds maximum dead time of {dab_calc.input_config.t_dead_2_max}.")

# Calculate the cost function.
i_cost_matrix = dab_calc.calc_currents.i_hf_1_rms ** 2 + dab_calc.calc_currents.i_hf_2_rms ** 2
# consider weighting
i_cost_matrix_weighted = i_cost_matrix

# Mean for not-NaN values, as there will be too many NaN results.
i_cost = np.mean(i_cost_matrix_weighted[~np.isnan(i_cost_matrix_weighted)])

# return zvs coverage based on calculation
# return dab_calc.calc_modulation.mask_zvs_coverage * 100, i_cos
print(dab_calc.calc_dead_time.zvs_coverage * 100, i_cost)

HandleDabDto.save(dab_dto=dab_calc, directory=optimization_directory, name="trial", timestamp=False)