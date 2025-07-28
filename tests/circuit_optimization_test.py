"""Python tests for the circuit optimization."""
# python libraries
import logging

# own libraries
import dct

# 3rd party libraries
import pytest
from _pytest.logging import LogCaptureFixture
from numpy.testing import assert_array_equal

design_space = dct.CircuitParetoDesignSpace(
    f_s_min_max_list=[100_000, 200_000],
    l_s_min_max_list=[10e-6, 10e-3],
    l_1_min_max_list=[10e-6, 10e-3],
    l_2__min_max_list=[10e-6, 10e-3],
    n_min_max_list=[1, 3],
    transistor_1_name_list=['CREE_C3M0065100J', 'CREE_C3M0120100J'],
    transistor_2_name_list=['CREE_C3M0060065J', 'CREE_C3M0120065J'],
    c_par_1=1e-12,
    c_par_2=1e-12)

output_range = dct.CircuitOutputRange(
    v1_min_max_list=[690, 710],
    v2_min_max_list=[175, 295],
    p_min_max_list=[-2000, 2000])

filter = dct.CircuitFilter(
    number_filtered_designs=1,
    difference_percentage=5)

info_message_dict = {0: "Number of sampling points has been updated from 5 to 8.",
                     1: "Note: meshgrid sampling does not take user-given operating points into account",
                     2: "Auto-weight given for all other 5 operating points: 0.1",
                     3: "Auto-weight given for all other 5 operating points: 0.2",
                     4: "Auto-weight given for all other 8 operating points: 0.125"}

@pytest.mark.parametrize(
    "sampling_method, v1_additional_user_point_list, v2_additional_user_point_list, p_additional_user_point_list, "
    "additional_user_weighting_point_list, expected_exception, result_weighting, expected_message_id_list",
    [
        # user-given operating points
        (dct.SamplingEnum.latin_hypercube, [700], [200], [1000], [0.5], None, [[[0.1], [0.1], [0.1], [0.1], [0.1], [0.5]]], [2]),
        # no user-given operating points
        (dct.SamplingEnum.latin_hypercube, [], [], [], [], None, [[[0.2], [0.2], [0.2], [0.2], [0.2]]], [3]),
        # meshgrid, no user-given operating points. Internal algorithm increases sampling points from 5 to 8, so weighting is 0.125
        (dct.SamplingEnum.meshgrid, [], [], [], [], None, [[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]], [0, 1, 4]),
        # meshgrid, with user-given operating points (will be ignored). Internal algorithm increases sampling points from 5 to 8, so weighting is 0.125
        (dct.SamplingEnum.meshgrid, [700], [200], [1000], [0.5], None, [[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]], [0, 1, 4]),
        # value error expected
        (dct.SamplingEnum.latin_hypercube, [700], [200], [1000], [1.5], ValueError, None, []),
    ]
)
def test_calculate_fix_parameters(caplog: LogCaptureFixture, sampling_method: dct.SamplingEnum, v1_additional_user_point_list: list[float],
                                  v2_additional_user_point_list: list[float],
                                  p_additional_user_point_list: list[float], additional_user_weighting_point_list: list[float],
                                  expected_exception: type, result_weighting: list[float], expected_message_id_list: dict) -> None:
    """
    Unit test to check the fix parameters, especially the sampling.

    :param caplog: pytest feature to read logger messages
    :param sampling_method: sampling method
    :param v1_additional_user_point_list: user-given operating points for v1
    :param v2_additional_user_point_list: user-given operating points for v2
    :param p_additional_user_point_list: user-given operating points for power
    :param additional_user_weighting_point_list: user-given operating point weighting
    :param expected_exception: expected exception
    :param result_weighting: unit test results
    :param expected_message_id_list: expected message IDs in a list
    """
    # set logging level to warning
    caplog.set_level(logging.INFO)

    # Create the instance
    test_object: dct.CircuitOptimization = dct.CircuitOptimization()

    # set up sampling
    sampling = dct.CircuitSampling(
        sampling_method=sampling_method,
        sampling_points=5,
        sampling_random_seed=1,
        v1_additional_user_point_list=v1_additional_user_point_list,
        v2_additional_user_point_list=v2_additional_user_point_list,
        p_additional_user_point_list=p_additional_user_point_list,
        additional_user_weighting_point_list=additional_user_weighting_point_list
    )

    # set up input configuration
    dab_config = dct.CircuitParetoDabDesign(
        circuit_study_name="test",
        project_directory="",
        design_space=design_space,
        output_range=output_range,
        sampling=sampling,
        filter=filter)

    if expected_exception:
        with pytest.raises(expected_exception):
            dct.CircuitOptimization.calculate_fixed_parameters(dab_config)
    else:
        output = dct.CircuitOptimization.calculate_fixed_parameters(dab_config)
        assert_array_equal(result_weighting, output.mesh_weights)

        for info_message_id in expected_message_id_list:
            assert info_message_dict[info_message_id] in caplog.text
