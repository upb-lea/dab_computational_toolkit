"""Example how to use the design_check for the DAB converter."""
# python libraries
import os

# own libraries
import dct

# 3rd party libraries

design_space = dct.DesignSpace(
    f_s_min_max_list=[50e3, 500e3],
    l_s_min_max_list=[1e-6, 200e-6],
    l_1_min_max_list=[100e-6, 800e-3],
    l_2_min_max_list=[100e-6, 800e-3],
    n_min_max_list=[4, 7],
    transistor_1_list=['CREE_C3M0065100J'],
    transistor_2_list=['CREE_C3M0060065J'],

    # misc
    working_directory=os.curdir
)

work_area = dct.WorkArea(
    v_1_min_nom_max_list=[600, 700, 800],
    v_2_min_nom_max_list=[175, 235, 295],
    p_min_nom_max_list=[0, 2000, 2200],
    steps_per_direction=5,
)

study_name = 'a2_200khz'

# dct.Optimization.start_proceed_study(study_name, design_space, work_area, 1000)
# dct.Optimization.show_study_results(study_name, design_space)

loaded_dto = dct.Optimization.load_dab_dto_from_study(study_name, design_space=design_space, trial_number=874, work_area=work_area)
loaded_dto = dct.HandleDabDto.add_gecko_simulation_results(loaded_dto, get_waveforms=True)
dct.plot_calc_waveforms(loaded_dto, compare_gecko_waveforms=True)
