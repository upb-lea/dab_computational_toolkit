# plot and compare waveforms

import pysignalscope as pss

from dct.topology.dab.dab_datasets import HandleDabDto
from dct.topology.dab.dab_functions_waveforms import full_time_waveforms_from_angles_currents
from dct.constant_path import CIRCUIT_WAVEFORMS_FOLDER
import numpy as np

DTO_FILEPATH = "/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/workspace/2026-02-28_dead_time_objective"
DTO_FILE = "2706.pkl"
PROJECT_NAME = 'DabCircuitConf'
CSV_MEASUREMENT_FILEPATH = "/home/nikolasf/Downloads/2026-02-13_tektronix_mso58"
# MEASUREMENT_NAME = "0.5kw"
#MEASUREMENT_NAME = "1kw"
MEASUREMENT_NAME = "1.5kw"
OPERATING_POINT_NUMBER = 8
TIME_SHIFT_GECKO = -25e-6
TIME_SHIFT_SCOPE = 0 # 2.5e-6

IS_GECKO = True
IS_CALCULATION = True
IS_MEASUREMENT = False

# Read curves from scope csv file
if IS_MEASUREMENT:
    [scope_i_l1] = pss.Scope.from_tektronix_mso58_multichannel(f'{CSV_MEASUREMENT_FILEPATH}/{MEASUREMENT_NAME}_i_lc1.csv')
    [scope_i_ls] = pss.Scope.from_tektronix_mso58_multichannel(f'{CSV_MEASUREMENT_FILEPATH}/{MEASUREMENT_NAME}_i_ls.csv')
    [scope_i_hf2] = pss.Scope.from_tektronix_mso58_multichannel(f'{CSV_MEASUREMENT_FILEPATH}/{MEASUREMENT_NAME}_i_lc2.csv')
    [scope_v1] = pss.Scope.from_tektronix_mso58_multichannel(f'{CSV_MEASUREMENT_FILEPATH}/{MEASUREMENT_NAME}_v1.csv')
    [scope_v2] = pss.Scope.from_tektronix_mso58_multichannel(f'{CSV_MEASUREMENT_FILEPATH}/{MEASUREMENT_NAME}_v2.csv')

    scope_i_ls = pss.Scope.modify(scope_i_ls, time_shift=-TIME_SHIFT_SCOPE)
    scope_i_l1 = pss.Scope.modify(scope_i_l1, time_shift=-TIME_SHIFT_SCOPE)
    scope_v1 = pss.Scope.modify(scope_v1, time_shift=-TIME_SHIFT_SCOPE)
    scope_v2 = pss.Scope.modify(scope_v2, time_shift=-TIME_SHIFT_SCOPE)

# extract gecko waveforms from dto
dto = HandleDabDto.load_from_file(f"{DTO_FILEPATH}/01_circuit/{PROJECT_NAME}/{CIRCUIT_WAVEFORMS_FOLDER}/{DTO_FILE}")


gecko_channel_time = dto.gecko_waveforms.time
gecko_channel_time = gecko_channel_time + TIME_SHIFT_GECKO
for count, vec_vvp in enumerate(np.ndindex(dto.calc_modulation.phi.shape)):
    #if count == OPERATING_POINT_NUMBER:
    if True:
        # get gecko simulation results
        gecko_i_ls = pss.Scope.from_numpy(np.array([gecko_channel_time, dto.gecko_waveforms.i_Ls[vec_vvp]]), mode="time")
        gecko_i_l1 = pss.Scope.from_numpy(np.array([gecko_channel_time, dto.gecko_waveforms.i_Lc1[vec_vvp]]), mode="time")
        gecko_i_hf2 = pss.Scope.from_numpy(np.array([gecko_channel_time, dto.gecko_waveforms.i_HF2[vec_vvp]]), mode="time")
        gecko_v_1 = pss.Scope.from_numpy(np.array([gecko_channel_time, dto.gecko_waveforms.v1[vec_vvp]]), mode="time")
        gecko_v_2 = pss.Scope.from_numpy(np.array([gecko_channel_time, dto.gecko_waveforms.v2[vec_vvp]]), mode="time")

        # get calculated simulation results
        angles_transposed = np.transpose(dto.calc_currents.angles_rad_sorted, (1,2,3,0))[vec_vvp]
        current_ils_transposed = np.transpose(dto.calc_currents.i_l_s_sorted, (1,2,3,0))[vec_vvp]
        current_il1_transposed = np.transpose(dto.calc_currents.i_l_1_sorted, (1,2,3,0))[vec_vvp]
        current_ihf2_transposed = np.transpose(dto.calc_currents.i_hf_2_sorted, (1,2,3,0))[vec_vvp]


        [time, calc_ils_numpy] = np.array(full_time_waveforms_from_angles_currents(dto.input_config.fs, angles_transposed, current_ils_transposed))
        [time, calc_il1_numpy] = np.array(full_time_waveforms_from_angles_currents(dto.input_config.fs, angles_transposed, current_il1_transposed))
        [time, calc_ihf2_numpy] = np.array(full_time_waveforms_from_angles_currents(dto.input_config.fs, angles_transposed, current_ihf2_transposed))
        time, index = np.unique(time, return_index=True)
        calc_ils_numpy =calc_ils_numpy[index]
        calc_il1_numpy = calc_il1_numpy[index]
        calc_ihf2_numpy = calc_ihf2_numpy[index]
        calc_i_ls = pss.Scope.from_numpy(np.array([time, calc_ils_numpy]), mode='time')
        calc_i_l1 = pss.Scope.from_numpy(np.array([time, calc_il1_numpy]), mode='time')
        calc_i_hf2 = pss.Scope.from_numpy(np.array([time, calc_ihf2_numpy]), mode='time')


        # print gecko values
        print(f"{dto.gecko_results.p_dc1[vec_vvp]=}")
        print(f"{dto.gecko_results.p_dc2[vec_vvp]=}")
        eta = dto.gecko_results.p_dc2[vec_vvp] / dto.gecko_results.p_dc1[vec_vvp]
        print(f"{eta=}")
        print(f"{dto.gecko_results.S11_p_sw[vec_vvp]=}")
        print(f"{dto.gecko_results.S11_p_cond[vec_vvp]=}")
        print(f"{dto.gecko_results.S23_p_sw[vec_vvp]=}")
        print(f"{dto.gecko_results.S23_p_cond[vec_vvp]=}")
        # print(f"{dto.gecko_results.zvs_coverage1[vec_vvp]=}")
        # print(f"{dto.gecko_results.zvs_coverage2[vec_vvp]=}")

        if IS_CALCULATION & IS_GECKO & IS_MEASUREMENT:
            fig1 = pss.Scope.plot_channels([gecko_i_ls, scope_i_ls, calc_i_ls], [gecko_i_l1, scope_i_l1, calc_i_l1], [gecko_i_hf2, calc_i_hf2], [gecko_v_1, scope_v1], [gecko_v_2, scope_v2], timebase='us')
        if IS_CALCULATION & IS_GECKO:
            fig1 = pss.Scope.plot_channels([calc_i_ls, gecko_i_ls], [calc_i_l1, gecko_i_l1], [gecko_v_1], [gecko_v_2], timebase='us')

# Plot channels, save as pdf
# fig1 = pss.Scope.plot_channels([scope_i_ls, scope_i_l1], [scope_v1, scope_v2], timebase='us')
# pss.save_figure(fig1, 'figure.pdf')
