import numpy as np
# plot and compare waveforms

import pysignalscope as pss
from dct.topology.dab import HandleDabDto
from dct.topology.dab.dab_datasets import HandleDabDto
from dct.constant_path import CIRCUIT_WAVEFORMS_FOLDER, FILTERED_RESULTS_PATH
import os

# WORKSPACE_FILEPATH = "/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/workspace/2026-03-02_new_parameters_fixed_half_qab"
#WORKSPACE_FILEPATH = "/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/workspace/2026-03-02_old_parameters_fixed_half_qab"
WORKSPACE_FILEPATH = "/home/nikolasf/Downloads/2026-03-02_test"
DTO_FILENAME = "trial"
PROJECT_NAME = "DabCircuitConf"
#PROJECT_NAME = "VijayCircuit"

# perform gecko simulation
dto = HandleDabDto.load_from_file(f"{WORKSPACE_FILEPATH}/01_circuit/{PROJECT_NAME}/{FILTERED_RESULTS_PATH}/{DTO_FILENAME}.pkl")

# dto.calc_dead_time.t_dead_1 = 100e-9
# dto.calc_dead_time.t_dead_2 = 100e-9
dto.gecko_additional_params.number_pre_sim_periods = 5
dto.gecko_additional_params.timestep_pre = 1e-9
dto.gecko_additional_params.number_sim_periods = 2
dto.gecko_additional_params.timestep = 1e-9

# dto_waveform = HandleDabDto.add_gecko_simulation_results(dto, get_waveforms=True)

# dto.input_config.Lc1 = 619e-6
# dto.input_config.Ls = 140e-6 # 116e-6
# dto.input_config.Lc2 = 639e-6 / 4.178 ** 2
# dto.input_config.n = 4.0
# dto.input_config.mesh_v2 = 205 * np.ones_like(dto.input_config.mesh_v2)
dto_waveform = HandleDabDto.add_gecko_simulation_results(dto, get_waveforms=True)

for vec_vvp in np.ndindex(dto.calc_modulation.phi.shape):
    if dto.calc_modulation.tau1[vec_vvp] != 0:
        print(f"Ideal power: {dto.input_config.mesh_p[vec_vvp]=}")
        print(f"Gecko input power: {dto.gecko_results.p_dc1[vec_vvp]=}")
        print(f"Gecko output power: {dto.gecko_results.p_dc2[vec_vvp]=}")
        if dto.input_config.mesh_p[vec_vvp] >= 0:
            print(f"Derivation in percent = {(dto.gecko_results.p_dc2[vec_vvp] - dto.input_config.mesh_p[vec_vvp]) / dto.input_config.mesh_p[vec_vvp] * 100}")
        else:
            print(f"Derivation in percent = {(dto.gecko_results.p_dc1[vec_vvp] - dto.input_config.mesh_p[vec_vvp]) / dto.input_config.mesh_p[vec_vvp] * 100}")

result_path = f"{WORKSPACE_FILEPATH}/01_circuit/{PROJECT_NAME}/{CIRCUIT_WAVEFORMS_FOLDER}"
print(f"{result_path=}")
if not os.path.exists(result_path):
    os.makedirs(result_path)
HandleDabDto.save(dto_waveform, DTO_FILENAME, result_path, timestamp=False)




