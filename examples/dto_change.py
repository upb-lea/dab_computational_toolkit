"""Show how to work with DTOs as object structure."""
import dct

loaded_dto = dct.load_dab_specification('initial', steps_in_mesh_per_direction=3)
loaded_dto = dct.HandleDabDto.add_gecko_simulation_results(loaded_dto, get_waveforms=True)

print(f"{type(loaded_dto.gecko_waveforms.time)=}")
print(f"{type(loaded_dto.calc_modulation.mask_zvs)=}")
print(f"{type(loaded_dto.calc_currents.i_l_s_sorted)=}")
print(f"{type(loaded_dto.gecko_results.i_Ls)=}")


dct.plot_calc_waveforms(loaded_dto, compare_gecko_waveforms=True)


# dct.HandleDabDto.save(loaded_dto, name='initial_with_simulation_results', comment='', directory=None, timestamp=False)

# loaded_dto = dct.HandleDabDto.load_from_file('initial_with_simulation_results.npz')
#
# relative_error = (loaded_dto.calc_currents.i_l_s_rms - loaded_dto.gecko_results.i_Ls) / loaded_dto.calc_currents.i_l_s_rms
# print(f"{loaded_dto.calc_currents.i_l_s_rms=}")
# print(f"{loaded_dto.gecko_results.i_Ls=}")

# dct.plot_gecko_simulation_results(loaded_dto, simulation_name='döner', comment='comment', directory=None, show_plot=True)

# dct.plot_calculation_results(loaded_dto)
