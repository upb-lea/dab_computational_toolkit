import dct



#
loaded_dto = dct.load_dab_specification('initial')
loaded_dto = dct.HandleDabDto.add_gecko_simulation_results(loaded_dto)
dct.HandleDabDto.save(loaded_dto, name='initial_with_simulation_results', comment='', directory=None, timestamp=False)


#loaded_dto = dct.HandleDabDto.load_from_file('initial_with_simulation_results.npz')

#dct.plot_gecko_simulation_results(loaded_dto, simulation_name='döner', comment='comment', directory=None, show_plot=True)


# dct.plot_calculation_results(loaded_dto)