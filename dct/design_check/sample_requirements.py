"""File contains DAB converter sample specifications."""
import dct
import math

def load_dab_specification(dab_configuration_name: str):
    """
    Load some predefined DAB specification from different lab samples or papers.

    :param dab_configuration_name: configuration name, which is from ["inital", "everts", "inital_reversed"]
    :type dab_configuration_name: str
    :return:
    """
    if dab_configuration_name.lower() == "initial":
        dab_config = dct.DabData()
        dab_config.V1_nom = 700
        dab_config.V1_min = 600
        dab_config.V1_max = 800
        dab_config.V1_step = 3
        dab_config.V2_nom = 235
        dab_config.V2_min = 175
        dab_config.V2_max = 295
        dab_config.V2_step = 5
        dab_config.P_min = -2200
        dab_config.P_max = 2200
        dab_config.P_nom = 2000
        dab_config.P_step = 5
        dab_config.n = 2.99
        dab_config.Ls = 83e-6
        dab_config.Lm = 595e-6
        dab_config.fs = 200000
        # Assumption for tests
        dab_config.Lc1 = 611e-6
        dab_config.Lc2 = 611e-6
        # Switch dead time, must be >= timestep_pre (equal is best)
        dab_config.t_dead1 = 100e-9
        dab_config.t_dead2 = 100e-9
        # Capacitance parallel to MOSFETs
        C_HB = 16e-12
        dab_config.C_HB11 = C_HB
        dab_config.C_HB12 = C_HB
        dab_config.C_HB21 = C_HB
        dab_config.C_HB22 = C_HB
        # Junction Temperature of MOSFETs
        dab_config.temp = 150
        dab_config.gen_meshes()
        return dab_config
    elif dab_configuration_name.lower() == "everts":
        dab_config = dct.DabData()
        dab_config.V1_nom = 250
        dab_config.V1_min = 125
        dab_config.V1_max = 325
        dab_config.V1_step = math.floor((dab_config.V1_max - dab_config.V1_min) / 10 + 1)  # 10V resolution gives 21 steps
        dab_config.V2_nom = 400
        dab_config.V2_min = 370
        dab_config.V2_max = 470
        dab_config.V2_step = math.floor((dab_config.V2_max - dab_config.V2_min) / 10 + 1)  # 5V resolution gives 25 steps
        dab_config.P_min = -3700
        dab_config.P_max = 3700
        dab_config.P_nom = 2000
        dab_config.P_step = math.floor((dab_config.P_max - dab_config.P_min) / 100 + 1)  # 100W resolution gives 19 steps
        dab_config.n = 1
        dab_config.Ls = 13e-6
        dab_config.Lc1 = 62.1e-6
        dab_config.Lc2 = 62.1e-6
        dab_config.fs = 120e3
        dab_config.gen_meshes()
        return dab_config
    elif dab_configuration_name.lower() == "initial_reversed":
        # Set the basic DAB Specification
        dab_config = dct.DabData()
        dab_config.V2_nom = 700
        dab_config.V2_min = 600
        dab_config.V2_max = 800
        dab_config.V2_step = 3
        dab_config.V1_nom = 235
        dab_config.V1_min = 175
        dab_config.V1_max = 295
        dab_config.V1_step = 25 * 3
        dab_config.P_min = -2200
        dab_config.P_max = 2200
        dab_config.P_nom = 2000
        dab_config.P_step = 19 * 3
        dab_config.n = 1 / 2.99
        dab_config.Ls = 83e-6 * dab_config.n ** 2
        dab_config.Lm = 595e-6 * dab_config.n ** 2
        # Assumption for tests
        dab_config.Lc1 = 611e-6 * dab_config.n ** 2
        dab_config.Lc2 = 611e-6 * dab_config.n ** 2
        dab_config.fs = 200000
        dab_config.gen_meshes()
        return dab_config
    elif dab_configuration_name.lower() == "dab_ds_default_gv8_sim":
        dab_config = dct.DabData()
        dab_config.V1_nom = 700
        dab_config.V1_min = 700
        dab_config.V1_max = 700
        dab_config.V1_step = 1
        dab_config.V2_nom = 235
        dab_config.V2_min = 235
        dab_config.V2_max = 295
        dab_config.V2_step = 7
        # dab_config.P_min = -2200
        # As requested only positive power transfer
        dab_config.P_min = 400
        dab_config.P_max = 2200
        dab_config.P_nom = 2000
        dab_config.P_step = 7
        dab_config.n = 2.99
        dab_config.Ls = 85e-6
        dab_config.Lc1 = 25.62e-3
        dab_config.Lc2 = 611e-6 / (dab_config.n ** 2)
        dab_config.fs = 200000
        # Switch dead time, must be >= timestep_pre (equal is best)
        dab_config.t_dead1 = 100e-9
        dab_config.t_dead2 = 100e-9
        # Capacitance parallel to MOSFETs
        C_HB = 16e-12
        dab_config.C_HB11 = C_HB
        dab_config.C_HB12 = C_HB
        dab_config.C_HB21 = C_HB
        dab_config.C_HB22 = C_HB
        # Junction Temperature of MOSFETs
        dab_config.temp = 150
        # Generate meshes
        dab_config.gen_meshes()
        return dab_config
    else:
        raise ValueError(f"{dab_configuration_name} not found.")
