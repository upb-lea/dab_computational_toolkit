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
        dab_config.V2_step = 25 * 3
        dab_config.P_min = -2200
        dab_config.P_max = 2200
        dab_config.P_nom = 2000
        dab_config.P_step = 19 * 3
        dab_config.n = 2.99
        dab_config.Ls = 83e-6
        dab_config.Lm = 595e-6
        dab_config.fs = 200000
        # Assumption for tests
        dab_config.Lc1 = 611e-6
        dab_config.Lc2 = 611e-6
        return dab_config
    elif dab_configuration_name.lower() == "everts":
        dab_config = dct.DabData()
        dab_config.V1_nom = 250
        #dab.V1_min = 30
        dab_config.V1_min = 125
        dab_config.V1_max = 325
        dab_config.V1_step = math.floor((dab_config.V1_max - dab_config.V1_min) / 10 + 1)  # 10V resolution gives 21 steps
        # dab.V1_step = math.floor((dab.V1_max - dab.V1_min) / 10 + 1)
        # dab.V1_step = 1
        dab_config.V2_nom = 400
        dab_config.V2_min = 370
        dab_config.V2_max = 470
        dab_config.V2_step = math.floor((dab_config.V2_max - dab_config.V2_min) / 10 + 1)  # 5V resolution gives 25 steps
        # dab.V2_step = math.floor((dab.V2_max - dab.V2_min) / 20 + 1)
        # dab.V2_step = 4
        #dab.P_min = -3700
        dab_config.P_min = -3700
        dab_config.P_max = 3700
        dab_config.P_nom = 2000
        dab_config.P_step = math.floor((dab_config.P_max - dab_config.P_min) / 100 + 1)  # 100W resolution gives 19 steps
        # dab.P_step = math.floor((dab.P_max - dab.P_min) / 300 + 1)
        # dab.P_step = 5
        dab_config.n = 1
        dab_config.Ls = 13e-6
        dab_config.Lc1 = 62.1e-6
        dab_config.Lc2 = 62.1e-6
        dab_config.fs = 120e3
        return dab_config
    elif dab_configuration_name.lower() == "initial_reversed":
        ## Reversed DAB
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
        #dab.V2_step = 4
        dab_config.P_min = -2200
        dab_config.P_max = 2200
        dab_config.P_nom = 2000
        dab_config.P_step = 19 * 3
        #dab.P_step = 5
        dab_config.n = 1 / 2.99
        dab_config.Ls = 83e-6 * dab_config.n ** 2
        dab_config.Lm = 595e-6 * dab_config.n ** 2
        #dab.Lc1 = 25.62e-3
        #dab.Lc1 = 800e-6
        # Assumption for tests
        dab_config.Lc1 = 611e-6 * dab_config.n ** 2
        dab_config.Lc2 = 611e-6 * dab_config.n ** 2
        #dab.Lc2 = 25e-3 * dab.n ** 2
        dab_config.fs = 200000
        return dab_config
    else:
        raise ValueError(f"{dab_configuration_name} not found.")
