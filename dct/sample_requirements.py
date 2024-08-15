"""File contains DAB converter sample specifications."""
# python libraries
import math

# own libraries
from dct.datasets import DabDTO, HandleDabDto

def load_dab_specification(dab_configuration_name: str, steps_in_mesh_per_direction: int | None = None) -> DabDTO:
    """
    Load some predefined DAB specification from different lab samples or papers.

    :param dab_configuration_name: configuration name, which is from ["initial", "everts", "initial_reversed"]
    :type dab_configuration_name: str
    :param steps_in_mesh_per_direction: Steps in each mesh direction (v_1, v_2, p_out), optional, e.g. 2: 2x2x2 = 8 simulations, 4: 4x4x4 = 64 simulations
    :type steps_in_mesh_per_direction: int | None
    :return:
    """
    if dab_configuration_name.lower() == "initial":
        dab_config = HandleDabDto.init_config(
            V1_nom=700,
            V1_min=600,
            V1_max=800,
            V1_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 3,
            V2_nom=235,
            V2_min=175,
            V2_max=295,
            V2_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 3,
            P_min=0,
            P_max=2200,
            P_nom=2000,
            P_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 3,
            n=2.99,
            Ls=83e-6,
            fs=200000,
            # Assumption for tests
            Lc1=611e-6,
            Lc2=595e-6,
            c_par_1=16e-12,
            c_par_2=16e-12,
            transistor_name_1='CREE_C3M0065100J',
            transistor_name_2='CREE_C3M0060065J'
        )

        return dab_config
    elif dab_configuration_name.lower() == "everts":
        dab_config = HandleDabDto.init_config(
            V1_nom=250,
            V1_min=125,
            V1_max=325,
            V1_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else math.floor((325 - 125) / 10 + 1),  # 10V resolution
            V2_nom=400,
            V2_min=370,
            V2_max=470,
            V2_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else math.floor((470 - 370) / 10 + 1),  # 5V resolution
            P_min=-3700,
            P_max=3700,
            P_nom=2000,
            P_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else math.floor((3700 + 3700) / 100 + 1),  # 100W resolution
            n=1,
            Ls=13e-6,
            Lc1=62.1e-6,
            Lc2=62.1e-6,
            fs=120e3,
            c_par_1=16e-12,
            c_par_2=16e-12,
            transistor_name_1='CREE_C3M0065100J',
            transistor_name_2='CREE_C3M0060065J'
        )

        return dab_config
    elif dab_configuration_name.lower() == "initial_reversed":
        # Set the basic DAB Specification
        n = 1 / 2.99
        dab_config = HandleDabDto.init_config(
            V2_nom=700,
            V2_min=600,
            V2_max=800,
            V2_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 3,
            V1_nom=235,
            V1_min=175,
            V1_max=295,
            V1_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 25 * 3,
            P_min=-2200,
            P_max=2200,
            P_nom=2000,
            P_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 19 * 3,
            n=n,
            Ls=83e-6 * n ** 2,
            # Assumption for tests
            Lc1=611e-6 * n ** 2,
            Lc2=595e-6 * n ** 2 + 611e-6 * n ** 2,
            fs=200000,
            c_par_1=16e-12,
            c_par_2=16e-12,
            # Junction Temperature of MOSFETs
            transistor_name_1='CREE_C3M0065100J',
            transistor_name_2='CREE_C3M0060065J'
        )
        return dab_config
    elif dab_configuration_name.lower() == "dab_ds_default_gv8_sim":
        n = 2.99
        dab_config = HandleDabDto.init_config(
            V1_nom=700,
            V1_min=700,
            V1_max=700,
            V1_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 1,
            V2_nom=235,
            V2_min=235,
            V2_max=295,
            V2_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 7,
            P_min=400,
            P_max=2200,
            P_nom=2000,
            P_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 7,
            n=2.99,
            Ls=85e-6,
            Lc1=25.62e-3,
            Lc2=611e-6 / (n ** 2),
            fs=200000,
            # Generate meshes
            c_par_1=16e-12,
            c_par_2=16e-12,
            # Junction Temperature of MOSFETs
            transistor_name_1='CREE_C3M0065100J',
            transistor_name_2='CREE_C3M0060065J')
        return dab_config
    else:
        raise ValueError(f"{dab_configuration_name} not found.")
