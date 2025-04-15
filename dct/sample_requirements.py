"""File contains DAB converter sample specifications."""
# python libraries
import math

import dct.datasets
# own libraries
from dct.datasets_dtos import CircuitDabDTO
from dct.datasets import HandleDabDto

def load_dab_specification(dab_configuration_name: str, steps_in_mesh_per_direction: int | None = None) -> CircuitDabDTO:
    """
    Load some predefined DAB specification from different lab samples or papers.

    :param dab_configuration_name: configuration name, which is from ["initial", "everts", "initial_reversed"]
    :type dab_configuration_name: str
    :param steps_in_mesh_per_direction: Steps in each mesh direction (v_1, v_2, p_out), optional, e.g. 2: 2x2x2 = 8 simulations, 4: 4x4x4 = 64 simulations
    :type steps_in_mesh_per_direction: int | None
    :return:
    """
    if dab_configuration_name.lower() == "initial":

        transistor_dto_1 = dct.HandleTransistorDto.tdb_to_transistor_dto('CREE_C3M0065100J')
        transistor_dto_2 = dct.HandleTransistorDto.tdb_to_transistor_dto('CREE_C3M0060065J')

        dab_config = HandleDabDto.init_config(
            name=dab_configuration_name,
            v1_nom=700,
            v1_min=600,
            v1_max=800,
            v1_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 3,
            v2_nom=235,
            v2_min=175,
            v2_max=295,
            v2_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 3,
            p_min=0,
            p_max=2200,
            p_nom=2000,
            p_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 3,
            n=2.99,
            ls=83e-6,
            fs=200000,
            # Assumption for tests
            lc1=611e-6,
            lc2=595e-6,
            c_par_1=16e-12,
            c_par_2=16e-12,
            transistor_dto_1=transistor_dto_1,
            transistor_dto_2=transistor_dto_2
        )

        return dab_config
    elif dab_configuration_name.lower() == "everts":
        transistor_dto_1 = dct.HandleTransistorDto.tdb_to_transistor_dto('CREE_C3M0065100J')
        transistor_dto_2 = dct.HandleTransistorDto.tdb_to_transistor_dto('CREE_C3M0060065J')

        dab_config = HandleDabDto.init_config(
            name=dab_configuration_name,
            v1_nom=250,
            v1_min=125,
            v1_max=325,
            v1_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else math.floor((325 - 125) / 10 + 1),  # 10V resolution
            v2_nom=400,
            v2_min=370,
            v2_max=470,
            v2_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else math.floor((470 - 370) / 10 + 1),  # 5V resolution
            p_min=-3700,
            p_max=3700,
            p_nom=2000,
            p_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else math.floor((3700 + 3700) / 100 + 1),  # 100W resolution
            n=1,
            ls=13e-6,
            lc1=62.1e-6,
            lc2=62.1e-6,
            fs=120e3,
            c_par_1=16e-12,
            c_par_2=16e-12,
            transistor_dto_1=transistor_dto_1,
            transistor_dto_2=transistor_dto_2,
        )

        return dab_config
    elif dab_configuration_name.lower() == "initial_reversed":
        # Set the basic DAB Specification
        n = 1 / 2.99
        transistor_dto_1 = dct.HandleTransistorDto.tdb_to_transistor_dto('CREE_C3M0065100J')
        transistor_dto_2 = dct.HandleTransistorDto.tdb_to_transistor_dto('CREE_C3M0060065J')

        dab_config = HandleDabDto.init_config(
            name=dab_configuration_name,
            v2_nom=700,
            v2_min=600,
            v2_max=800,
            v2_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 3,
            v1_nom=235,
            v1_min=175,
            v1_max=295,
            v1_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 25 * 3,
            p_min=-2200,
            p_max=2200,
            p_nom=2000,
            p_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 19 * 3,
            n=n,
            ls=83e-6 * n ** 2,
            # Assumption for tests
            lc1=611e-6 * n ** 2,
            lc2=595e-6 * n ** 2 + 611e-6 * n ** 2,
            fs=200000,
            c_par_1=16e-12,
            c_par_2=16e-12,
            # Junction Temperature of MOSFETs
            transistor_dto_1=transistor_dto_1,
            transistor_dto_2=transistor_dto_2
        )
        return dab_config
    elif dab_configuration_name.lower() == "dab_ds_default_gv8_sim":
        n = 2.99
        transistor_dto_1 = dct.HandleTransistorDto.tdb_to_transistor_dto('CREE_C3M0065100J')
        transistor_dto_2 = dct.HandleTransistorDto.tdb_to_transistor_dto('CREE_C3M0060065J')
        dab_config = HandleDabDto.init_config(
            name=dab_configuration_name,
            v1_nom=700,
            v1_min=700,
            v1_max=700,
            v1_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 1,
            v2_nom=235,
            v2_min=235,
            v2_max=295,
            v2_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 7,
            p_min=400,
            p_max=2200,
            p_nom=2000,
            p_step=steps_in_mesh_per_direction if isinstance(steps_in_mesh_per_direction, int) else 7,
            n=2.99,
            ls=85e-6,
            lc1=25.62e-3,
            lc2=611e-6 / (n ** 2),
            fs=200000,
            # Generate meshes
            c_par_1=16e-12,
            c_par_2=16e-12,
            # Junction Temperature of MOSFETs
            transistor_dto_1=transistor_dto_1,
            transistor_dto_2=transistor_dto_2)
        return dab_config
    else:
        raise ValueError(f"{dab_configuration_name} not found.")
