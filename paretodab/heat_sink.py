"""Functions for the heat sink optimization."""

# python libraries

# 3rd party libraries

# own libraries
from paretodab.heat_sink_dtos import *

def calculate_r_th_copper_coin(cooling_area: float, height_pcb: float = 1.55e-3, height_pcb_heat_sink: float = 3.0e-3) -> tuple[float, float]:
    """
    Calculate the thermal resistance of the copper coin.

    Assumptions are made with some geometry factors from a real copper coin for TO263 housing.
    :param cooling_area: cooling area in m²
    :type cooling_area: float
    :param height_pcb: PCB thickness, e.g. 1.55 mm
    :type height_pcb: float
    :param height_pcb_heat_sink: Distance from PCB to heat sink in m
    :type height_pcb_heat_sink: float
    :return: r_th_copper_coin, effective_copper_coin_cooling_area
    :rtype: tuple[float, float]
    """
    factor_pcb_area_copper_coin = 1.42
    factor_bottom_area_copper_coin = 0.39
    thermal_conductivity_copper = 136  # W/(m*K)

    effective_pcb_cooling_area = cooling_area / factor_pcb_area_copper_coin
    effective_bottom_cooling_area = effective_pcb_cooling_area / factor_bottom_area_copper_coin

    r_pcb = 1 / thermal_conductivity_copper * height_pcb / effective_pcb_cooling_area
    r_bottom = 1 / thermal_conductivity_copper * height_pcb_heat_sink / effective_bottom_cooling_area

    r_copper_coin = r_pcb + r_bottom

    return r_copper_coin, effective_bottom_cooling_area


def calculate_r_th_tim(copper_coin_bot_area: float, transistor_cooling: TransistorCooling) -> float:
    """
    Calculate the thermal resistance of the thermal interface material (TIM).

    :param copper_coin_bot_area: bottom copper coin area in m²
    :type copper_coin_bot_area: float
    :param transistor_cooling: Transistor cooling DTO
    :type transistor_cooling: TransistorCooling
    :return: r_th of TIM material
    :rtype: float
    """
    r_th_tim = 1 / transistor_cooling.tim_conductivity * transistor_cooling.tim_thickness / copper_coin_bot_area

    return r_th_tim
