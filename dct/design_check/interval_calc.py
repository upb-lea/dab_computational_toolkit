import numpy as np
from dct.debug_tools import warning

# The dict keys this modulation will return
MOD_KEYS = ['phi', 'tau1', 'tau2', 'mask_zvs', 'mask_Im2', 'mask_IIm2',
            'mask_IIIm1', 'zvs_coverage', 'zvs_coverage_notnan']


def calc_modulation(n, Ls, Lc1, Lc2, fs: np.ndarray | int | float, Coss1: np.ndarray, Coss2: np.ndarray,
                    V1: np.ndarray, V2: np.ndarray, P: np.ndarray) -> dict:
    """
    OptZVS (Optimal ZVS) Modulation calculation, which will return phi, tau1 and tau2

    :param n: Transformer turns ratio n1/n2.
    :param Ls: DAB converter series inductance. (Must not be zero!)
    :param Lc1: Side 1 commutation inductance. Use np.inf it not present.
    :param Lc2: Side 2 commutation inductance. Use np.inf it not present. (Must not be zero!)
    :param fs: Switching frequency, can be a fixed value or a meshgrid with same shape as the other meshes.
    :param Coss1: Side 1 MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
    :param Coss2: Side 2 MOSFET Coss(Vds) curve from Vds=0V to >= V2_max. Just one row with Coss data and index = Vds.
    :param V1: Input voltage meshgrid (voltage on side 1).
    :param V2: Output voltage meshgrid (voltage on side 2).
    :param P: DAB input power meshgrid (P=V1*I1).
    :return: dict with phi, tau1, tau2, masks (phi has First-Falling-Edge alignment!)
    """

    # Create empty meshes
    phi = np.full_like(V1, np.nan)
    tau1 = np.full_like(V1, np.nan)
    tau2 = np.full_like(V1, np.nan)
    zvs = np.full_like(V1, False)
    _Im2_mask = np.full_like(V1, False)
    _IIm2_mask = np.full_like(V1, False)
    _IIIm1_mask = np.full_like(V1, False)

    ## Precalculate all required values
    # Transform Lc2 to side 1
    Lc2_ = Lc2 * n ** 2
    # Transform V2 to side 1
    V2_ = V2 * n
    # For negative P we have to recalculate phi at the end
    _negative_power_mask = np.less(P, 0)
    I1 = np.abs(P) / V1
    # Convert fs into omega_s
    ws = 2 * np.pi * fs
    # Calculate required Q for each voltage
    # FIXME Check if factor 2 is right here!
    Q_AB_req1 = _integrate_Coss(Coss1 * 2, V1)
    Q_AB_req2 = _integrate_Coss(Coss2 * 2, V2)

    # FIXME HACK for testing V1, V2 interchangeability
    # _V1 = V1
    # V1 = V2_
    # V2_ = V1

    ## Calculate the Modulations
    # ***** Change in contrast to paper *****
    # Instead of fist checking the limits for each modulation and only calculate each mod. partly,
    # all the modulations are calculated first even for useless areas, and we decide later which part is useful.
    # This should be faster and easier.

    # Int. I (mode 2): calculate phi, tau1 and tau2
    phi_I, tau1_I, tau2_I = _calc_interval_1(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

    # Int. II (mode 2): calculate phi, tau1 and tau2
    phi_II, tau1_II, tau2_II = _calc_interval_2(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

    # Int. III (mode 1): calculate phi, tau1 and tau2
    phi_III, tau1_III, tau2_III = _calc_interval_3(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

    ## Decision Logic
    # Int. I (mode 2):
    # if phi <= 0:
    _phi_I_leq_zero_mask = np.less_equal(phi_I, 0)
    # debug('_phi_I_leq_zero_mask', _phi_I_leq_zero_mask)
    # if tau1 <= pi:
    _tau1_I_leq_pi_mask = np.less_equal(tau1_I, np.pi)
    # debug('_tau1_I_leq_pi_mask', _tau1_I_leq_pi_mask)
    # if phi > 0:
    _phi_I_g_zero_mask = np.greater(phi_I, 0)
    # debug('_phi_I_g_zero_mask', _phi_I_g_zero_mask)
    # Int. I (mode 2): use results
    phi[np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)] = phi_I[
        np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)]
    tau1[np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)] = tau1_I[
        np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)]
    tau2[np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)] = tau2_I[
        np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)]
    _Im2_mask = np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)
    # debug('_Im2_mask', _Im2_mask)
    zvs[np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)] = True

    # Int. II (mode 2):
    # if tau1 <= pi:
    _tau1_II_leq_pi_mask = np.less_equal(tau1_II, np.pi)
    # debug('_tau1_II_leq_pi_mask', _tau1_II_leq_pi_mask)
    # if tau1 > pi:
    _tau1_II_g_pi_mask = np.greater(tau1_II, np.pi)
    # debug('_tau1_II_g_pi_mask', _tau1_II_g_pi_mask)
    # Int. II (mode 2): use results
    phi[np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)] = phi_II[
        np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)]
    tau1[np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)] = tau1_II[
        np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)]
    tau2[np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)] = tau2_II[
        np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)]
    _IIm2_mask = np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)
    # debug('_IIm2_mask', _IIm2_mask)
    zvs[np.bitwise_and(_phi_I_g_zero_mask, _tau1_II_leq_pi_mask)] = True

    # Int. III (mode 1):
    # if tau2 <= pi:
    _tau2_III_leq_pi_mask = np.less_equal(tau2_III, np.pi)
    # debug('_tau2_III_leq_pi_mask', _tau2_III_leq_pi_mask)
    # Int. III (mode 1): use results
    phi[np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)] = phi_III[
        np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)]
    tau1[np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)] = tau1_III[
        np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)]
    tau2[np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)] = tau2_III[
        np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)]
    _IIIm1_mask = np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)
    # debug('_IIIm1_mask', _IIIm1_mask)
    zvs[np.bitwise_and(_tau1_II_g_pi_mask, _tau2_III_leq_pi_mask)] = True

    # Int. I (mode 2): ZVS is analytically IMPOSSIBLE!
    zvs[np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))] = False
    # debug('zvs', zvs)
    # zvs = np.bitwise_not(np.bitwise_and(_phi_leq_zero_mask, np.bitwise_not(_tau1_leq_pi_mask)))
    # debug('zvs bitwise not', zvs)

    ## Recalculate phi for negative power
    phi_nP = - (tau1 + phi - tau2)
    phi[_negative_power_mask] = phi_nP[_negative_power_mask]

    # Init return dict
    da_mod_results = dict()
    # Save the results in the dict
    # da_mod_results[MOD_KEYS[0]] = phi
    # Convert phi because the math from the paper uses Middle-Pulse alignment but we use First-Falling-Edge alignment!
    da_mod_results[MOD_KEYS[0]] = phi
    da_mod_results[MOD_KEYS[1]] = tau1
    da_mod_results[MOD_KEYS[2]] = tau2
    da_mod_results[MOD_KEYS[3]] = zvs
    da_mod_results[MOD_KEYS[4]] = _Im2_mask
    da_mod_results[MOD_KEYS[5]] = _IIm2_mask
    da_mod_results[MOD_KEYS[6]] = _IIIm1_mask

    ## ZVS coverage based on calculation
    da_mod_results[MOD_KEYS[7]] = np.count_nonzero(zvs) / np.size(zvs)
    ## ZVS coverage based on calculation
    da_mod_results[MOD_KEYS[8]] = np.count_nonzero(zvs[~np.isnan(tau1)]) / np.size(zvs[~np.isnan(tau1)])

    # debug(da_mod_results)
    return da_mod_results



def _calc_interval_1(n: float, l_s: float, l_c_b1: float, l_c_b2_: float, omega_s: np.ndarray | int | float, q_ab_req_b1: np.ndarray, q_ab_req_b2: np.ndarray,
                     v_b1: np.ndarray, v_b2_: np.ndarray, i_b1: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 2 Modulation (interval I) calculation, which will return phi_rad, tau_1_rad and tau_2_rad in rad.

    :param n: transformer ratio
    :type n: float
    :param l_s: series inductance
    :type l_s: float
    :param l_c_b1: commutation inductance L_1
    :type l_c_b1: float
    :param l_c_b2_: primary side reflected commutation inductance L_2
    :type l_c_b2_: float
    :param omega_s: omega_s = 2 * pi * f
    :type omega_s: float
    :param q_ab_req_b1: required coss charge for A/B-interval of bridge 1 semiconductors
    :type q_ab_req_b1: float
    :param q_ab_req_b2: required coss charge for A/B-interval of bridge 2 semiconductors
    :type q_ab_req_b2: float
    :param v_b1: DC voltage of bridge 1
    :type v_b1: float
    :param v_b2_: DC voltage of bridge 2
    :type v_b2_: float
    :param i_b1: current out of bridge 1 at the time of transistor turn-off
    :type i_b1: float
    """
    # Predefined Terms
    e1 = v_b2_ * q_ab_req_b2 * omega_s

    e2 = n * v_b1 * np.pi * i_b1

    # FIXME e3 gets negative for all values n*V2 < V1, why? Formula is checked against PhD.
    # TODO Maybe Ls is too small? Is that even possible? Error in Formula?
    e3 = n * (v_b2_ * (l_c_b2_ + l_s) - v_b1 * l_c_b2_)
    if np.any(np.less(e3, 0)):
        warning('Something is wrong. Formula e3 is negative and it should not!')
        warning('Please check your DAB Params, probably you must check n or iterate L, Lc1, Lc2.')
        # warning(V2_, Lc2_, V1, Ls)

    e4 = 2 * n * np.sqrt(q_ab_req_b1 * l_s * np.power(omega_s, 2) * v_b1 * l_c_b1 * (l_c_b1 + l_s))

    e5 = l_s * l_c_b2_ * omega_s * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    # Solution for interval I (mode 2)
    tau_1_rad = (np.sqrt(2) * (l_c_b1 * np.sqrt(v_b2_ * e3 * e5) + e4 * e3 * 1 / n)) / (v_b1 * e3 * (l_c_b1 + l_s))

    tau_2_rad = np.sqrt((2 * e5) / (v_b2_ * e3))

    phi_rad = (tau_2_rad - tau_1_rad) / 2 + (i_b1 * omega_s * l_s * np.pi) / (tau_2_rad * v_b2_)

    return phi_rad, tau_1_rad, tau_2_rad


def _calc_interval_2(n, l_s, l_c_b1, l_c_b2_, omega_s: np.ndarray | int | float, q_ab_req_b1: np.ndarray, q_ab_req_b2: np.ndarray,
                     v_b1: np.ndarray, v_b2_: np.ndarray, i_b1: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 2 Modulation (interval II) calculation, which will return phi_rad, tau_1_rad and tau_2_rad in rad
    """
    # Predefined Terms
    e1 = v_b2_ * q_ab_req_b2 * omega_s

    e2 = n * v_b1 * np.pi * i_b1

    e3 = n * (v_b2_ * (l_c_b2_ + l_s) - v_b1 * l_c_b2_)

    e4 = 2 * n * np.sqrt(q_ab_req_b1 * l_s * np.power(omega_s, 2) * v_b1 * l_c_b1 * (l_c_b1 + l_s))

    e5 = l_s * l_c_b2_ * omega_s * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    # Solution for interval II (mode 2)
    tau_1_rad = (np.sqrt(2) * (e5 + omega_s * l_s * l_c_b2_ * e2 * (v_b2_ / v_b1 * (l_s / l_c_b2_ + 1) - 1))) / (np.sqrt(v_b2_ * e3 * e5))

    tau_2_rad = np.sqrt((2 * e5) / (v_b2_ * e3))

    phi_rad = np.full_like(v_b1, 0)

    # debug(phi_rad, tau_1_rad, tau_2_rad)
    return phi_rad, tau_1_rad, tau_2_rad


def _calc_interval_3(n, l_s, l_c_b1, l_c_b2_, omega_s: np.ndarray | int | float, q_ab_req_b1: np.ndarray, q_ab_req_b2: np.ndarray,
                     v_b1: np.ndarray, v_b2: np.ndarray, i_b1: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 1 Modulation (interval III) calculation, which will return phi_rad, tau_1_rad and tau_2_rad
    """
    # Predefined Terms
    e1 = v_b2 * q_ab_req_b2 * omega_s

    e2 = n * v_b1 * np.pi * i_b1

    e3 = n * (v_b2 * (l_c_b2_ + l_s) - v_b1 * l_c_b2_)

    e4 = 2 * n * np.sqrt(q_ab_req_b1 * l_s * np.power(omega_s, 2) * v_b1 * l_c_b1 * (l_c_b1 + l_s))

    e5 = l_s * l_c_b2_ * omega_s * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    # Solution for interval III (mode 1)
    tau_1_rad = np.full_like(v_b1, np.pi)

    tau_2_rad = np.sqrt((2 * e5) / (v_b2 * e3))

    phi_rad = (- tau_1_rad + tau_2_rad + np.pi) / 2 - np.sqrt(
        (- np.power((tau_2_rad - np.pi), 2) + tau_1_rad * (2 * np.pi - tau_1_rad)) / 4 - (i_b1 * omega_s * l_s * np.pi) / v_b2)

    # Check if tau_2_rad > pi: Set tau_2_rad = pi and recalculate phi_rad for these points

    # if tau_2_rad > pi:
    _tau2_III_g_pi_mask = np.greater(tau_2_rad, np.pi)
    # debug('_tau2_III_g_pi_mask', _tau2_III_g_pi_mask)
    tau2_ = np.full_like(v_b1, np.pi)
    phi_ = (- tau_1_rad + tau2_ + np.pi) / 2 - np.sqrt(
        (- np.power((tau2_ - np.pi), 2) + tau_1_rad * (2 * np.pi - tau_1_rad)) / 4 - (i_b1 * omega_s * l_s * np.pi) / v_b2)
    tau_2_rad[_tau2_III_g_pi_mask] = tau2_[_tau2_III_g_pi_mask]
    phi_rad[_tau2_III_g_pi_mask] = phi_[_tau2_III_g_pi_mask]

    return phi_rad, tau_1_rad, tau_2_rad


def _integrate_Coss(coss: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Integrate Coss for each voltage from 0 to V_max
    :param coss: MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
    :return: Qoss(Vds) as one row of data and index = Vds.
    """

    # Integrate from 0 to v
    def integrate(v):
        v_interp = np.arange(v + 1)
        coss_v = np.interp(v_interp, np.arange(coss.shape[0]), coss)
        return np.trapz(coss_v)

    coss_int = np.vectorize(integrate)
    # get an qoss vector that has the resolution 1V from 0 to V_max
    v_vec = np.arange(coss.shape[0])
    # get an qoss vector that fits the mesh_V scale
    # v_vec = np.linspace(V_min, V_max, int(V_step))
    qoss = coss_int(v_vec)

    # Calculate a qoss mesh that is like the V mesh
    # Each element in V gets its q(v) value
    def meshing_q(v):
        return np.interp(v, v_vec, qoss)

    q_meshgrid = np.vectorize(meshing_q)
    qoss_mesh = q_meshgrid(V)

    return qoss_mesh

