import numpy as np

# The dict keys this modulation will return
MOD_KEYS = ['phi', 'tau1', 'tau2', 'mask_zvs', 'mask_Im2', 'mask_IIm2',
            'mask_IIIm1', 'zvs_coverage', 'zvs_coverage_notnan', 'mask_m2', 'mask_m1n', 'mask_m1p', 'q_ab_req1', 'q_ab_req2']



# original felix algorithm
# MOD_KEYS = ['phi', 'tau1', 'tau2', 'mask_zvs', 'mask_Im2', 'mask_IIm2',
#             'mask_IIIm1', 'mask_zvs_coverage', 'mask_zvs_coverage_notnan', 'mask_m1n', 'mask_m1p', 'q_ab_req1', 'q_ab_req2']


def calc_single_set_of_control_parameters(n: np.float64, l_s: np.float64, l_c_b1: np.float64, l_c_b2_: np.float64,
                     omega_s: float, q_ab_req_b1: float, q_ab_req_b2: float,
                     v_b1: float, v_b2_: float, i_b1: float):
    # print(f"{n=}")
    # print(f"{l_s=}")
    # print(f"{l_c_b1=}")
    # print(f"{l_c_b2_=}")
    # print(f"{omega_s=}")
    # print(f"{q_ab_req_b1=}")
    # print(f"{q_ab_req_b2=}")
    # print(f"{v_b1=}")
    # print(f"{v_b2_=}")
    # print(f"{i_b1=}")


    e1 = v_b2_ * q_ab_req_b2 * omega_s

    e2 = n * v_b1 * np.pi * i_b1

    e3 = n * (v_b2_ * (l_c_b2_ + l_s) - v_b1 * l_c_b2_)

    e4 = 2 * n * np.sqrt(q_ab_req_b1 * l_s * np.power(omega_s, 2) * v_b1 * l_c_b1 * (l_c_b1 + l_s))

    e5 = l_s * l_c_b2_ * omega_s * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    # calculate interval I
    tau_1_rad = (np.sqrt(2) * (l_c_b1 * np.sqrt(v_b2_ * e3 * e5) + e4 * e3 * 1 / n)) / (v_b1 * e3 * (l_c_b1 + l_s))
    tau_2_rad = np.sqrt((2 * e5) / (v_b2_ * e3))
    phi_rad = (tau_2_rad - tau_1_rad) / 2 + (i_b1 * omega_s * l_s * np.pi) / (tau_2_rad * v_b2_)

    if (phi_rad <= 0) and (tau_1_rad <= np.pi):
        return phi_rad, tau_1_rad, tau_2_rad, True, False, False, True
    else:
        if (phi_rad <= 0) and (tau_1_rad > np.pi):
            return np.nan, np.nan, np.nan, False, False, False, False
        else:
            # phi > 0
            # calculate interval II
            tau_1_rad = (np.sqrt(2) * (e5 + omega_s * l_s * l_c_b2_ * e2 * (v_b2_ / v_b1 * (l_s / l_c_b2_ + 1) - 1))) / (np.sqrt(v_b2_ * e3 * e5))
            tau_2_rad = np.sqrt((2 * e5) / (v_b2_ * e3))
            phi_rad = np.full_like(v_b1, 0)
            if tau_1_rad <= np.pi:
                return phi_rad, tau_1_rad, tau_2_rad, False, True, False, True
            else:
                # calculate interval III
                tau_1_rad = np.full_like(v_b1, np.pi)
                tau_2_rad = np.sqrt((2 * e5) / (v_b2_ * e3))
                sqrt_part = (- np.power((tau_2_rad - np.pi), 2) + tau_1_rad * (2 * np.pi - tau_1_rad)) / 4 - (i_b1 * omega_s * l_s * np.pi) / v_b2_
                phi_rad = (- tau_1_rad + tau_2_rad + np.pi) / 2 - np.sqrt(sqrt_part)
                if phi_rad == np.nan:
                    print("Nan beim ersten")
                if tau_2_rad <= np.pi:
                    return phi_rad, tau_1_rad, tau_2_rad, False, False, True, True
                else:
                    # tau2 = pi, recalculate phi
                    tau2_ = np.pi
                    phi_ = (- tau_1_rad + tau2_ + np.pi) / 2 - np.sqrt(
                        (- np.power((tau2_ - np.pi), 2) + tau_1_rad * (2 * np.pi - tau_1_rad)) / 4 - (i_b1 * omega_s * l_s * np.pi) / v_b2_)
                    if phi_ == np.nan:
                        print("Nan beim zweiten")
                    return phi_, tau_1_rad, tau2_, False, False, True, True


def _integrate_c_oss(coss: np.ndarray, voltage: np.ndarray) -> np.ndarray:
    """
    Integrate Coss for each voltage from 0 to V_max.

    :param coss: MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
    :return: Qoss(Vds) as one row of data and index = Vds.
    """
    # Integrate from 0 to v
    def integrate(v):
        v_interp = np.arange(v + 1)
        coss_v = np.interp(v_interp, np.arange(coss.shape[0]), coss)
        return np.trapezoid(coss_v)

    coss_int = np.vectorize(integrate)
    # get a qoss vector that has the resolution 1V from 0 to V_max
    v_vec = np.arange(coss.shape[0])
    # get a qoss vector that fits the mesh_V scale
    # v_vec = np.linspace(V_min, V_max, int(V_step))
    qoss = coss_int(v_vec)

    # Calculate a qoss mesh that is like the V mesh
    # Each element in V gets its q(v) value
    def meshing_q(v):
        return np.interp(v, v_vec, qoss)

    q_meshgrid = np.vectorize(meshing_q)
    qoss_mesh: np.ndarray = q_meshgrid(voltage)

    return qoss_mesh


def calc_modulation_params_serial(n: np.float64, ls: np.float64, lc1: np.float64, lc2: np.float64, fs: np.ndarray | int | float,
                           c_oss_1: np.ndarray, c_oss_2: np.ndarray,
                           v1: np.ndarray, v2: np.ndarray, power: np.ndarray) -> dict:

    # Create empty meshes
    phi_result = np.full_like(v1, np.nan)
    tau1_result = np.full_like(v1, np.nan)
    tau2_result = np.full_like(v1, np.nan)
    zvs = np.full_like(v1, np.nan)

    # Calculate all required values
    # Transform Lc2 to side 1
    Lc2_ = lc2 * n ** 2
    # Transform V2 to side 1
    V2_ = v2 * n
    # For negative P we have to recalculate phi at the end
    _negative_power_mask = np.less(power, 0)
    _positive_power_mask = np.less_equal(0, power)
    I1 = np.abs(power) / v1
    # Convert fs into omega_s
    ws = 2 * np.pi * fs
    # Calculate required Q for each voltage
    # FIXME Check if factor 2 is right here!
    Q_AB_req1 = _integrate_c_oss(c_oss_1 * 2, v1)
    Q_AB_req2 = _integrate_c_oss(c_oss_2 * 2, v2)

    m2i1_result = np.full_like(v1, False)
    m2i2_result = np.full_like(v1, False)
    m1i3_result = np.full_like(v1, False)
    zvs_result = np.full_like(v1, False)

    for vec_vvp in np.ndindex(v1.shape):
        phi_rad, tau_1_rad, tau_2_rad, m2i1, m2i2, m1i3, zvs = calc_single_set_of_control_parameters(n, ls, lc1, Lc2_,
                                                  ws, Q_AB_req1[vec_vvp], Q_AB_req2[vec_vvp],
                                                  v1[vec_vvp], V2_[vec_vvp], I1[vec_vvp])
        phi_result[vec_vvp] = phi_rad
        tau1_result[vec_vvp] = tau_1_rad
        tau2_result[vec_vvp] = tau_2_rad
        m2i1_result[vec_vvp] = m2i1
        m2i2_result[vec_vvp] = m2i2
        m1i3_result[vec_vvp] = m1i3
        zvs_result[vec_vvp] = zvs


    # Recalculate phi for negative power
    _negative_power_mask = np.less(power, 0)
    phi_nP = -(tau1_result + phi_result - tau2_result)
    phi_result[_negative_power_mask] = phi_nP[_negative_power_mask]
    # Initialize return dict
    da_mod_results: dict[str, np.ndarray | float | None] = dict()
    # Save the results in the dict
    # Convert phi because the math from the paper uses Middle-Pulse alignment, but we use First-Falling-Edge alignment!
    da_mod_results[MOD_KEYS[0]] = phi_result
    da_mod_results[MOD_KEYS[1]] = tau1_result
    da_mod_results[MOD_KEYS[2]] = tau2_result
    da_mod_results[MOD_KEYS[3]] = zvs_result
    da_mod_results[MOD_KEYS[4]] = m2i1_result
    da_mod_results[MOD_KEYS[5]] = m2i2_result
    da_mod_results[MOD_KEYS[6]] = m1i3_result

    # ZVS coverage based on calculation: Percentage ZVS based on all points (full operating range)
    da_mod_results[MOD_KEYS[7]] = np.count_nonzero(zvs_result) / np.size(zvs_result)
    # ZVS coverage based on calculation: Percentage ZVS based on all points where the converter can be operated (not full operating range)
    da_mod_results[MOD_KEYS[8]] = np.count_nonzero(zvs_result[~np.isnan(tau1_result)]) / np.size(zvs_result[~np.isnan(tau1_result)])

    # low / medium power mode 1
    print(f"{m2i1_result=}")
    print(f"{m2i2_result=}")
    print(f"{m1i3_result=}")
    print(f"{phi_result=}")
    print(f"{tau1_result=}")
    print(f"{tau2_result=}")
    print(f"mask m2 = {m2i1_result.astype(int) | m2i2_result.astype(int)}")
    print(f"mask m1n = {_negative_power_mask.astype(int) & m1i3_result.astype(int)}")
    print(f"mask m1p = {np.bitwise_and(_positive_power_mask.astype(int), m1i3_result.astype(int))}")


    # 'mask_m2', 'mask_m1n', 'mask_m1p'
    # mask m2
    da_mod_results[MOD_KEYS[9]] = np.bitwise_or(m2i1_result.astype(int), m2i2_result.astype(int))
    # negative high power mode 1-
    da_mod_results[MOD_KEYS[10]] = np.bitwise_and(_negative_power_mask.astype(int), m1i3_result.astype(int))
    # positive high power mode 1+
    da_mod_results[MOD_KEYS[11]] = np.bitwise_and(_positive_power_mask.astype(int), m1i3_result.astype(int))

    # add required charge
    da_mod_results[MOD_KEYS[12]] = Q_AB_req1
    da_mod_results[MOD_KEYS[13]] = Q_AB_req2

    return da_mod_results
