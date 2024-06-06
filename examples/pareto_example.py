"""Example how to use the design_check for the DAB converter."""
# python libraries

# own libraries
import dct

# 3rd party libraries
import numpy as np
import transistordatabase as tdb
from matplotlib import pyplot as plt

db = tdb.DatabaseManager()
db.set_operation_mode_json()

mosfet1 = 'CREE_C3M0065100J'
mosfet2 = 'CREE_C3M0060065J'
transistor_1 = db.load_transistor(mosfet1)
transistor_2 = db.load_transistor(mosfet2)

dab_design_config = dct.load_dab_specification("initial")

# Generate meshes
dab_design_config.gen_meshes()

dab_design_config.import_c_oss_from_tdb(transistor_1)
dab_design_config.import_c_oss_from_tdb(transistor_2)

# calculate phi, tau_1 and tau_2
da_mod = dct.calc_modulation(dab_design_config.n,
                             dab_design_config.Ls,
                             dab_design_config.Lc1,
                             dab_design_config.Lc2,
                             dab_design_config.fs,
                             dab_design_config['coss_' + mosfet1],
                             dab_design_config['coss_' + mosfet2],
                             dab_design_config.mesh_V1,
                             dab_design_config.mesh_V2,
                             dab_design_config.mesh_P)

dab_design_config.append_result_dict(da_mod, name_pre='mod_zvs_')

i_l_s_rms, i_l_1_rms, i_l_2_rms = dct.calc_rms_currents(dab_design_config)

v1_middle = int(np.shape(dab_design_config.mesh_P)[1] / 2)
# [:, v1_middle, :]
plt.contourf(dab_design_config.mesh_P[:, v1_middle, :], dab_design_config.mesh_V2[:, v1_middle, :], i_l_s_rms[:, v1_middle, :])
plt.figure()
plt.contourf(dab_design_config.mesh_P[:, v1_middle, :], dab_design_config.mesh_V2[:, v1_middle, :], i_l_1_rms[:, v1_middle, :])
plt.figure()
plt.contourf(dab_design_config.mesh_P[:, v1_middle, :], dab_design_config.mesh_V2[:, v1_middle, :], i_l_2_rms[:, v1_middle, :])
plt.grid()
plt.xlabel('Power in W')
plt.ylabel('V_2 in V')
plt.show()
