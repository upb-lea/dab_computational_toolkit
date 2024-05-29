"""Example how to use the design_check for the DAB converter."""

import dct
import transistordatabase as tdb

print("Start of Module ZVS ...")
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

dct.calculate_and_plot_dab_results(dab_design_config, mosfet1, mosfet2)
