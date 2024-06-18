"""Example how to use the design_check for the DAB converter."""

import dct
import transistordatabase as tdb

print("Start of Module ZVS ...")
db = tdb.DatabaseManager()
db.set_operation_mode_json()

transistor_1 = db.load_transistor('CREE_C3M0065100J')
transistor_2 = db.load_transistor('CREE_C3M0060065J')

dab_design_config = "initial"

results = dct.calculate_and_plot_dab_results(dab_design_config, transistor_1, transistor_2)

# dct.save_to_file(results, name='trial01')

# loaded_result = dct.load_from_file('2024-06-18_18:54:17_trial01.npz')
