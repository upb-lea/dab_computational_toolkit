

import optuna
from matplotlib import pyplot as plt

import pandas as pd

filepath = '/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/workspace/2026-03-02_new_parameters_fixed_half_qab/01_circuit/DabCircuitConf/DabCircuitConf.csv'
df = pd.read_csv(filepath)

print(df.columns)

plt.scatter(df["values_0"], df["values_1"], label='ideal')
#plt.scatter(df["user_attrs_zvs_coverage"], df["values_1"], label='ideal')
#plt.scatter(df["user_attrs_dead_time_zvs_coverage"], df["values_1"], label='detailed')

plt.legend()
plt.grid()
plt.show()