import pickle
import dct.topology.dab.dab_datasets_dtos as d_dtos
import dct.topology.dab.dab_datasets as d_sets
from dct.topology.dab.dab_circuit_topology import DabCircuitOptimization

combination_id = "0.pkl"
circuit_source_id_filepath = "/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/workspace/2026-02-26_design_vijay_meshgrid_dead_time/01_circuit/VijayCircuit/filtered_results/0.pkl"

with open(circuit_source_id_filepath, 'rb') as pickle_file_data:
    combination_dto: d_dtos.DabCircuitDTO = pickle.load(pickle_file_data)
(DabCircuitOptimization.generate_operating_point_table(combination_dto, "/home/nikolasf/Downloads/results", combination_id))