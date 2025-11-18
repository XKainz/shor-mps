'''ana_bond_dim.py'''

import numpy as np
import pickle
from qm_shors_algorithm import *
from shor_helpers import *
import Gates as gates
from MPO import MPO as mpo

def get_bond_dim_information(N,x,xi):
    mps_fat, len_a = get_shor_mps_fat_ten(x,N,xi)
    patch = mps_fat.measure_subspace(0,len_a)
    if success_prob_measurement_patch(patch,x,N) < 1e-2:
        return -1
    cu_gate = gates.cx_mod_N(N,x)
    MPO = mpo.create_MPO_from_tensor(cu_gate,xi,cutoff)
    return [N,
            x,
            MPO.maximum_bond_dim(),
            MPO.maximum_entanglement(),
            mps_fat.maximum_bond_dim(),
            mps_fat.maximum_entanglement()]

def main_1(i,j):
    xi_start = 2**13
    bond_dim_array = []
    success = -1
    for i in range(i,j):
        if N_valid(i):
            bond_dim_array.append(success)
            success = -1
            while success == -1:
                x = sh.get_x_for_N(i)
                success = get_bond_dim_information(i,x,xi_start)
    pickle.dump(bond_dim_array[1:],open("./pickles/bond_dim_array_"+str(i)+"_"+str(j)+".pkl","wb"))

main_1(15,300)

