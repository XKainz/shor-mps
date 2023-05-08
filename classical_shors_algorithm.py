'''classical_shors_alorithm'''
from qm_shors_algorithm import *
from shor_helpers import *
from to_pickle import *
import pickle


def success_prob_measurement_patch(patch,x,N):
    prob_success = 0
    for i in range(len(patch)):
        if patch[i] > 1/(len(patch)*2):
            r = get_r(i,len(patch),N)
            if does_r_work(x,r,N)==1:
                prob_success += patch[i]
    return prob_success

def pickle_object(N,x,xi_start):
    mps_fat, len_a = get_shor_mps_fat_ten(x,N,xi_start)
    patch = mps_fat.measure_subspace(0,len_a)
    if success_prob_measurement_patch(patch,x,N) < 1e-2:
        return 0
    max_bond_dim_fat = mps_fat.maximum_bond_dim()
    print(max_bond_dim_fat,"max_bond_dim_fat")
    p_suc = []
    ran = range(max_bond_dim_fat,2,-3)
    for i in ran:
        print(i,"current_xi")
        mps, len_a = get_shor_mps_fat_ten(x,N,i)
        patch = mps.measure_subspace(0,len_a)
        p_suc.append(success_prob_measurement_patch(patch,x,N))
    
    p_suc_fat = np.zeros((len(p_suc),2))
    p_suc_fat[:,0] = [*ran]
    p_suc_fat[:,1] = p_suc
    print(p_suc_fat,"p_suc_fat")

    
    mps_mpo, len_a, mpo_mpo = get_shor_mps_mpo(x,N,xi_start)
    max_bond_dim_mps_mpo = mps_mpo.maximum_bond_dim()
    print(max_bond_dim_mps_mpo,"max_bond_dim_mps_mpo")
    max_bond_dim_mpo_mpo = mpo_mpo.maximum_bond_dim()
    print(max_bond_dim_mpo_mpo,"max_bond_dim_mpo_mpo")
    p_suc = []
    ran = range(max(max_bond_dim_mpo_mpo,max_bond_dim_mps_mpo),4,-3)
    for i in ran:
        print(i,"current_xi")
        mps, len_a, mpo = get_shor_mps_mpo(x,N,i)
        patch = mps.measure_subspace(0,len_a)
        p_suc.append(success_prob_measurement_patch(patch,x,N))
    p_suc_mpo = np.zeros((len(p_suc),2))
    p_suc_mpo[:,0] = [*ran]
    p_suc_mpo[:,1] = p_suc
    print(p_suc_mpo,"p_suc_mpo")

    to_pickle = ToPickle(N,
                         x,
                         len_a,
                         mps_mpo,
                         mps_fat,
                         p_suc_mpo,
                         p_suc_fat,
                         mpo_mpo,
                         max_bond_dim_mpo_mpo,
                         max_bond_dim_mps_mpo,
                         max_bond_dim_fat)
    pickle.dump(to_pickle,open("./pickles/to_pickle"+str(to_pickle.N)+"_"+str(to_pickle.x)+".pkl","wb"))
    return 1

def main(i,j):
    xi_start = 2**13
    for i in range(i,j):
        if N_valid(i):
            success = 0
            while success == 0:
                x = sh.get_x_for_N(i)
                success = pickle_object(i,x,xi_start)


if __name__ == "__main__":
    main(15,1023)