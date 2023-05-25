'''classical_shors_alorithm'''
from qm_shors_algorithm import *
from shor_helpers import *
from to_pickle import *
import pickle
import tim



def pickle_object(N,x,xi_start):
    mps_fat, len_a = get_shor_mps_fat_ten(x,N,xi_start)
    timer = tim.Timer()
    patch = mps_fat.measure_subspace(0,len_a)
    timer.print_since_last("patch measured in")
    if success_prob_measurement_patch(patch,x,N) < 1e-2:
        return 0
    timer.print_since_last("success prob calculated in")
    
    max_bond_dim_fat = mps_fat.maximum_bond_dim()
    print(max_bond_dim_fat,"max_bond_dim_fat")
    p_suc = []
    ran = range(max_bond_dim_fat,0,-3)
    for i in ran:
        print(i,"current_xi")
        timer.set_last_time()
        mps, len_a = get_shor_mps_fat_ten(x,N,i)
        timer.print_since_last("MPS created in")
        patch = mps.measure_subspace(0,len_a)
        timer.print_since_last("patch measured in")
        p_suc.append(success_prob_measurement_patch(patch,x,N))
        timer.print_since_last("success prob calculated in")
    
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
    ran = range(max(max_bond_dim_mpo_mpo,max_bond_dim_mps_mpo),1,-3)
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

def main_1(i,j):
    xi_start = 2**13
    for i in range(i,j):
        if N_valid(i):
            success = 0
            while success == 0:
                x = sh.get_x_for_N(i)
                success = pickle_object(i,x,xi_start)

def main_2(x,N,xi_start,mpo = True):
    if mpo:
        mps_fat, len_a, mpo_mpo = get_shor_mps_mpo(x,N,xi_start)
    else: 
        mps_fat, len_a = get_shor_mps_fat_ten(x,N,xi_start)
    timer = tim.Timer()
    patch = mps_fat.measure_subspace(0,len_a)
    timer.print_since_last("patch measured in")
    success_p = success_prob_measurement_patch(patch,x,N)
    timer.print_since_last("success prob calculated in")
    print(success_p,"success_p")
    n_samples = 1000
    samples = mps_fat.sample_range(0,len_a,n_samples)
    timer.print_since_last("samples created in")
    success_sampling = success_prob_measurement_samples(samples,x,N)
    timer.print_since_last("success prob calculated in")
    print(success_sampling,"success_sampling")

    max_bond_dim_fat = mps_fat.maximum_bond_dim()
    print(max_bond_dim_fat,"max_bond_dim_fat")



if __name__ == "__main__":
    #main_1(50,200)
    main_2(97,221,2**12,mpo=False)
    main_2(97,221,2**12,mpo=True)
    #main_2(108,667,2**12,mpo=True)
