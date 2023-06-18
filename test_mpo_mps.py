import numpy as np
import Gates as gates
import MPS as mps
import MPO as mpo
import Circuits as circ
import matplotlib.pyplot as plt
import numpy_helpers as nph
from qm_shors_algorithm import *
import time
import copy
import pickle
#from to_pickle import *

def plot_state_exp(MPS,i,j,title="plot state amplitude"):
    r = MPS.measure_subspace(i,j)
    plt.figure(figsize=(15,7))
    plt.plot(r)
    plt.title(title)
    plt.show()
    plt.savefig(title+".png")

def test_H_MPO():
    #create MPO from list of H
    MPOlist = [np.ones(1)]
    for i in range(10):
        MPOlist.append(gates.H.reshape(1,2,2,1))
        MPOlist.append(np.ones(1)) 
    
    MPO = mpo.MPO.create_MPO_from_tensor_array(MPOlist,xi=2**8,cutoff=1e-8)
    print(type(MPO))
    #create MPS
    MPS = mps.MPS.create_MPS_init_to_1(10,xi=2**8,cutoff=1e-8)
    MPS.apply_mpo(MPO,0)
    plot_state_exp(MPS,0,10)

def test_fourier_MPS():
    N = 10
    MPS = mps.MPS.create_MPS_init_to_1(N,xi=2**8,cutoff=1e-8)
    MPS = circ.fourier_transform_MPS(MPS,0,N)
    print(MPS.measure_subspace(0,N))
    plot_state_exp(MPS,0,10)

def test_CNOT_Gate():
    MPS = mps.MPS.create_MPS_init_to_1(10,xi=2**8,cutoff=1e-8)
    MPS.apply_1_site_gate(gates.H,0)
    #MPS.apply_2_site_gate(gates.SWAP,0)
    #MPS.apply_2_site_gate(gates.CNOT,0)
    plot_state_exp(MPS,0,10)

def test_two_H_gates():
    MPS = mps.MPS.create_MPS_init_to_1(2,xi=2**8,cutoff=1e-8)
    MPS.apply_1_site_gate(gates.H,0)
    MPS.apply_2_site_gate(gates.R(2),0)
    MPS.apply_2_site_gate(gates.SWAP,0)
    MPS.apply_1_site_gate(gates.H,0)
    plot_state_exp(MPS,0,2)

def double_fourier_transform():
    N = 10
    MPS = mps.MPS.create_MPS_init_to_1(N,xi=2**8,cutoff=1e-8)
    plot_state_exp(MPS,0,N)
    MPS = circ.fourier_transform_MPS(MPS,0,N,inv=False,rev=False)
    plot_state_exp(MPS,0,N)
    MPS.plot_bond_dims()
    MPS = circ.fourier_transform_MPS(MPS,0,N,inv=True,rev=False)
    MPS.plot_bond_dims()
    print(MPS.measure_subspace(0,N))
    plot_state_exp(MPS,0,N)


def test_create_MPS_init_to_N(N,L):
    MPS = mps.MPS.create_MPS_init_to_N(N,L,xi=2**8,cutoff=1e-8)
    plot_state_exp(MPS,0,L)

def test_apply_mpo(L,N):
    MPS = mps.MPS.create_MPS_init_to_N(4,L,xi=2**8,cutoff=1e-8)
    #MPS.print_all_shapes()
    for i in range(N):
        U = nph.get_random_unitary(2**L)
        U = U.reshape((2,)*2*L)
        MPO = mpo.MPO.create_MPO_from_tensor(U,xi=2**8,cutoff=1e-8)
        MPO.plot_bond_dims()
        MPO.plot_schmidt_values()
        MPS.apply_mpo(MPO,0)
        MPS.plot_bond_dims()
        MPS.plot_schmidt_values()
        MPS.print_all_shapes()


def test_U_MPO(x,k,N):
    L = int(np.ceil(np.log2(N)))
    MPS = mps.MPS.create_MPS_init_to_N(k,L,xi=2**8,cutoff=1e-8)
    U = gates.x_mod_N(N,x)
    MPO = mpo.MPO.create_MPO_from_tensor(U,xi=2**8,cutoff=1e-8)
    MPS.apply_mpo(MPO,0)
    plot_state_exp(MPS,0,L)

def test_CU_MPO(x,k,N):
    L = int(np.ceil(np.log2(N)))+1
    MPS = mps.MPS.create_MPS_init_to_N(k,L,xi=2**8,cutoff=1e-8)
    CU_gates = gates.cx_pow_2k_mod_N(N,x,5)
    MPS.apply_1_site_gate(gates.H,0)
    MPO = mpo.MPO.create_MPO_from_tensor(CU_gates[0],xi=2**8,cutoff=1e-8)
    MPS.apply_mpo_regularily(MPO,0)
    plot_state_exp(MPS,1,L)

def test_sampling():
    N = 15
    x = 7
    MPS, len_a, MPO = get_shor_mps_mpo(x,N,2**8)
    patch = MPS.measure_subspace(0,len_a)
    sample_number = 1000
    samples = MPS.sample_range(0,len_a,sample_number)
    #count occurences of each number
    counts = [0]*2**len_a
    for i in range(sample_number):
        counts[nph.binary_array_to_number(samples[i,:])] += 1
    plt.figure(figsize=(17,5))
    plt.plot(patch)
    plt.plot(counts/np.sum(counts))
    plt.show()

#test_CU_MPO(7,1,15)

def test_merge_mpo(x,k,N,iteration):
    L = int(np.ceil(np.log2(N)))+1
    mps = MPS.create_MPS_init_to_N(k,L,xi=2**8,cutoff=1e-8)
    mps.apply_1_site_gate(gates.NOT,0)
    merge_mpos = gates.cx_pow_2k_mod_N_mpo(N,x,iteration,xi=2**8,cutoff=1e-8)
    gate_mpos = get_mpos_from_gates_directly(x,N,iteration)
    for i in range(1,iteration):
        merge_mpos[i].plot_bond_dims(title="merge mpo bond dims"+str(i))
        gate_mpos[i].plot_bond_dims(title="gate mpo bond dims"+str(i))
        merge_mpos[i].plot_schmidt_values(title="merge mpo schmidt values"+str(i))
        gate_mpos[i].plot_schmidt_values(title="gate mpo schmidt values"+str(i))
        mpscopy = copy.deepcopy(mps)
        mpscopy.apply_mpo_regularily(merge_mpos[i],0)
        plot_state_exp(mpscopy,1,L,title="merge mpo state"+str(i))
        mpscopy = copy.deepcopy(mps)
        mpscopy.apply_mpo_regularily(gate_mpos[i],0)
        plot_state_exp(mpscopy,1,L,title="gate mpo state"+str(i))
    

def get_mpos_from_gates_directly(x,N,iteration,xi=2**8,cutoff=1e-8):
    cu_gates = gates.cx_pow_2k_mod_N(N,x,iteration)
    mpos = []
    for i in range(iteration):
        mpos.append(MPO.create_MPO_from_tensor(cu_gates[i],xi,cutoff))
    return mpos

def test_merge_mpo_H():
    MPS = mps.MPS.create_MPS_init_to_N(0,10,xi=2**8,cutoff=1e-8)
    MPOlist = [np.ones(1)]
    for i in range(10):
        MPOlist.append(gates.H.reshape(1,2,2,1))
        MPOlist.append(np.ones(1)) 
    
    MPO = mpo.MPO.create_MPO_from_tensor_array(MPOlist,xi=2**8,cutoff=1e-8)
    MPO.merge_mpo_zip_up(MPO,0)
    MPS.apply_1_site_gate(gates.H,0)
    MPS.apply_mpo_regularily(MPO,0)
    plot_state_exp(MPS,0,10)

def unpickle_test():
    with open("./pickles/to_pickle119_45.pkl","rb") as f:
        to_pickle = pickle.load(f)
    to_pickle.MPO_mpo.plot_schmidt_values()
    p_suc_fat = to_pickle.p_success_fat
    #print(p_suc_fat)
    plt.figure(figsize=(17,5))
    plt.plot(p_suc_fat[:,0],p_suc_fat[:,1])
    plt.savefig("p_suc_fat119_45.png")

    p_suc_mpo = to_pickle.p_success_mpo
    #print(p_suc_mpo)
    plt.figure(figsize=(17,5))
    plt.plot(p_suc_mpo[:,0],p_suc_mpo[:,1])
    plt.savefig("p_suc_mpo119_45.png")

    len_a = int(2*np.ceil(np.log2(to_pickle.N)))
    psi = to_pickle.MPS_mpo.measure_subspace(0,len_a)
    plt.figure(figsize=(17,5))
    plt.plot(psi)
    plt.savefig("psi_mpo119_45.png")
    
    psi = to_pickle.MPS_fat.measure_subspace(0,len_a)
    plt.figure(figsize=(17,5))
    plt.plot(psi)
    plt.savefig("psi_fat119_45.png")

def test_fourier_MPO():
    N = 8
    MPS = mps.MPS.create_MPS_init_to_N(1,N,xi=2**8,cutoff=1e-8)
    MPO = circ.get_fourier_transform_mpo(N,xi=2**8)
    MPS.apply_mpo_regularily(MPO,0)
    #MPS = circ.fourier_transform_MPS(MPS,0,N)
    plot_state_exp(MPS,0,N)

def test_fourier_phase():
    L=4
    mpo = circ.get_identity_mpo(L)
    fourier_mpo = circ.get_fourier_transform_mpo(L,xi=2**13,inv=True)
    mpo.merge_mpo_zip_up(fourier_mpo,0)
    mpo = circ.reverse_bit_order_mpo(mpo)
    #ten = mpo.get_contracted_tensor_in_readable_form()
    #print(np.angle(ten,deg=True))
    mps = MPS.create_MPS_init_to_N(1,4,xi=2**13)
    mps.apply_mpo_zip_up_2(mpo,0)
    a = mps.get_contracted_tensor(0,L).reshape(2**L)
    print(np.abs(a),np.angle(a,deg=True))



#test_H_MPO()
#test_fourier_MPS()
#test_CNOT_Gate()
#test_two_H_gates()
#double_fourier_transform()
#test_create_MPS_init_to_N(5,5)
#test_apply_mpo(5,2)
#test_U_MPO(7,3,15)
#test_CU_MPO(7,1,15)
#test_sampling()
#test_merge_mpo(7,1,11,5)
#test_merge_mpo_H()
#unpickle_test()
#test_fourier_MPO()


    

