import numpy as np
import Gates as gates
import MPS as mps
import MPO as mpo
import Circuits as circ
import matplotlib.pyplot as plt
import numpy_helpers as nph
from qm_shors_algorithm import *

def plot_state_exp(MPS,i,j):
    r = MPS.measure_subspace(i,j)
    plt.figure(figsize=(15,7))
    plt.plot(r)
    plt.show()

def test_H_MPO():
    #create MPO from list of H
    MPOlist = [np.ones(1)]
    for i in range(10):
        MPOlist.append(gates.H.reshape(1,2,2,1))
        MPOlist.append(np.ones(1))

    MPO = mpo.MPO(MPOlist,xi=2**8,cutoff=1e-8)

    #create MPS
    MPS = mps.create_MPS_init_to_1(10,xi=2**8,cutoff=1e-8)
    MPS.apply_mpo(MPO,0)
    plot_state_exp(MPS)

def test_fourier_MPS():
    N = 10
    MPS = mps.create_MPS_init_to_1(N,xi=2**8,cutoff=1e-8)
    MPS = circ.fourier_transform_MPS(MPS,0,N)
    print(MPS.measure_subspace(0,N))
    plot_state_exp(MPS)

def test_CNOT_Gate():
    MPS = mps.create_MPS_init_to_1(10,xi=2**8,cutoff=1e-8)
    MPS.apply_1_site_gate(gates.H,0)
    MPS.apply_2_site_gate(gates.SWAP,0)
    #MPS.apply_2_site_gate(gates.CNOT,0)
    plot_state_exp(MPS)

def test_two_H_gates():
    MPS = mps.create_MPS_init_to_1(2,xi=2**8,cutoff=1e-8)
    MPS.apply_1_site_gate(gates.H,0)
    MPS.apply_2_site_gate(gates.R(2),0)
    MPS.apply_2_site_gate(gates.SWAP,0)
    MPS.apply_1_site_gate(gates.H,0)
    plot_state_exp(MPS)

def double_fourier_transform():
    N = 10
    MPS = mps.create_MPS_init_to_1(N,xi=2**8,cutoff=1e-8)
    plot_state_exp(MPS)
    MPS = circ.fourier_transform_MPS(MPS,0,N,inv=False,rev=False)
    plot_state_exp(MPS)
    MPS.plot_bond_dims()
    MPS = circ.fourier_transform_MPS(MPS,0,N,inv=True,rev=False)
    MPS.plot_bond_dims()
    print(MPS.measure_subspace(0,N))
    plot_state_exp(MPS)


def test_create_MPS_init_to_N(N,L):
    MPS = mps.MPS.create_MPS_init_to_N(N,L,xi=2**8,cutoff=1e-8)
    plot_state_exp(MPS)

def test_apply_mpo(L,N):
    MPS = mps.MPS.create_MPS_init_to_N(4,L,xi=2**8,cutoff=1e-8)
    #MPS.print_all_shapes()
    for i in range(N):
        U = nph.get_random_unitary(2**L)
        U = U.reshape((2,)*2*L)
        MPO = mpo.MPO(U,xi=2**8,cutoff=1e-8)
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
    MPO = mpo.MPO(U,xi=2**8,cutoff=1e-8)
    MPS.apply_mpo(MPO,0)
    plot_state_exp(MPS)

def test_CU_MPO(x,k,N):
    L = int(np.ceil(np.log2(N)))+1
    MPS = mps.MPS.create_MPS_init_to_N(k,L,xi=2**8,cutoff=1e-8)
    CU_gates = gates.cx_pow_2k_mod_N(N,x,5)
    MPS.apply_1_site_gate(gates.H,0)
    MPO = mpo.MPO(CU_gates[0],xi=2**8,cutoff=1e-8)
    MPS.apply_mpo(MPO,0)
    plot_state_exp(MPS,1,L)

def test_sampling():
    N = 15
    x = 7
    MPS, len_a = get_shor_mps_mpo(x,N,2**8)
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

test_sampling()


