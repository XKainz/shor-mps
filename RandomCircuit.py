import numpy as np
import numpy.linalg as la
import numpy_helpers as nph
from MPS import MPS
from MPS import create_MPS_init_to_1
from MPO import MPO
import matplotlib.pyplot as plt
from Gates import *

N=24
xi = 2**8
cutoff = 1e-8

MPS = create_MPS_init_to_1(N,xi,cutoff)
MPS.print_all_shapes()

depth = 100 #Number of layers
for i in range(depth):
    for j in range(N):
        MPS.apply_1_site_gate(rand_1_site_U(),j)
    k = i%2
    for j in range(k,N-2-k,2):
        MPS.apply_2_site_gate(CNOT,j)
    if i%10==0:
        print(i)
        MPS.plot_bond_dims()
        MPS.plot_schmidt_values()
