'''classical_shors_alorithm'''
from qm_shors_algorithm import *
from shor_helpers import *


def success_prob_measurement_patch(patch,x,N):
    prob_success = 0
    for i in range(len(patch)):
        if patch[i] > 1/(len(patch)*2):
            r = get_r(i,len(patch),N)
            if does_r_work(x,r,N)==1:
                prob_success += patch[i]
    return prob_success

N = 77
x = 5
print(x,"x")
xi_start = 2**8

p_suc = []
ran = range(xi_start,30,-10)
MPS, len_a  = get_shor_mps_mpo(x,N,xi_start)
for i in ran:
    MPS.set_xi(i)
    print(MPS.maximum_bond_dim(),"Maximum Bond Dimension")
    patch = MPS.measure_subspace(0,len_a)
    p_suc.append(success_prob_measurement_patch(patch,x,N))

#print(p_suc)
plt.figure(figsize=(15,7))
plt.plot([*ran],p_suc)
plt.show()

