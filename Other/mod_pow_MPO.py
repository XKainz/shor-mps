from MPO import MPO
from Gates import *
from shor_helpers import *
import matplotlib.pyplot as plt

xi = 2**14 
cutoff = 1e-8


ran = range(15,700,2)
max_bond_dim = [0,]*len(ran)
max_ent_bon_dim = [0,]*len(ran)
for N in ran:
    print(N)
    x = get_x_for_N(N)
    tensor = cx_mod_N(N,x)
    modMPO = MPO(tensor,2,xi=xi,cutoff=cutoff)
    max_bond_dim[ran.index(N)] = modMPO.maximum_bond_dim()
    max_ent_bon_dim[ran.index(N)] = 2**(len(tensor.shape)/2)

plt.figure(figsize=(15,7))
plt.plot(ran,max_bond_dim,label="max bond dim in MPO")
plt.plot(ran,max_ent_bon_dim,label="max entanglement bond dim")
plt.show()

N = 650
x = get_x_for_N(N)
tensor = cx_mod_N(N,x)
modMPO = MPO(tensor,2,xi=xi,cutoff=cutoff)
modMPO.plot_bond_dims()
modMPO.plot_schmidt_values()