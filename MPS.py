'''MPS.py'''
from SuperMPS import SuperMPS
import numpy as np
import numpy_helpers as nph
import numpy.linalg as la
import Gates as gt


class MPS(SuperMPS):
    def __init__(self,*args,xi,cutoff=1e-8):
        super().__init__(*args,xi=xi,cutoff=cutoff)
        if self.indices_per_node != 1:
            raise ValueError("MPS must have 1 index per node")
    
    def apply_1_site_gate(self, gate, i):
        if i < 0 or i >= self.L:
            raise ValueError("i must be in range [0,self.L)")
        if gate.shape != (2,2):
            raise ValueError("gate must be a 1x1 matrix")
        self[i] = np.tensordot(gate,self[i],axes=(1,1)).transpose((1,0,2))

    def apply_2_site_gate(self,gate,i):
        if i < 0 or i >= self.L-2:
            raise ValueError("i must be in range [0,self.L-2)")
        if gate.shape != (2,)*4:
            raise ValueError("gate must be a 2x2x2x2 matrix")
        theta = self.get_contracted_tensor(i,i+2)
        theta = np.einsum('komn,imnj->ikoj',gate,theta)
        s1 = theta.shape[:2]
        s2 = theta.shape[2:]
        theta = np.reshape(theta,(int(np.prod(s1)),int(np.prod(s2))))
        u,s,v,ximin = nph.trunc_svd(theta,self.xi,self.cutoff)
        u = np.reshape(u,s1+(ximin,))
        v = np.reshape(v,(ximin,)+s2)
        u = np.einsum('k,k...->k...',1/self.get_schmidt_values(i,'l'),u)
        v = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(i+1,'r'))
        self[i] = u
        self.set_schmidt_values(i,'r',s)
        self[i+1] = v

    def sample(self,n_samples):
        BMPS = self.get_B_config()
        r = np.zeros(len(BMPS),dtype="int")
        contracted_tensor = np.ones((1,1))
        p_total = 1
        samples = np.zeros((n_samples,len(BMPS)),dtype="int")
        for j in range(n_samples):
            for i in range(len(BMPS)):
                contracted_tensor = np.tensordot(contracted_tensor[r[i],:],BMPS[i],axes=([0],[0]))
                density_matrix = np.tensordot(contracted_tensor,np.conj(contracted_tensor),axes=([-1],[-1]))/p_total
                p = np.diagonal(density_matrix)
                p = np.real_if_close(p,tol=10**4)
                r[i+1] = np.random.choice([0,1],p=p)
                p_total *= p[r[i+1]]
            samples[j,:] = r[1:]
        return samples
    
def create_MPS_init_to_1(length,xi,cutoff=1e-8):
    mps = [np.ones(1,dtype="complex"),np.array([[[0],[1]]],dtype="complex")]
    for i in range(1,length):
        mps.append(np.ones(1,dtype="complex"))
        mps.append(np.array([[[1],[0]]],dtype="complex"))
    mps.append(np.ones(1,dtype="complex"))
    return MPS(mps,xi=xi,cutoff=cutoff)
        