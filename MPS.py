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
        if i < 0 or i >= self.L-1:
            raise ValueError("i must be in range [0,self.L-1)")
        if gate.shape != (2,)*4:
            raise ValueError("gate must be a 2x2x2x2 matrix")
        theta = self.get_contracted_tensor(i,i+2)
        theta = np.einsum('komn,imnj->ikoj',gate,theta)
        u,s,v = nph.trunc_svd_before_index(theta,2,xi=self.xi,cutoff=self.cutoff)
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
    
    def apply_mpo(self,mpo,i):
        if i < 0 or i > self.L-mpo.L:
            raise ValueError("i must be in range [0,self.L-1-mpo.L)")
        k1 = len(self.get_schmidt_values(i,'l'))
        C = np.identity(k1).reshape((k1,1,k1))
        Amps = self.get_all_A(i,i+mpo.L)
        Ampo = mpo.get_all_A(0,mpo.L)
        for j in range(mpo.L):
            #print(Amps[j].shape,"Amps[j] shape")
            #print(Ampo[j].shape,"Ampo[j] shape")
            #print(C.shape,"C shape")
            C = np.einsum('ijk,klm->ijlm',C,Amps[j])
            C = np.einsum('ijlm,jqlp->iqpm',C,Ampo[j])
            u,s,v = nph.trunc_svd_before_index(C,2,self.xi,self.cutoff)
            self[i+j]=np.einsum('k,k...->k...',1/self.get_schmidt_values(i+j,'l'),u)
            self.set_schmidt_values(i+j,'r',s)
            #print(self.to_MPS_index(i+j)+1,"self index +1")
            C = np.einsum('k,k...->k...',s,v)
        C = C.reshape((int(np.prod(C.shape)),))
        s = self.get_schmidt_values(i+mpo.L,'l')
        self.set_schmidt_values(i+mpo.L,'l',np.einsum('k,k->k',s,C))
        self.into_canonical_form()

    def into_canonical_form(self):
        for i in range(self.L,1,-1):
            contracted_tensor = self.get_contracted_tensor(i-2,i)
            u,s,v = nph.trunc_svd_before_index(contracted_tensor,2,xi=self.xi,cutoff=self.cutoff)
            u = np.einsum('k,k...->k...',1/self.get_schmidt_values(i-2,'l'),u)
            v = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(i-1,'r'))
            self[i-2] = u
            self.set_schmidt_values(i-1,'l',s)
            self[i-1] = v

    def measure_subspace(self,i,j):
        if i < 0 or i > self.L-1:
            raise ValueError("i must be in range [0,self.L)")
        if j <= i or j > self.L:
            raise ValueError("j must be in range [i+1,self.L+1]")
        contracted_tensor = self.get_contracted_tensor(i,j)
        s = contracted_tensor.shape
        contracted_tensor = np.reshape(contracted_tensor,(s[0],int(np.prod(s[1:-1])),s[-1]))
        r = np.einsum('ijk,ijk->j',contracted_tensor,np.conj(contracted_tensor))
        r = np.real_if_close(r,tol=10**4)
        return r
    
    @staticmethod
    def create_MPS_init_to_r(r,xi,cutoff=1e-8):
        mps = [np.ones(1)]
        for i in r:
            if i == 0:
                mps.append(np.array([[[1],[0]]]))
            elif i == 1:
                mps.append(np.array([[[0],[1]]]))
            else:
                raise ValueError("r not binary")
            mps.append(np.ones(1))
        return MPS(mps,xi=xi,cutoff=cutoff)

    @staticmethod
    def create_MPS_init_to_N(N,L,xi,cutoff=1e-8):
        if N < 0:
            raise ValueError("N must be positive")
        if np.ceil(np.log2(N)) > L:
            raise ValueError("N must be less than 2**L")
        r = nph.number_to_binary_array(N)
        zeros = [0,]*(L-len(r))
        r = zeros + r
        return MPS.create_MPS_init_to_r(r,xi,cutoff)

    @staticmethod
    def create_MPS_init_to_1(length,xi,cutoff=1e-8):
        return MPS.create_MPS_init_to_N(1,length,xi,cutoff)