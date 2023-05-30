'''MPO.py'''

import numpy as np
import numpy_helpers as nph

from SuperMPS import SuperMPS
class MPO(SuperMPS):
    def __init__(self, *args, xi, cutoff=1e-8):
        super().__init__(*args,2, xi=xi, cutoff=cutoff)
        if self.indices_per_node != 2:
            raise ValueError("MPO must have 2 indices per node")
                
    @staticmethod
    def create_MPO_from_tensor(tensor,xi,cutoff):
        mpo_object = SuperMPS.create_SuperMPS_from_tensor(tensor,2,xi,cutoff)
        mpo_object.__class__ = MPO
        return mpo_object
    
    @staticmethod
    def create_MPO_from_tensor_array(tensor_array,xi,cutoff):
        mpo_object = SuperMPS.create_SuperMPS_from_tensor_array(tensor_array,xi,cutoff)
        mpo_object.__class__ = MPO
        return mpo_object
    
    def merge_1_site_gate(self,gate,i):
        if i < 0 or i >= self.L:
            raise ValueError("i must be in range [0,self.L)")
        if gate.shape != (2,2):
            raise ValueError("gate must be a 2x2 matrix")
        self[i] = np.einsum('ij,kjlm->kilm',gate,self[i])

    def merge_2_site_gate(self,gate,i):
        if i < 0 or i >= self.L-1:
            raise ValueError("i must be in range [0,self.L-1)")
        if gate.shape != (2,)*4:
            raise ValueError("gate must be a 2x2x2x2 tensor")
        theta = self.get_contracted_tensor(i,i+2)
        theta = np.einsum('komn,imlngj->iklogj',gate,theta)
        u,s,v = nph.trunc_svd_before_index(theta,3,xi=self.xi,cutoff=self.cutoff,norm=2**((self.L)/2))
        u = np.einsum('k,k...->k...',1/self.get_schmidt_values(i,'l'),u)
        v = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(i+1,'r'))
        self[i] = u
        self.set_schmidt_values(i,'r',s)
        self[i+1] = v
    def merge_mpo_zip_up(self,other_mpo,i):
        if i < 0 or i > self.L-other_mpo.L:
            raise ValueError("i must be in range [0,self.L-1-other_mpo.L)")
        k1 = len(self.get_schmidt_values(i,'l'))
        C = np.identity(k1).reshape((k1,1,k1))
        Bmpo1 = self.get_all_B(i,i+other_mpo.L)
        Bmpo2 = other_mpo.get_all_B(0,other_mpo.L)
        for j in range(other_mpo.L):
            C = np.einsum('ijk,knlm->ijnlm',C,Bmpo1[j])
            C = np.einsum('jqnp,ijnlm->iqlpm',Bmpo2[j],C)
            u,s,v = nph.trunc_svd_before_index(C,3,self.xi,1e-15,norm=2**((self.L)/2))
            u = np.einsum('k,k...->k...',1/self.get_schmidt_values(i+j,'l'),u)
            self[i+j]=u
            self.set_schmidt_values(i+j,'r',s)
            C = np.einsum('k,k...->k...',s,v)
        C = C.reshape((int(np.prod(C.shape)),))
        s = self.get_schmidt_values(i+other_mpo.L,'l')
        self.set_schmidt_values(i+other_mpo.L,'l',np.einsum('k,k->k',s,C))
        self.into_canonical_form()

    def merge_mpo_regularily(self,other_mpo,i):
        if i < 0 or i > self.L-other_mpo.L:
            raise ValueError("i must be in range [0,self.L-1-other_mpo.L)")
        self.set_schmidt_values(i,'l',np.tensordot(other_mpo.get_schmidt_values(0,'l'),self.get_schmidt_values(i,'l'),axes=0).reshape(-1))
        for j in range(other_mpo.L):
            nten = np.einsum('mijn,kjql->mkiqnl',other_mpo[j],self[i+j])
            shape = nten.shape
            nten = np.reshape(nten,(shape[0]*shape[1],shape[2],shape[3],shape[4]*shape[5]))
            self[i+j] = nten
            self.set_schmidt_values(i+j,'r',np.tensordot(other_mpo.get_schmidt_values(j,'r'),self.get_schmidt_values(i+j,'r'),axes=0).reshape(-1))
        Amps = self.get_all_A(i,i+other_mpo.L)
        for j in range(other_mpo.L-1):
            theta = np.einsum('kijl,lqpm->kijqpm',Amps[j],Amps[j+1])
            q,r = nph.qr_before_index(theta,3)
            Amps[j]=q
            Amps[j+1]=r
        for j in range(other_mpo.L,1,-1):
            contracted_tensor = np.einsum('kijl,lqpm->kijqpm',Amps[j-2],Amps[j-1])
            contracted_tensor = np.einsum('kijqpm,m->kijqpm',contracted_tensor,self.get_schmidt_values(j+i-1,'r'))
            u,s,v = nph.trunc_svd_before_index(contracted_tensor,3,xi=self.xi,cutoff=self.cutoff,norm=2**((self.L)/2))
            v = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(j+i-1,'r'))
            self.set_schmidt_values(j+i-1,'l',s)
            self[j+i-1] = v
            Amps[j-2] = u
        self[i] = np.einsum('k,k...->k...',1/self.get_schmidt_values(i,'l'),Amps[0])

    def merge_mpo_naive(self,other_mpo,i):
        pass
    