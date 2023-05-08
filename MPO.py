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
    
    def merge_mpo_zip_up(self,other_mpo,i):
        if self.L < other_mpo.L:
            raise ValueError("MPOs must have the same length")
        k1 = len(self.get_schmidt_values(i,'l'))
        C = np.identity(k1).reshape((k1,1,k1))
        Amps = self.get_all_A(i,i+other_mpo.L)
        Ampo = other_mpo.get_all_A(0,other_mpo.L)
        for j in range(other_mpo.L):
            C = np.einsum('ijk,knlm->ijnlm',C,Amps[j])
            C = np.einsum('ijnlm,jqnp->iqlpm',C,Ampo[j])
            u,s,v = nph.trunc_svd_before_index(C,3,self.xi,1e-15)
            u = np.einsum('k,k...->k...',1/self.get_schmidt_values(i+j,'l'),u)
            self[i+j]=np.einsum('k,k...->k...',1/self.get_schmidt_values(i+j,'l'),u)
            self.set_schmidt_values(i+j,'r',s)
            C = np.einsum('k,k...->k...',s,v)
        C = C.reshape((int(np.prod(C.shape)),))
        s = self.get_schmidt_values(i+other_mpo.L,'l')
        self.set_schmidt_values(i+other_mpo.L,'l',np.einsum('k,k->k',s,C))
        self.into_canonical_form()

    def merge_mpo_naive(self,other_mpo,i):
        pass
    