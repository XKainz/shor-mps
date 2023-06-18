'''MPSU.py'''
from MPS import MPS
from SuperMPS import SuperMPS
import numpy as np
import numpy_helpers as nph
import copy 

class MPSU(MPS):
    def __init__(self,mps_array,L,indices_per_node,register_b,xi,cutoff=1e-8):
        super().__init__(mps_array,L,indices_per_node,xi,cutoff)
        self.register_b = register_b
        self.Lrb = len(self.register_b.shape)-1

    @staticmethod
    def create_MPSU_init_to_1(len_a,len_b,xi,cutoff):
        mps_object = MPS.create_MPS_init_to_N(0,len_a,xi=xi,cutoff=cutoff)
        #create tensor initialized to |1>
        register_b=np.zeros(2**len_b,dtype='complex')
        register_b[1] = 1
        register_b = register_b.reshape((1,)+(2,)*len_b)
        mpsu_object = MPSU(mps_object.MPS,mps_object.L,mps_object.indices_per_node,register_b,mps_object.xi,mps_object.cutoff)
        return mpsu_object
    
    def apply_fat_gate(self,gate):
        if len(gate.shape) != 2*self.Lrb:
            raise ValueError("Gate is not of appropriate size for register_b")
        print(self.register_b.shape,gate.shape)
        self.register_b = np.tensordot(gate,self.register_b,axes=([*range(self.Lrb,2*self.Lrb)],[*range(1,self.Lrb+1)]))
        self.register_b = self.register_b.transpose([self.Lrb]+[*range(self.Lrb)])
        print(self.register_b.shape)

    def apply_fat_xgate(self,gate):
        if len(gate.shape) != 2*(self.Lrb+1):
            raise ValueError("Gate is not of appropriate size for register_b")
        con_ten = np.einsum('...i,i->...i',self.get_A_of_site(self.L-1),self.get_schmidt_values(self.L-1,'r'))
        con_ten = np.tensordot(con_ten,self.register_b,axes=([-1],[0]))
        con_ten = np.tensordot(gate,con_ten,([*range(self.Lrb+1,2*(self.Lrb+1))],[*range(1,self.Lrb+2)]))
        con_ten = np.transpose(con_ten,[self.Lrb+1]+[*range(self.Lrb+1)])
        u,s,v = nph.trunc_svd_before_index(con_ten,2,self.xi,self.cutoff)
        self.register_b = v
        self.set_schmidt_values(self.L-1,'r',s)
        u = np.einsum('i,i...->i...',1/self.get_schmidt_values(self.L-1,'l'),u)
        self[self.L-1]=u
    
    def collapse_U(self):
        U = np.einsum('i,i...->i...',self.get_schmidt_values(self.L-1,'r'),self.register_b)
        U = np.reshape(U,(U.shape[0],2**self.Lrb))
        prop_distribution = np.einsum('ik,ik->k',U,np.conj(U))
        print(np.sum(prop_distribution),"total probability")
        prop_distribution = np.real_if_close(prop_distribution,tol=10**4)
        j = np.random.choice(np.arange(2**self.Lrb),p=prop_distribution)
        
        tensors = [self.get_schmidt_values(0,'l')]
        for i in range(self.L-1):
            tensors.append(copy.deepcopy(self[i]))
            tensors.append(copy.deepcopy(self.get_schmidt_values(i,'r')))
        last = copy.deepcopy(self[self.L-1])
        last = np.einsum('...i,i->...i',last,self.get_schmidt_values(self.L-1,'r'))
        last = np.tensordot(last,U[:,j],axes=([-1],[0]))
        last = last.reshape(last.shape+(1,))
        tensors.append(last)
        tensors.append(np.ones(1))

        new =  SuperMPS.create_SuperMPS_from_tensor_array(tensors,self.xi,self.cutoff)
        new.__class__ = MPS 
        new.into_canonical_form('up')
        return new





