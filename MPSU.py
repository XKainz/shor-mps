'''MPSU.py'''
from MPS import MPS
import numpy as np
import numpy_helpers as nph

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
        self.register_b = np.tensordot(gate,self.register_b,axes=([*range(self.Lrb,2*self.Lrb)],[*range(1,self.Lrb+1)]))

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
        


