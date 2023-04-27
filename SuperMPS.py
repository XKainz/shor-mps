'''SuperMPS.py'''
import numpy as np
import numpy.linalg as la
import numpy_helpers as nph
import matplotlib.pyplot as plt

class SuperMPS:
    def __init__(self,*args,xi,cutoff=1e-8):
        self.xi = xi
        self.cutoff = cutoff
        if isinstance(args[0],list):
            self.MPS = args[0]
            if self.MPS[0].shape != (1,1):
                self.MPS = [np.ones((1,1),dtype="complex")] + self.MPS
            if self.MPS[-1].shape != (1,1):
                self.MPS = self.MPS + [np.ones((1,1),dtype="complex")]
            self.indices_per_node = len(self.MPS[1].shape)-2
            self.L = len(self.MPS)//2
            for i in range(len(self.MPS)):
                if max(self.MPS[i].shape) > self.xi:
                    raise ValueError("xi too small for MPS")

        elif isinstance(args[0],np.ndarray) and isinstance(args[1],int):
            self.indices_per_node = args[1]
            tensor = args[0]
            if len(tensor.shape)%self.indices_per_node != 0:
                raise ValueError("Tensor shape is not compatible with indices_per_node")
            self.L = len(tensor.shape)//self.indices_per_node
            if self.indices_per_node != 1:
                tensor = transpose_gate_ind_format(tensor,self.indices_per_node)
            MPS = [np.ones((1,1),dtype="complex")]
            tensor = np.reshape(tensor,(1,)+tensor.shape)
            for i in range(self.L):
                s1 = tensor.shape[:1+self.indices_per_node]
                s2 = tensor.shape[1+self.indices_per_node:]
                tensor = np.reshape(tensor,(int(np.prod(s1)),int(np.prod(s2))))
                u,s,v,ximin = nph.trunc_svd(tensor,self.xi,self.cutoff)
                u = np.reshape(u,s1+(ximin,))
                v = np.reshape(v,(ximin,)+s2)
                d = np.diag(s)
                MPS.append(u)
                MPS.append(d)
                tensor = v
            self.MPS = MPS

    def __len__(self):
        return self.L
    
    def __getitem__(self,i):
        return self.MPS[2*i+1]
    
    def __setitem__(self,i,value):
        self.MPS[2*i+1] = value
    
    def get_contracted_tensor(self,i_start,i_end):
        if i_start > i_end:
            raise ValueError("i_start must be less than or equal to i_end")
        if i_start < 0 or i_end >= self.L:
            raise ValueError("i_start and i_end must be in range [0,self.L)")
        i_start = self.to_MPS_index(i_start)-1
        i_end = self.to_MPS_index(i_end)+1
        contracted_tensor = self.MPS[i_start]
        for i in range(i_start+1,i_end+1):
            contracted_tensor = np.tensordot(contracted_tensor,self.MPS[i],axes=([-1],[0]))
        return contracted_tensor

    def to_MPS_index(self,i):
        return 2*i+1
    
    def to_Tensor(self):
        tensor = self.get_contracted_tensor(0,self.L-1)
        tensor = np.reshape(tensor,tensor.shape[1:-1])
        return tensor
    
    def get_schmidt_matrix(self,i,side):
        index = self.to_MPS_index(i)
        if side == 'r':
            return self.MPS[index+1]
        elif side == 'l':
            return self.MPS[index-1]
        else:
            raise ValueError("side must be 'l' or 'r'")
    
    def set_schmidt_matrix(self,i,side,value):
        index = self.to_MPS_index(i)
        if side == 'r':
            self.MPS[index+1] = value
        elif side == 'l':
            self.MPS[index-1] = value
        else:
            raise ValueError("side must be 'l' or 'r'")
    
    def get_A_B_config(self,j):
        ABMPS = []
        for i in range(j):
            A = np.tensordot(self.get_schmidt_matrix(i,'l'),self[i],axes=([-1],[0]))
            ABMPS.append(A)
        ABMPS.append(self.get_schmidt_matrix(j,'l'))
        for i in range(j,self.L):
            B = np.tensordot(self[i],self.get_schmidt_matrix(i,'r'),axes=([-1],[0]))
            ABMPS.append(B)
        return ABMPS
    
    def get_B_config(self):
        return self.get_A_B_config(0)
    
    def get_A_config(self):
        return self.get_A_B_config(self.L-1)
    
    def print_all_shapes(self):
        for i in range(self.to_MPS_index(len(self))):
            print("[",i,"]",self.MPS[i].shape)
    
    def plot_schmidt_values(self):
        plt.figure(figsize=(15,7))
        for i in range(self.L):
            d = np.diagonal(self.get_schmidt_matrix(i,'l'))
            plt.plot(range(len(d)),np.absolute(d),label="i="+str(i))
        plt.show()

    def plot_bond_dims(self):
        plt.figure(figsize=(15,7))
        d = []
        for i in range(self.L-1):
            d.append(self[i].shape[-1])
        plt.plot(range(len(d)),np.absolute(d),label="Bond Dimensions from i to i+1")
        plt.show()
def transpose_gate_ind_format(gate,ind_per_node):
    if len(gate.shape)%ind_per_node != 0:
        raise ValueError("Tensor shape is not compatible with indices_per_node")
    L = len(gate.shape)//ind_per_node
    tranpose_vec = [0,]*len(gate.shape)
    for i in range(L):
        for j in range(ind_per_node):
            tranpose_vec[i*ind_per_node+j] = i+j*L
    return np.transpose(gate,tranpose_vec)