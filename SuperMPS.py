'''SuperMPS.py'''
import numpy as np
import numpy.linalg as la
import numpy_helpers as nph
import matplotlib.pyplot as plt

class SuperMPS:
    def __init__(self,mps_array,L,indices_per_node,xi,cutoff):
        self.MPS = mps_array
        self.L = L
        self.indices_per_node = indices_per_node
        self.xi = xi
        self.cutoff = cutoff
    
    @staticmethod
    def create_SuperMPS_from_tensor(tensor,indices_per_node,xi,cutoff):
        if len(tensor.shape)%indices_per_node != 0:
                raise ValueError("Tensor shape is not compatible with indices_per_node")
        L = len(tensor.shape)//indices_per_node
        if indices_per_node != 1:
            tensor = transpose_gate_ind_format(tensor,indices_per_node)
        MPS = [np.ones(1,dtype="complex")]
        tensor = np.reshape(tensor,(1,)+tensor.shape+(1,))
        for i in range(L):
            u,s,v = nph.trunc_svd_before_index(tensor,1+indices_per_node,xi,cutoff)
            u = np.einsum('k,k...->k...',1/MPS[-1],u)
            MPS.append(u)
            MPS.append(s)
            tensor = np.einsum('k,k...->k...',s,v)
        return SuperMPS(MPS,L,indices_per_node,xi,cutoff)
    
    @staticmethod
    def create_SuperMPS_from_tensor_array(mps_array,xi,cutoff):
        MPS = mps_array
        if MPS[0].shape != (1,):
            MPS = [np.ones(1,dtype="complex")] + MPS
        if MPS[-1].shape != (1,):
            MPS = MPS + [np.ones(1,dtype="complex")]
        indices_per_node = len(MPS[1].shape)-2
        L = len(MPS)//2
        return SuperMPS(MPS,L,indices_per_node=indices_per_node,xi=xi,cutoff=cutoff)
            

    def __len__(self):
        return self.L
    
    def __getitem__(self,i):
        return self.MPS[2*i+1]
    
    def __setitem__(self,i,value):
        self.MPS[2*i+1] = value
    
    def get_contracted_tensor(self,i_start,i_end):
        if i_start <0 or i_start >= self.L:
            raise ValueError("i_start out of range")
        if i_end < i_start or i_end > self.L:
            raise ValueError("i_end out of range")
        contracted_tensor = np.diag(self.get_schmidt_values(i_start,'l'))
        for i in range(i_start,i_end):
            contracted_tensor = np.tensordot(contracted_tensor,self[i],axes=([-1],[0]))
            contracted_tensor = np.einsum('...k,k->...k',contracted_tensor,self.get_schmidt_values(i,'r'))
        return contracted_tensor

    def to_MPS_index(self,i):
        return 2*i+1
    
    def to_Tensor(self):
        tensor = self.get_contracted_tensor(0,self.L)
        tensor = np.reshape(tensor,tensor.shape[1:-1])
        return tensor
    
    def get_schmidt_values(self,i,side):
        index = self.to_MPS_index(i)
        if side == 'r':
            return self.MPS[index+1]
        elif side == 'l':
            return self.MPS[index-1]
        else:
            raise ValueError("side must be 'l' or 'r'")
    
    def set_schmidt_values(self,i,side,value):
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
            A = np.einsum('k,k...->k...',self.get_schmidt_values(i,'l'),self[i])
            ABMPS.append(A)
        ABMPS.append(self.get_schmidt_values(j,'l'))
        for i in range(j,self.L):
            B = np.einsum('...k,k->...k',self[i],self.get_schmidt_values(i,'r'))
            ABMPS.append(B)
        return ABMPS
    
    def get_all_A(self,i,j):
        if i < 0 or i >= self.L:
            raise ValueError("i out of range")
        if j < i or j > self.L:
            raise ValueError("j out of range")
        As = []
        for k in range(i,j):
            As.append(self.get_A_of_site(k))
        return As
    
    def get_all_B(self,i,j):
        if i < 0 or i >= self.L:
            raise ValueError("i out of range")
        if j < i or j > self.L:
            raise ValueError("j out of range")
        Bs = []
        for k in range(i,j):
            Bs.append(self.get_B_of_site(k))
        return Bs
    
    def get_B_config(self):
        return self.get_A_B_config(0)
    
    def get_A_config(self):
        return self.get_A_B_config(self.L)
    
    def get_A_of_site(self,i):
        A = np.einsum('k,k...->k...',self.get_schmidt_values(i,'l'),self[i])
        return A
    
    def get_B_of_site(self,i):
        B = np.einsum('...k,k->...k',self[i],self.get_schmidt_values(i,'r'))
        return B
    
    def print_all_shapes(self):
        for i in range(self.to_MPS_index(len(self))):
            print("[",i,"]",self.MPS[i].shape)
    
    def plot_schmidt_values(self,title="Schmidt Values"):
        plt.figure(figsize=(15,7))
        for i in range(self.L):
            d = self.get_schmidt_values(i,'l')
            plt.plot(range(len(d)),np.absolute(d),label="i="+str(i),)
        plt.xlabel("Schmidt Value Index")
        plt.ylabel("Schmidt Value")
        plt.title(title)
        plt.show()
        plt.savefig(title+".png")

    def plot_bond_dims(self,title="Bond Dimensions"):
        plt.figure(figsize=(15,7))
        d = [self[0].shape[0]]
        for i in range(self.L):
            d.append(self[i].shape[-1])
        plt.plot(range(len(d)),np.absolute(d),label="Bond Dimensions from i to i+1")
        plt.xlabel("Site Index")
        plt.ylabel("Bond Dimension")
        plt.title(title)
        plt.savefig(title+".png")
    
    def maximum_bond_dim(self):
        d = []
        for i in range(self.L-1):
            d.append(self[i].shape[-1])
        return max(d)

    def set_xi(self,xi):
        s = self.get_schmidt_values(0,'l')
        xin = min(xi,len(s))
        if xin < len(s):
            s = s[:xin]
            self.set_schmidt_values(0,'l',s)
            self[0] = self[0][:xin,...]
        for i in range(self.L-1):
            s = self.get_schmidt_values(i,'r')
            xin = min(xi,len(s))
            if xin < len(s):
                s = s[:xin]
                s = nph.renormalize_vector(s)
                self.set_schmidt_values(i,'r',s)
                self[i] = self[i][...,:xin]
                self[i+1] = self[i+1][:xin,...]
        s = self.get_schmidt_values(self.L-1,'r')
        xin = min(xi,len(s))
        if xin < len(s):
            s = s[:xin]
            self.set_schmidt_values(0,'r',s)
            self[self.L-1] = self[self.L-1][:xin,...]
        self.xi = xi
    
    def into_canonical_form(self,up_down='up'):
        if self.indices_per_node == 1:
            norm = 1
        else:
            norm = 2**(self.L/2)
        if up_down == 'down':
            for i in range(self.L-1):
                contracted_tensor = self.get_contracted_tensor(i,i+2)
                u,s,v = nph.trunc_svd_before_index(contracted_tensor,1+self.indices_per_node,xi=self.xi,cutoff=self.cutoff,norm=norm)
                u = np.einsum('k,k...->k...',1/self.get_schmidt_values(i,'l'),u)
                v = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(i+1,'r'))
                self[i] = u
                self.set_schmidt_values(i,'r',s)
                self[i+1] = v
        elif up_down == 'up':
            for i in range(self.L,1,-1):
                contracted_tensor = self.get_contracted_tensor(i-2,i)
                u,s,v = nph.trunc_svd_before_index(contracted_tensor,self.indices_per_node+1,xi=self.xi,cutoff=self.cutoff,norm=norm)
                u = np.einsum('k,k...->k...',1/self.get_schmidt_values(i-2,'l'),u)
                v = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(i-1,'r'))
                self[i-2] = u
                self.set_schmidt_values(i-1,'l',s)
                self[i-1] = v
        else:
            raise ValueError("up_down must be 'up' or 'down'")
        
    def entanglement_at_site(self,i,side):
        if i < 0 or i >= self.L:
            raise ValueError("i out of range")
        return nph.von_neumann_entropy(self.get_schmidt_values(i,side))
    
    def get_entanglement_all_sites(self):
        ent = []
        for i in range(self.L):
            ent.append(self.entanglement_at_site(i,'l'))
        ent.append(self.entanglement_at_site(self.L-1,'r'))
        return ent

    def get_schmidt_values_all_sites(self):
        schmidt_values = [self.get_schmidt_values(0,'l')]
        for i in range(self.L):
            schmidt_values.append(self.get_schmidt_values(i,'r'))
        return schmidt_values

    def plot_entanglement(self,title="Entaglement"):
        plt.figure(figsize=(15,7))
        ent = self.entanglement_all_site()
        plt.plot(range(len(ent)),ent,label="Entanglement")
        plt.xlabel("Site Index")
        plt.ylabel("Von Neuman Entanglement Entropy")
        plt.title(title)
        plt.show()

    def maximum_entanglement(self):
        ent = self.entanglement_all_site()
        return max(ent)
    
    def print_all(self):
        for i in range(self.L):
            print("Site",i)
            print("Schmidt Values Left:",self.get_schmidt_values(i,'l').shape,self.get_schmidt_values(i,'l'))
            print("Schmidt Values Right:",self.get_schmidt_values(i,'l').shape,self.get_schmidt_values(i,'r'))
            print("Tensor:",self[i].shape,self[i])

def transpose_gate_ind_format(gate,ind_per_node):
    if len(gate.shape)%ind_per_node != 0:
        raise ValueError("Tensor shape is not compatible with indices_per_node")
    L = len(gate.shape)//ind_per_node
    tranpose_vec = [0,]*len(gate.shape)
    for i in range(L):
        for j in range(ind_per_node):
            tranpose_vec[i*ind_per_node+j] = i+j*L
    return np.transpose(gate,tranpose_vec)