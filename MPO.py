'''MPO.py'''

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
    