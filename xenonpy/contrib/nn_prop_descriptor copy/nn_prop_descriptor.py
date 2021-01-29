from typing import Union
from xenonpy.descriptor.base import BaseFeaturizer, BaseDescriptor

from xenonpy.model.training import Trainer
from xenonpy.datatools import Scaler
import torch, numpy

class NNPropDescriptor(BaseFeaturizer):
    def __init__(self, fp_calc:Union[BaseDescriptor, BaseFeaturizer], nnmdl:Trainer, scaler:Scaler):
    
        super().__init__(n_jobs=0)
        self.fp = fp_calc
        self.nn = nnmdl
        self.scaler = scaler
        self.columns = ['thermal_conductivity', 'thermal_diffusivity', 'density',
       'static_dielectric_const', 'nematic_order_param', 'Cp', 'Cv',
       'compress_T', 'compress_S', 'bulk_modulus_T', 'bulk_modulus_S',
       'speed_of_sound', 'volume_expansion', 'linear_expansion']
        
    def featurize(self, x,):
        tmp_df = self.fp.transform(x)
        output = self.nn.predict(x_in=torch.tensor(tmp_df.values, dtype=torch.float)).detach().numpy()
        return  pd.DataFrame(self.scaler.inverse_transform(output), index=tmp_df.index,columns=self.columns)
    
    @property
    def feature_labels(self):
        return self.columns
    