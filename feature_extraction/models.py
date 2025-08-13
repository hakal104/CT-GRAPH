import sys
import os

from typing import Optional, Tuple, List, Any, Union
from abc import ABC, abstractmethod

import torch
from monai.networks.nets import SwinUNETR
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from collections import OrderedDict
from monai.utils import ensure_tuple_rep
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged, 
    Transposed,
    Flipd,
    Resized,
    Spacingd,
    RandRotate90d,
    EnsureTyped
)
from monai.transforms import ThresholdIntensityd, NormalizeIntensityd, EnsureTyped


from feature_extraction.model_archs.vox2vec import fpn
from feature_extraction.model_archs.transvw.ynet3d import *
from lighter_zoo import SegResEncoder


#Following pretrained feature encoder models are used as in the official paper.
MODEL_REGISTRY = {
    'swinunetr': None,
    'swinunetr_voco': None,
    'swinunetr_voco_10k': None,
    'vox2vec': None,
    'transvw': None,
    'ctfm': None
    }
    
class PretrainedEncoder(ABC):
    """
    Abstract class for pretrained vision encoders used to provide the 
    models and corresponding information such as preprocessing, feature
    dimensions and patch size needed to perform feature extraction.
    """

    def __init__(
        self,
        weight_path = None,
    ) -> None:
        
        self.weight_path = weight_path

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the initialized torch module (encoder/backbone)."""
        pass

    @abstractmethod
    def get_transforms(self) -> Any:
        """Return preprocessing/augmentation transforms."""
        pass

    @abstractmethod
    def get_feature_dims(self) -> List[int]:
        """Return feature channel sizes (e.g., per layers)."""
        pass

    @abstractmethod
    def get_roi(self) -> Tuple[int, int, int]:
        """Return patch size for sliding window inference."""
        pass

class swinunetr_model(PretrainedEncoder):
    def __init__(self, weight_path=None, is_voco=False, arch_size='B'):
        # Define the architecture
        self.weight_path = weight_path
        self.feature_sizes = {'B':48,
                              'L':96,
                              'H':192}
        self.arch_size = arch_size
        self.is_voco = is_voco
        
    def get_model(self):
        # Construct the SwinUNETR architecture
        
        if self.is_voco:
            state_dict = torch.load(self.weight_path, map_location=torch.device('cpu'))
            model = SwinUNETR(img_size=(96, 96, 96),
                    in_channels=1,
                    out_channels=21,
                    feature_size=self.feature_sizes[self.arch_size],
                    use_v2=True)
            current_model_dict = model.state_dict()
            for k in current_model_dict.keys():
                if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
                    print(k)
            new_state_dict = {
                k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
                for k in current_model_dict.keys()}
            model.load_state_dict(new_state_dict, strict=True)
            swinViT = model.swinViT
        else:
            patch_size = ensure_tuple_rep(2, 3)
            window_size = ensure_tuple_rep(7, 3)
            
            swinViT = SwinViT(
                in_chans=1,
                embed_dim=48,
                window_size=window_size,
                patch_size=patch_size,
                depths=[2, 2, 2, 2],
                num_heads=[3, 6, 12, 24],
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0,
                norm_layer=torch.nn.LayerNorm,
                use_checkpoint=False,
                spatial_dims=3,
            )

            new_state_dict = OrderedDict((k.replace("module.", "").replace("fc", "linear"), v) for k, v in torch.load(self.weight_path)['state_dict'].items())
            swinViT.load_state_dict(new_state_dict, strict=False)
                
            swinViT.patch_embed.proj.weight.requires_grad = False
            swinViT.patch_embed.proj.bias.requires_grad = False     
            for name, param in swinViT.named_parameters():
                param.requires_grad = False
                if any(f"layers{i}" in name for i in range(3)):  # Adjust based on architecture
                    param.requires_grad = False
                
        return swinViT
    
    def get_transforms(self):
    
        transforms = ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)
        return transforms
                
    
    def get_feature_dims(self):
        base_dim = self.feature_sizes[self.arch_size]
        dims = [base_dim*2**i for i in range(0,5)]
        return dims
    
    def get_roi(self):
        return (96,96,96)

class vox2vec_model(PretrainedEncoder):
    def __init__(self, weight_path):
        
        self.weight_path = weight_path
        
    def get_model(self):

        weights=torch.load(self.weight_path)
        model=fpn.FPN3d(in_channels=1, base_channels=16, num_scales=6)
        model.load_state_dict(weights)

        return model
    
    def get_transforms(self):

        transforms = ScaleIntensityRanged(keys=["image"], a_min=-1350, a_max=1000, b_min=0.0, b_max=1.0, clip=True)
        
        return transforms 
    
    def get_feature_dims(self):
        return [16,32,64,128,256,512]

    def get_roi(self):
        return (128,128,64)
    
class transvw_model(PretrainedEncoder):
    def __init__(self, weight_path):
        # Define the architecture
        self.weight_path = weight_path
        
    def get_model(self):
        # Construct the SwinUNETR architecture
        model = UNet3D_encoder().to('cuda')
        state_dict = torch.load(self.weight_path)['state_dict']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        delete = [key for key in state_dict if "projection_head" in key]
        for key in delete: del state_dict[key]
        delete = [key for key in state_dict if "prototypes" in key]
        for key in delete: del state_dict[key]
        model.load_state_dict(state_dict,strict=False)
        model.eval()
        #ve_dict = {k.replace('embedding_layer.', '', 1): v for k, v in weights.items()}
        #model.load_state_dict(ve_dict,strict=False)

        return model
    
    def get_transforms(self):

        transforms = ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)

        return transforms 
    
    def get_feature_dims(self):
        return [64,128,256,512]
    
    def get_roi(self):
        return (128,128,64)
    
class ctfm_model(PretrainedEncoder):
    def __init__(self, weight_path):
        # Define the architecture
        self.weight_path = weight_path
        
    def get_model(self):
        # Construct the SwinUNETR architecture
        model = SegResEncoder.from_pretrained(
            "project-lighter/ct_fm_feature_extractor"
        )
        model.eval()

        return model
    
    def get_transforms(self):

        transforms = Compose([
            ScaleIntensityRanged(keys="image", a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
        ])

        return transforms 
    
    def get_feature_dims(self):
        return [32,64,128,256,512]
    
    def get_roi(self):
        return (64,128,128)
    
    
MODEL_REGISTRY.update({
    'swinunetr': swinunetr_model,
    'swinunetr_voco': swinunetr_model,
    'swinunetr_voco_10k': swinunetr_model,
    'vox2vec': vox2vec_model,
    'transvw': transvw_model,
    'ctfm': ctfm_model
})

    
def get_models(arch, weight_path, is_voco=False, arch_size='B',  feature_dims_only=False):
    """
    Retrieve the model and associated transforms based on the architecture name.
    
    Args:
        arch (str): The model architecture (e.g., 'swinunetr', 'totalseg')
        weight_path (str): Path to the pre-trained weights.
        is_voco (bool): Whether VoCo is used or not. 
        arch_size (str): Architecture size (needed for swinunetr).
        feature_dims_only(bool): Whether only the feature dims are needed and
                                 no model initialization.
    
    Returns: 
        The model and its corresponding transforms.
    """

    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}")

    model_class = MODEL_REGISTRY[arch]
    if 'swinunetr' in arch:
        model_instance = model_class(weight_path, is_voco=is_voco, arch_size=arch_size)
    else:
        model_instance = model_class(weight_path)
    feature_dims = model_instance.get_feature_dims()
    
    if feature_dims_only:
        return feature_dims
    else:
        model = model_instance.get_model()
        transforms = model_instance.get_transforms()
        feature_dims = model_instance.get_feature_dims()
        roi = model_instance.get_roi()

    return model, transforms, feature_dims, roi
