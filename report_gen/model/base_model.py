import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from report_gen.model.variants import VARIANT_REGISTRY

class CTBaseModel(LlamaModel):
    """Base model that utilizes a variant from `VARIANT_REGISTRY`
    to encode local and global visual features and prepares the 
    input embeddings for report generation."""
    
    def __init__(self, config):
        """
        Args:
            config: Model configuration.
                Must include the mode, which is the key for selecting a variant class from VARIANT_REGISTRY.
        """
        
        super().__init__(config)
        self.config = config
        variant_cls = VARIANT_REGISTRY[config.mode]
        self.variant = variant_cls(config)

    def encode_features(self, local_features, global_features):
        """
        Fuse and project local and global features via the selected variant.

        Args:
            local_features (torch.Tensor): shape [B, N, D_local] — batch of N local feature vectors
            global_features (torch.Tensor): shape [B, D_global, H, W, D] — batch of global feature maps
            with spatial dims (H, W, D)

        Returns:
            Projected embeddings for report generation.
        """
        
        return self.variant.encode_features(local_features, global_features)

    def prepare_input_embeddings(
        self, input_ids, local_features, global_features
    ):
        """
        Build token embeddings that include visual features.

        Args:
            input_ids (torch.LongTensor): Token IDs for the textual input.
            local_features (torch.Tensor): shape [B, N, D_local] — batch of N local feature vectors
            global_features (torch.Tensor): shape [B, D_global, H, W, D] — batch of global feature maps
            with spatial dims (H, W, D)

        Returns:
            inputs_embeds: Tensor of combined visual + text embeddings.
        """
        
        image_features = self.encode_features(local_features, global_features)
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.cat(
            (inputs_embeds[:, :2, :],
             image_features,
             inputs_embeds[:, (image_features.shape[1] + 2):, :]),
            dim=1
        )
        return inputs_embeds