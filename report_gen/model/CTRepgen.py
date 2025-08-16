from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from report_gen.model.base_model import CTBaseModel
from report_gen.config import ModelConfig
from dataclasses import dataclass, fields 


class CTConfig(LlamaConfig, ModelConfig):
    """
    Custom configuration class for CT Llama model that extends both LlamaConfig and ModelConfig.
    """
    model_type = "ct_llama"
    
    def __init__(self, **kwargs):
        # Initialize LlamaConfig first
        super().__init__(**kwargs)
        
        # Override LlamaConfig attributes with ModelConfig defaults
        model_config_instance = ModelConfig()  # This gives the default values for ModelConfig
        
        # Now, override any matching parameters from ModelConfig
        for field in fields(ModelConfig):  # Iterate over the fields in ModelConfig
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])  # Override with value from kwargs
            else:
                setattr(self, field.name, getattr(model_config_instance, field.name))  # Use default value if not in kwargs


class CTLlamaForCausalLM(LlamaForCausalLM):
    """
    Causal language model based on LLaMA, extended with a CTBaseModel base model / backbone 
    for multimodal processing. The CTBaseModel handles the main feature encoding.
    """
    
    config_class = CTConfig

    def __init__(self, config):
        """Initialize model with CTBaseModel and LM head."""
        super().__init__(config)
        self.model = CTBaseModel(config)
        self.lm_head = nn.Linear(config.llama_dim, config.vocab_size, bias=False)
        self.config = config
        self.post_init()

        
    def resize(self, tokenizer):
        """Resize token embeddings to match tokenizer vocabulary."""
        self.resize_token_embeddings(len(tokenizer))
    
    def get_model(self):
        """Return the underlying CTBaseModel."""
        return self.model

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing."""
        self.model._set_gradient_checkpointing(True)

    def forward(
            self,
            local_features: Optional[torch.FloatTensor] = None,
            global_features: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            input_texts: Optional[str] = None,
            reports: Optional[List[str]] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for report generation.

        This method prepares multimodal token embeddings via
        `self.model.prepare_inputs_for_repgen(...)` from the base model (CTBaseModel).
        It then uses the standard LLama forward pass for report generation 
        based on the multimodal embeddings.

        Args:
            local_features, global_features (torch.FloatTensor): Local/global feature tensors for feature encoding.
            input_ids (torch.LongTensor): Token IDs for the textual input.
            labels (torch.LongTensor): Next-token labels (with -100 for ignored positions).
            attention_mask, position_ids, past_key_values (torch.Tensor): Standard HF forward pass arguments.
            inputs_embeds (torch.FloatTensor): Precomputed embeddings to bypass embed construction.
            output_attentions, output_hidden_states, return_dict, use_cache, cache_position:
                Standard generation flags.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: Standard HF causal LM output.
        """
        
        if inputs_embeds is not None or past_key_values is not None:
            # Skip multimodal prep; just run the parent forward
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        

        inputs_embeds = self.model.prepare_input_embeddings(
            input_ids,
            local_features,
            global_features
        )

        output = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return output
        


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        local_features: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,  # keep this if you still want to catch any extra args
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        """
        Generates report based on given textual input (inputs) and local and global features.

        Args:
            inputs (torch.LongTensor): Token IDs for the textual input.
            local_features, global_features (torch.FloatTensor): Local/global feature tensors for feature encoding.
            labels (torch.LongTensor): Next-token labels (with -100 for ignored positions).
            attention_mask, position_ids (torch.Tensor): Standard HF forward pass arguments.
            pad_token_id (int): Token ID for padding.
            eos_token_id (int): Token ID that ends the generation process.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: Standard HF causal LM output.
        """
        
        inputs_embeds = self.model.prepare_input_embeddings(
            inputs,
            local_features,
            global_features
        )

        output_ids = super().generate(
            inputs=inputs,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=self.config.max_length
        )

        return output_ids
    

AutoConfig.register("ct_llama", CTConfig)
AutoModelForCausalLM.register(CTConfig, CTLlamaForCausalLM)