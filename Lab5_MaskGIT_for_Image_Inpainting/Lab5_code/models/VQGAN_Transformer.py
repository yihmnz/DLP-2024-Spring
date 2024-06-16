import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path, device):
        # self.transformer.load_state_dict(torch.load(load_ckpt_path))
        checkpoint = torch.load(load_ckpt_path, map_location=device)
        self.load_state_dict(checkpoint)

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        # output = self.vqgan.encode(x)
        # print(output)
        # print(11111)
        quant_z, indices, metric = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r:1-r
        elif mode == "cosine":
            return lambda r:np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r:1-r ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        quant_z, z_indices = self.encode_to_z(x)
    
        r = np.random.uniform()
        mask_rate = self.gamma(r)
        mask = torch.rand(z_indices.shape, device=z_indices.device) < mask_rate
        masked_indices = z_indices.masked_fill(mask, self.mask_token_id)
        
        logits = self.transformer(masked_indices)
        
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, x, mask_b, token_update, ratio, true_count, mask_diff_count, last_epoch = False):
        # encode original image to latent 
        quant_z, z_indices = self.encode_to_z(x)

        # Correct latent which was updated previously
        selected_z_indices = z_indices
        non_zero_indices = token_update != 0
        token_update = token_update.long()
        try:
            selected_z_indices[non_zero_indices] = token_update[non_zero_indices]
            # print(selected_z_indices)
        except Exception as e:
            selected_z_indices = selected_z_indices
            print(f"An error occurred: {e}")
        # print(selected_z_indices.shape)

        # Mask tokens through masked_fill
        selected_z_indices = selected_z_indices.masked_fill(mask_b, self.mask_token_id)
        # print(selected_z_indices)

        # Biderectional transformer prediction
        logits = self.transformer(selected_z_indices)
        # Convert logits to a probability distribution
        logits = F.softmax(logits, dim=-1)

        # Simulate Gumbel noise for stochastic sampling
        g = -torch.log(-torch.log(torch.rand_like(logits)))
        temperature = self.choice_temperature *(1-ratio)  # Adjust temperature for annealing
        # Calculate confidence scores by adding temperature-scaled Gumbel noise
        confidence = torch.log(logits) + temperature * g
        # Get the predicted indices with the highest confidence
        z_indices_predict_prob, z_indices_predict = torch.max(confidence, dim=-1)
        
        # Addback original tokens (unmasked ones)
        inverted_mask = ~mask_b
        z_indices_predict[inverted_mask] = selected_z_indices[inverted_mask]
        
        # calculated threshold using only masked token prediction (# true means mask)
        count_threshold_z_indices = z_indices_predict_prob[mask_b]
        if not last_epoch: #(last epoch no threshold problem)
            # Update the mask: Reduce the number of masked indices based on confidence and ratio
            # Sort confidence to find a threshold for high confidence predictions
            sorted_confidence, _ = torch.sort(count_threshold_z_indices, descending=True)
            threshold_index = int((1-ratio)*true_count)-mask_diff_count # correct through mask_diff amount
            try:
                threshold_value = sorted_confidence[threshold_index].unsqueeze(-1)
            except:
                threshold_index = int((1 - ratio) * true_count)
                print(threshold_index, ':error')
                threshold_value = sorted_confidence[threshold_index].unsqueeze(-1)
            # Update the mask where confidence is below threshold (above threshold = no mask)
            z_indices_predict_prob[~mask_b] = float('inf')
            mask_bc = z_indices_predict_prob > threshold_value
            mask_bc = ~mask_bc
        else:
            threshold_value = float('-inf')
            mask_bc = z_indices_predict_prob > threshold_value
            mask_bc = ~mask_bc
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
