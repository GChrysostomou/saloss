import torch
import torch.nn as nn
import math 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import json
import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))


# class mag(nn.Module):

#     def __init__(self, hidden_dim = 768):

#         super(mag, self).__init__()

#         if args.saliency_scorer == "return_all":

#             self.transform = nn.Linear(3, hidden_dim // 2)

#             self.realign = nn.Linear(hidden_dim // 2 , 1, bias = False)

#         else:

#             pass
        
#     def forward(self, text_embeds, salient, attention_mask):


#         if args.saliency_scorer == "return_all":
            
#             mask = attention_mask.squeeze(1).transpose(1,-1)

#             mask = (mask == -0.).float()

#             first_transformation = nn.ReLU()(self.transform(salient))
#             realignment_scores = self.realign(first_transformation) * mask

#             return realignment_scores * text_embeds

#         else:

#             return text_embeds * salient.unsqueeze(-1)


"""
Approach 2
"""

class mag(nn.Module):

    def __init__(self, hidden_dim = 768):

        super(mag, self).__init__()

        if args.saliency_scorer == "return_all":

            self.transform = nn.Linear(3, hidden_dim // 2)

            self.realign = nn.Linear(hidden_dim // 2 , hidden_dim, bias = False)

        else:

            pass
        
    def forward(self, text_embeds, salient, attention_mask):


        if args.saliency_scorer == "return_all":
            
            mask = attention_mask.squeeze(1).transpose(1,-1)

            mask = (mask == -0.).float()

            first_transformation = nn.ReLU()(self.transform(salient))
            second_transformation = self.realign(first_transformation) * mask

            return second_transformation * text_embeds

        else:

            return text_embeds * salient.unsqueeze(-1).float().to(device)

"""
Approach 3
"""

# class mag(nn.Module):

#     def __init__(self, hidden_dim = 768):

#         super(mag, self).__init__()

#         if args.saliency_scorer == "return_all":

#             self.reform = nn.Linear(hidden_dim + 3, hidden_dim, bias = True)

#         else:

#             pass
        
#     def forward(self, text_embeds, salient, attention_mask):


#         if args.saliency_scorer == "return_all":
            
#             combined = torch.cat([text_embeds, salient], axis = -1)

#             reformed = nn.ReLU()(self.reform(combined))

#             # scale = (torch.sqrt((reformed ** 2).sum()).detach()/ torch.sqrt((text_embeds ** 2).sum()).detach())

#             # scale = min(scale, 1)

#             return scale * reformed + text_embeds

#         else:

#             return text_embeds * salient.unsqueeze(-1)


"""
Approach 4
"""

# class mag(nn.Module):

#     def __init__(self, hidden_dim = 768):

#         super(mag, self).__init__()

#         if args.saliency_scorer == "return_all":

#             self.reform = nn.Linear(3,  hidden_dim // 2, bias = True)

#             self.realign = nn.Linear(hidden_dim // 2 , hidden_dim, bias = False)

#         else:

#             pass
        
#     def forward(self, text_embeds, salient, attention_mask):


#         if args.saliency_scorer == "return_all":
            
#             first_transformation = nn.ReLU()(self.reform(salient))
#             reformed = self.realign(first_transformation) 

#             scale = (torch.sqrt((reformed ** 2).sum()).detach()/ torch.sqrt((text_embeds ** 2).sum()).detach())

#             scale = min(scale, 1)

#             return scale * reformed + text_embeds

#         else:

#             return text_embeds * salient.unsqueeze(-1)


"""
Approach 5 : only on embeds
"""

# class mag(nn.Module):

#     def __init__(self, hidden_dim = 768):

#         super(mag, self).__init__()

#         if args.saliency_scorer == "return_all":

#             self.transform = nn.Linear(3, 64)

#             self.realign = nn.Linear(64 , 1, bias = False)

#         else:

#             pass
        
#     def forward(self, text_embeds, salient, attention_mask):


#         if args.saliency_scorer == "return_all":

#             import pdb; pdb.set_trace();
            
#             first_transformation = nn.ReLU()(self.transform(salient))
#             realignment_scores = torch.sigmoid(self.realign(first_transformation))
            
#             return text_embeds * realignment_scores

#         else:

#             return text_embeds * salient.unsqueeze(-1)


"""
FINAL ATTEMPT
"""

# class mag(nn.Module):

#     def __init__(self, hidden_dim = 768):

#         super(mag, self).__init__()

#         with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
#             args = AttrDict(json.load(f))

#         if args.saliency_scorer == "return_all":

#             self.transform = nn.Linear(3, hidden_dim // 2)

#             self.realign = nn.Linear(hidden_dim // 2 , 1, bias = False)

#         else:

#             if args.tfidf_asvec:
                
#                 self.W_hs = nn.Linear(hidden_dim + args.tfidf_vecdim, hidden_dim)
#                 self.W_s = nn.Linear(args.tfidf_vecdim, hidden_dim)

#                 self.beta_shift = 1e-3

#                 self.LayerNorm = nn.LayerNorm(hidden_dim)
#                 self.dropout = nn.Dropout(0.5)

#             else:

#                 pass
        
#     def forward(self, text_embeds, salient, attention_mask):


#         if args.saliency_scorer == "return_all":
            

#             mask = attention_mask.squeeze(1).transpose(1,-1)

#             mask = (mask == -0.).float()

#             first_transformation = nn.ReLU()(self.transform(salient))
#             realignment_scores = self.realign(first_transformation) * mask

#             return realignment_scores * text_embeds


#         else:
            
#             if args.tfidf_asvec:
                
#                 salient = salient.unsqueeze(1).repeat(1, text_embeds.size(1), 1).float()

#                 eps = 1e-6

#                 weight_s = nn.ReLU()(self.W_hs(torch.cat((salient, text_embeds), dim=-1)))

#                 h_m = weight_s * self.W_s(salient)

#                 em_norm = text_embeds.norm(2, dim=-1)
#                 hm_norm = h_m.norm(2, dim=-1)

#                 hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(device)
#                 hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

#                 thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

#                 ones = torch.ones(thresh_hold.shape, requires_grad=True).to(device)

#                 alpha = torch.min(thresh_hold, ones)
#                 alpha = alpha.unsqueeze(dim=-1)

#                 salient_embedding = alpha * h_m

#                 embedding_output = self.dropout(
#                     self.LayerNorm(salient_embedding + text_embeds)
#                 )

#                 return embedding_output
                
#             else:

#                 return text_embeds * salient.unsqueeze(-1)