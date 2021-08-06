import torch
import torch.nn as nn
import math 
import json
from tqdm import trange
import numpy as np
from collections import OrderedDict
import pandas as pd
import os 
import logging
import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nn.deterministic = True
torch.backends.cudnn.benchmark = False
    

torch.manual_seed(25)
torch.cuda.manual_seed(25)
np.random.seed(25)





def extractor(model, data, save_path):

    """
        Info: computes the average fraction of tokens required to cause a decision flip (prediction change)
        Input:
            model : pretrained model
            data : torch.DataLoader loaded data
            save_path : path to save the results
        Output:
            saves the results to a csv file under the save_path
    """

    ig_true = True

    results_collect = {}


    pbar = trange(len(data), desc='extracting imporance_scores', leave=True)
    
    if args.saliency_scorer is None: sal_scorer = "vanilla"

    new_path = os.getcwd() + "/saved_importance_metrics/" + args.dataset + "/"#+ sal_scorer + "_" + args.dataset + 

    os.makedirs(new_path, exist_ok = True)


    for batch in data:
        
        model.eval()
        model.zero_grad()

        batch = [torch.stack(t).transpose(0,1) if type(t) is list else t for t in batch]
        
        inputs = {
            "sentences" : batch[0].to(device),
            "lengths" : batch[1].to(device),
            "labels" : batch[2].to(device),
            "annotation_id" : batch[3],
            "query_mask" : batch[4].to(device),
            "token_type_ids" : batch[5].to(device),
            "attention_mask" : batch[6].to(device),
            "retain_gradient" : True
        }

        
        yhat, attentions =  model(**inputs)

        yhat.max(-1)[0].sum().backward(retain_graph = True)

        #embedding gradients
        embed_grad = model.bert_model.model.embeddings.word_embeddings.weight.grad
        g = embed_grad[inputs["sentences"].long()][:,:max(inputs["lengths"])]

        # cutting to length to save time
        attentions = attentions[:,:max(inputs["lengths"])]
        query_mask = inputs["query_mask"][:,:max(inputs["lengths"])]

        # import pdb; pdb.set_trace();

        em = model.bert_model.model.embeddings.word_embeddings.weight[inputs["sentences"].long()][:,:max(inputs["lengths"])]

        gradients = (g* em).sum(-1).abs() * query_mask.float()

        if ig_true:

            integrated_grads = model.integrated_grads(
                    original_grad = g, 
                    original_pred = yhat.max(-1),
                    **inputs    
            )

            # normalised integrated gradients of input
            normalised_ig = model.normalise_scores(integrated_grads * query_mask.float(), inputs["sentences"][:, :max(inputs["lengths"])])
            normalised_ig = torch.masked_fill(normalised_ig, model.normalised_mask, float("-inf"))
            normalised_ig = torch.softmax(normalised_ig, -1)

        # normalised gradients of input
        normalised_grads = model.normalise_scores(gradients, inputs["sentences"][:, :max(inputs["lengths"])])

        # normalised attention
        normalised_attentions = model.normalise_scores(attentions * query_mask.float(), inputs["sentences"][:, :max(inputs["lengths"])])

        # retrieving attention*attention_grad
        attention_gradients = model.weights_or.grad[:,:,0,:].mean(1)[:,:max(inputs["lengths"])]
        
        normalised_attention_grads =  model.normalise_scores(attentions * attention_gradients * query_mask.float(), inputs["sentences"][:, :max(inputs["lengths"])])
        
        # softmaxing due to negative attention gradients 
        # therefore we receive also negative values and as such
        # the pad and unwanted tokens need to be converted to -inf 
        normalised_attention_grads = torch.masked_fill(normalised_attention_grads, model.normalised_mask, float("-inf"))
        normalised_attention_grads = torch.softmax(normalised_attention_grads, -1)
        
        for indx in range(yhat.size(0)):
            
            annot = inputs["annotation_id"][indx]

            results_collect[annot] = {
                "attention" : list(normalised_attentions[indx].detach().cpu().numpy()),
                "gradients" : list(normalised_grads[indx].detach().cpu().numpy()),
                "attention-gradients" : list(normalised_attention_grads[indx].detach().cpu().numpy())
            }

            if ig_true:

                results_collect[annot]["ig"] = list(normalised_ig[indx].detach().cpu().numpy()),

        pbar.update(1)


    if args.saliency_scorer is None: sal_scorer = "vanilla"
    else: sal_scorer = args.saliency_scorer
    new_file = new_path + sal_scorer + "_importance_scores"

    np.save(new_file, results_collect)

    logging.info("--- Saved importance scores in {}".format(
        new_file + ".npy"
    ))