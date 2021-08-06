import torch
import torch.nn as nn
import json
from tqdm import trange
import numpy as np
import pandas as pd

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


def register_flips_(model, model_inputs, ranking, original_prediction, original_sentences, 
                    rows, results_dictionary, no_of_tokens, feat_attr_name):

    mask = torch.zeros(model_inputs["sentences"].shape).to(device)
               
    mask = mask.scatter_(1,  ranking[rows, no_of_tokens+1:], 1)

    model_inputs["sentences"] = (original_sentences.float() * mask.float()).long()

    masked_prediction, _ = model(**model_inputs)
    
    flips = (original_prediction.max(-1)[1] != masked_prediction.max(-1)[1]).nonzero()

    if flips.nelement() != 0: 

        for indx in flips:

            annotation_id = model_inputs["annotation_id"][indx]
            
            if feat_attr_name not in results_dictionary[annotation_id].keys():

                results_dictionary[annotation_id][feat_attr_name] = (no_of_tokens + 1) / model_inputs["lengths"][indx].item() 
                
            else:
                
                pass




def conduct_experiments_(model, data, save_path, data_split_name):

    """
        Info: computes the average fraction of tokens required to cause a decision flip (prediction change)
        Input:
            model : pretrained model
            data : torch.DataLoader loaded data
            save_path : path to save the results
        Output:
            saves the results to a json file under save_path
    """

    pbar = trange(
        len(data) * data.batch_size, 
        desc=f"running experiments for fraction of tokens on -> {data_split_name}", 
        leave=True
    )
    

    flip_results = {}

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

        
        {flip_results.update({x:{}}) for x in inputs["annotation_id"]}

        original_prediction, attentions =  model(**inputs)

        original_prediction.max(-1)[0].sum().backward(retain_graph = True)

        #embedding gradients
        embed_grad = model.bert_model.model.embeddings.word_embeddings.weight.grad
        g = embed_grad[inputs["sentences"].long()][:,:max(inputs["lengths"])]

        # cutting to length to save time
        attentions = attentions[:,:max(inputs["lengths"])]
        query_mask = inputs["query_mask"][:,:max(inputs["lengths"])]

        em = model.bert_model.model.embeddings.word_embeddings.weight[inputs["sentences"].long()][:,:max(inputs["lengths"])]

        gradients = (g* em).sum(-1).abs() * query_mask.float()

        integrated_grads = model.integrated_grads(
                original_grad = g, 
                original_pred = original_prediction.max(-1),
                **inputs    
        )

        # normalised integrated gradients of input
        normalised_ig = model.normalise_scores(integrated_grads * query_mask.float(), inputs["sentences"][:, :max(inputs["lengths"])])

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

              
        top_rand = torch.randn(attentions.shape).to(device)
        # softmaxing due to negative random weights
        # therefore we receive also negative values and as such
        # the pad and unwanted tokens need to be converted to -inf 
        top_rand = torch.masked_fill(top_rand, model.normalised_mask, float("-inf"))

        top_rand = torch.topk(top_rand, k = attentions.size(1))[1].to(device)

        normalised_grads = torch.topk(normalised_grads, k = normalised_grads.size(1))[1].to(device)
            
        normalised_attentions = torch.topk(normalised_attentions, k = normalised_attentions.size(1))[1].to(device)
        
        normalised_attention_grads = torch.topk(normalised_attention_grads, k = normalised_attention_grads.size(1))[1].to(device)

        normalised_ig = torch.topk(normalised_ig, k = normalised_ig.size(1))[1].to(device)

        ### lets speed it up and search every 5\%
        maximum = max(inputs["lengths"])
        increments =  torch.round(maximum.float() * 0.05).int()
        ## ensure there is at least one for increments
        increments = max(1,increments)
        
        rows = torch.arange(inputs["sentences"].size(0)).long().to(device)

        original_sentences = inputs["sentences"].clone().detach()
        

        used_for_iter = {
            "attention" : normalised_attentions,
            "gradients" : normalised_grads,
            "scaled attention" : normalised_attention_grads,
            "ig" : normalised_ig, 
            "random" : top_rand,
        }

        with torch.no_grad():

            for no_of_tokens in range(0,maximum+increments, increments):

                for feat_name, feat_score in used_for_iter.items():

                    register_flips_(
                        model = model, 
                        model_inputs = inputs, 
                        ranking = feat_score, 
                        original_prediction = original_prediction, 
                        original_sentences = original_sentences, 
                        rows = rows, 
                        results_dictionary = flip_results, 
                        no_of_tokens = no_of_tokens, 
                        feat_attr_name = feat_name
                    )


        pbar.update(data.batch_size)
        pbar.refresh()

    ### if we didnt register any flips for particular instances
    ## it means we reached the max so fraction of is 1.
    for annot_id in flip_results.keys():

        for feat_name in used_for_iter.keys():

            if feat_name not in flip_results[annot_id]:

                flip_results[annot_id][feat_name] = 1.
    

    if args["saliency_scorer"]: sal_scorer = args["saliency_scorer"] + "-"
    else: sal_scorer = ""

    with open(f"{save_path}{sal_scorer}{data_split_name}-fraction-of.json", "w") as file:

        json.dump(
            flip_results,
            file,
            indent = 4
        )

    summary = pd.DataFrame(flip_results).mean(axis=1).to_dict()

    with open(f"{save_path}{sal_scorer}{data_split_name}-fraction-of-summary.json", "w") as file:

        json.dump(
            summary,
            file,
            indent = 4
        )
    
    return