import torch
import torch.nn as nn
import json
from tqdm import trange
import numpy as np
import pandas as pd
from src.utils.assistant import batch_from_dict_
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


import os

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

    if args["saliency_scorer"]: sal_scorer = args["saliency_scorer"] + "-"
    else: sal_scorer = ""

    pbar = trange(
        len(data) * data.batch_size, 
        desc=f"running experiments for fraction of tokens on -> {data_split_name}", 
        leave=True
    )
    

    flip_results = {}

    ## now to create folder where results will be loaded from
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        "importance_scores",
        ""
    )

    os.makedirs(fname, exist_ok = True)

    scorenames = f"{fname}{data_split_name}-{sal_scorer}importance_scores.npy"

    ## check if importance scores exist first 
    if os.path.exists(scorenames):

        importance_scores = np.load(scorenames, allow_pickle = True).item()

    else:

        raise FileNotFoundError(f"importance scores not found in -> {scorenames}")

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
        
        
        top_rand = torch.randn(attentions.shape).to(device)
        # softmaxing due to negative random weights
        # therefore we receive also negative values and as such
        # the pad and unwanted tokens need to be converted to -inf 
        top_rand = model.normalise_scores((top_rand * inputs["query_mask"].float())[:, :max(inputs["lengths"])],  inputs["sentences"][:, :max(inputs["lengths"])])
        top_rand = torch.masked_fill(top_rand, model.normalised_mask, float("-inf"))

        ### lets speed it up and search every 5\%
        maximum = max(inputs["lengths"])
        increments =  torch.round(maximum.float() * 0.05).int()
        ## ensure there is at least one for increments
        increments = max(1,increments)
        
        rows = torch.arange(inputs["sentences"].size(0)).long().to(device)

        original_sentences = inputs["sentences"].clone().detach()
        

        with torch.no_grad():
            
            for feat_name in {"random", "attention", "gradients", "scaled attention", "ig"}:
                
                if feat_name != "random":

                    feat_score =  batch_from_dict_(
                            batch_data = inputs, 
                            metadata = importance_scores, 
                            target_key = feat_name,
                        )
                
                else:

                    feat_score = top_rand

                feat_rank = torch.topk(feat_score, k = feat_score.size(1))[1].to(device)

                for no_of_tokens in range(0,maximum+increments, increments):

                    register_flips_(
                        model = model, 
                        model_inputs = inputs, 
                        ranking = feat_rank, 
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

        for feat_name in {"random", "attention", "gradients", "scaled attention", "ig"}:

            if feat_name not in flip_results[annot_id]:

                flip_results[annot_id][feat_name] = 1.
    

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