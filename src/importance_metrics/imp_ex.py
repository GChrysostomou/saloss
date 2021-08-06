import torch
import torch.nn as nn
import math 
import json
from src.extractor.thresholders import thresholders
from tqdm import trange
import numpy as np
import logging 

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if args.dataset == "multirc": from src.utils.assistant import wpiece2word_r as wpiece2word
else: from src.utils.assistant import wpiece2word

def extract_importance_scores_(model, data, tokenizer, key):

    pbar = trange(len(data) * data.batch_size, desc='Extracting importance ' + key, leave=True)

    nn.deterministic = True
    torch.backends.cudnn.benchmark = False
        

    torch.manual_seed(25)
    torch.cuda.manual_seed(25)
    np.random.seed(25)
    
    results = []

    # get the thresholder
    thresholder = getattr(thresholders(), args["thresholder"])
    
    for batch in data:
            
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

        if len(batch) < 8: inputs["salient_scores"] = None
        else: 
            
            if len(batch) == 10:

                sals = [x.float() for x in batch[-3:]]

                inputs["salient_scores"] = torch.stack(sals).transpose(0,1).transpose(1,-1).to(device)

            else:

                inputs["salient_scores"] = batch[7].to(device)

        if (args["saliency_scorer"] and args.tfidf_asvec == False):
            
            ### normalising the salient scores
            if args["saliency_scorer"] != "return_all":

                inputs["salient_scores"] = torch.masked_fill(inputs["salient_scores"], ~inputs["attention_mask"].bool(), float("-inf"))
                inputs["salient_scores"] = torch.softmax(inputs["salient_scores"], dim = -1)

            else:
                
                inputs["salient_scores"] = torch.masked_fill(
                        inputs["salient_scores"],
                        ~inputs["attention_mask"].repeat(3, 1, 1).transpose(0,1).transpose(1,2).bool(),  
                        float("-inf")
                )

                inputs["salient_scores"] = torch.softmax(inputs["salient_scores"], dim = 1)
                                        
        assert inputs["sentences"].size(0) == len(inputs["labels"]), "Error: batch size for item 1 not in correct position"

        
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

        integrated_grads = model.integrated_grads(
                original_grad = g, 
                original_pred = yhat.max(-1),
                **inputs    
        )

        # normalised gradients of input
        normalised_grads = model.normalise_scores(gradients, inputs["sentences"][:, :max(inputs["lengths"])])

        # normalised integrated gradients of input
        normalised_ig = model.normalise_scores(integrated_grads * query_mask.float(), inputs["sentences"][:, :max(inputs["lengths"])])

        # normalised attention
        normalised_attentions = model.normalise_scores(attentions * query_mask.float(), inputs["sentences"][:, :max(inputs["lengths"])])

        # retrieving attention*attention_grad
        attention_gradients = model.weights_or.grad[:,:,0,:].mean(1)[:,:max(inputs["lengths"])]
        
        normalised_attention_grads =  model.normalise_scores(attentions * attention_gradients * query_mask.float(), inputs["sentences"][:, :max(inputs["lengths"])])

        # softmaxing due to negative attention gradients 
        # therefore we receive also negative values and as such
        # the pad and unwanted tokens need to be converted to -inf 
        normalised_attention_grads = torch.masked_fill(normalised_attention_grads, model.normalised_mask, float("-inf"))
        
        for j, item in enumerate(inputs["annotation_id"]):

            results.append({
                "annotation_id": item,
                "attention": normalised_attentions[j].detach().cpu().numpy(),
                "gradients": normalised_grads[j].detach().cpu().numpy(),
                "attention-gradients": normalised_attention_grads[j].detach().cpu().numpy(),
                "ig": normalised_ig.detach()[j].cpu().numpy()
            })
        
        pbar.update(data.batch_size)

    np.save("ner_pos_tag_importance_results/saloss/" + args.dataset + "_importance_scores.npy", results)

    logging.info("extracted importance")
    torch.cuda.empty_cache()
