import torch
import torch.nn as nn
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

def extractor_(model, data, tokenizer, key):

    pbar = trange(len(data) * data.batch_size, desc='Extracting rationales ' + key, leave=True)

    nn.deterministic = True
    torch.backends.cudnn.benchmark = False
        

    torch.manual_seed(25)
    torch.cuda.manual_seed(25)
    np.random.seed(25)
    
    attention_rationales = {}
    attention_rationales["importance_metric"] = "attention"
    attention_rationales["annotation_id"] = []
    if args["query"]: 
            attention_rationales["document"] = []
            attention_rationales["query"] = []
    else:        attention_rationales["text"] = []
    attention_rationales["gold_label"] = []
    attention_rationales["thresholder"] = args["thresholder"]
    attention_rationales["saliency_scorer"] = args.saliency_scorer

    gradient_rationales = {}
    gradient_rationales["importance_metric"] = "gradient"
    gradient_rationales["annotation_id"] = []
    if args["query"]: 
            gradient_rationales["document"] = []
            gradient_rationales["query"] = []
    else:        gradient_rationales["text"] = []
    gradient_rationales["gold_label"] = []
    gradient_rationales["thresholder"] = args["thresholder"]
    gradient_rationales["saliency_scorer"] = args.saliency_scorer


    attention_grads = {}
    attention_grads["importance_metric"] = "attention-gradients"
    attention_grads["annotation_id"] = []
    if args["query"]: 
            attention_grads["document"] = []
            attention_grads["query"] = []
    else:        attention_grads["text"] = []
    attention_grads["gold_label"] = []
    attention_grads["thresholder"] = args["thresholder"]
    attention_grads["saliency_scorer"] = args.saliency_scorer

    ig_set = {}
    ig_set["importance_metric"] = "integrated-gradients"
    ig_set["annotation_id"] = []
    if args["query"]: 
            ig_set["document"] = []
            ig_set["query"] = []
    else:        ig_set["text"] = []
    ig_set["gold_label"] = []
    ig_set["thresholder"] = args["thresholder"]
    ig_set["saliency_scorer"] = args.saliency_scorer

    # get the thresholder
    thresholder = getattr(thresholders(), args["thresholder"])
    
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

        # retrieving rationales from ig
        for nig, igg in enumerate(normalised_ig):

            sequence, word_ig = wpiece2word(tokenizer = tokenizer, sentence = inputs["sentences"][nig][:inputs["lengths"][nig]], weights = igg)

            # -2 for length due to sep and cls tokens
            indxs = thresholder(word_ig.detach().data, original_length = inputs["lengths"][nig].item() - 2,rationale_length = args["rationale_length"])

            rationale = sequence[indxs]

            ig_set["annotation_id"].append(inputs["annotation_id"][nig])
            if args["query"]:   
                ig_set["document"].append(" ".join(rationale))
                local_sentence = inputs["sentences"][nig][:inputs["lengths"][nig]]
                if args.dataset == "multirc": query_indxs = ((local_sentence == 2).nonzero())[1]
                else: query_indxs = ((local_sentence == 103).nonzero())[0]
                query = local_sentence[query_indxs[0]+1:-1]
                query = tokenizer.convert_ids_to_tokens(query)
                query = tokenizer.convert_tokens_to_string(query)
                ig_set["query"].append(query)
            else:   ig_set["text"].append(" ".join(rationale))
            ig_set["gold_label"].append(inputs["labels"][nig].item())

        # retrieving rationales from attentions
        for no, att in enumerate(normalised_attentions):

            sequence, word_attentions = wpiece2word(tokenizer = tokenizer, sentence = inputs["sentences"][no][:inputs["lengths"][no]], weights = att)

            # -2 for length due to sep and cls tokens
            indxs = thresholder(word_attentions.detach().data, original_length = inputs["lengths"][no].item() - 2,rationale_length = args["rationale_length"])

            rationale = sequence[indxs]

            attention_rationales["annotation_id"].append(inputs["annotation_id"][no])
            if args["query"]:   
                attention_rationales["document"].append(" ".join(rationale))
                local_sentence = inputs["sentences"][no][:inputs["lengths"][no]]
                if args.dataset == "multirc": query_indxs = ((local_sentence == 2).nonzero())[1]
                else: query_indxs = ((local_sentence == 103).nonzero())[0]
                query = local_sentence[query_indxs[0]+1:-1]
                query = tokenizer.convert_ids_to_tokens(query)
                query = tokenizer.convert_tokens_to_string(query)
                attention_rationales["query"].append(query)
            else:   attention_rationales["text"].append(" ".join(rationale))
            attention_rationales["gold_label"].append(inputs["labels"][no].item())
        
        # retrieving rationales from gradients
        for nu, grad in enumerate(normalised_grads):

            sequence, word_gradients = wpiece2word(tokenizer = tokenizer, sentence = inputs["sentences"][nu][:inputs["lengths"][nu]], weights = grad)
            
            # -2 for length due to sep and cls tokens
            indxs = thresholder(word_gradients.detach().data, original_length = inputs["lengths"][nu].item() - 2, rationale_length = args["rationale_length"])

            rationale = sequence[indxs]

            gradient_rationales["annotation_id"].append(inputs["annotation_id"][nu])
            if args["query"]:   
                gradient_rationales["document"].append(" ".join(rationale))
                local_sentence = inputs["sentences"][nu][:inputs["lengths"][nu]]
                if args.dataset == "multirc": query_indxs = ((local_sentence == 2).nonzero())[1]
                else: query_indxs = ((local_sentence == 103).nonzero())[0]
                query = local_sentence[query_indxs[0]+1:-1]
                query = tokenizer.convert_ids_to_tokens(query)
                query = tokenizer.convert_tokens_to_string(query)
                gradient_rationales["query"].append(query)
            else:   gradient_rationales["text"].append(" ".join(rationale))
            gradient_rationales["gold_label"].append(inputs["labels"][nu].item())



        # retrieving rationales from attentions
        for ne, att_grad in enumerate(normalised_attention_grads):

            sequence, word_attentions = wpiece2word(tokenizer = tokenizer, sentence = inputs["sentences"][ne][:inputs["lengths"][ne]], weights = att_grad)

            # -2 for length due to sep and cls tokens
            indxs = thresholder(word_attentions.detach().data, original_length = inputs["lengths"][ne].item() - 2,rationale_length = args["rationale_length"])

            rationale = sequence[indxs]

            attention_grads["annotation_id"].append(inputs["annotation_id"][ne])
            if args["query"]:   
                attention_grads["document"].append(" ".join(rationale))
                local_sentence = inputs["sentences"][ne][:inputs["lengths"][ne]]
                if args.dataset == "multirc": query_indxs = ((local_sentence == 2).nonzero())[1]
                else: query_indxs = ((local_sentence == 103).nonzero())[0]
                query = local_sentence[query_indxs[0]+1:-1]
                query = tokenizer.convert_ids_to_tokens(query)
                query = tokenizer.convert_tokens_to_string(query)
                attention_grads["query"].append(query)
            else:   attention_grads["text"].append(" ".join(rationale))
            attention_grads["gold_label"].append(inputs["labels"][ne].item())

        pbar.update(data.batch_size)

    logging.info("extracted rationales")
    torch.cuda.empty_cache()

    return attention_rationales, gradient_rationales, attention_grads, ig_set
    