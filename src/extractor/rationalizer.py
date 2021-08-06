import torch
import torch.nn as nn
import json
from src.extractor.thresholders import thresholders
from tqdm import trange
import numpy as np
import os 

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if args.dataset == "multirc": from src.utils.assistant import wpiece2word_r as wpiece2word
else: from src.utils.assistant import wpiece2word

def register_importance_(model, data, data_split_name):

    if args["saliency_scorer"] is None: sal_scorer = ""
    else: sal_scorer = args["saliency_scorer"] + "_"

    pbar = trange(len(data) * data.batch_size, desc=f"Registering importance scores for {data_split_name}", leave=True)

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        "importance_scores",
        ""
    )

    os.makedirs(fname, exist_ok = True)

    scorenames = f"{fname}{data_split_name}-{sal_scorer}importance_scores.npy"

    ## check if importance scores exist first to avoid unecessary calculations
    if os.path.exists(scorenames):

        print(f"importance scores already saved in -> {scorenames}")

        return

    nn.deterministic = True
    torch.backends.cudnn.benchmark = False
        

    torch.manual_seed(25)
    torch.cuda.manual_seed(25)
    np.random.seed(25)
    
    feature_attribution = {}

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
            "retain_gradient" : True,
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

        for _i_ in range(attentions.size(0)):
        
            annotation_id = inputs["annotation_id"][_i_]
            ## storing feature attributions
            feature_attribution[annotation_id] = {
                "attention" : normalised_attentions[_i_].cpu().detach().numpy(),
                "gradients" : normalised_grads[_i_].cpu().detach().numpy(),
                "ig" : normalised_ig[_i_].cpu().detach().numpy(),
                "scaled attention" : normalised_attention_grads[_i_].cpu().detach().numpy()
            }

        pbar.update(data.batch_size)

    print(f"feature attribution scores stored in -> {scorenames}")

    ## save them
    np.save(scorenames, feature_attribution)


    pbar.update(data.batch_size)


    return

def extractor_(data_as_df, data_split_name, tokenizer, thresholder_name):


    if args["saliency_scorer"] is None: sal_scorer = ""
    else: sal_scorer = args["saliency_scorer"] + "_"

    
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
    
    ## get the thresholder fun
    thresholder = getattr(thresholders, thresholder_name)

    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        thresholder_name,
        ""
        
    )

    os.makedirs(fname, exist_ok = True)
    ## filter only relevant parts in our dataset
    data = data_as_df[["text", "annotation_id", "exp_split", "label", "label_id"]]

    annotation_text = dict(data[["annotation_id", "text"]].values)

    del data["text"]

    ## time to register rationales
    for feature_attribution in {"attention", "gradients", "ig", "scaled attention"}:
        
        temp_registry = {}

        pbar = trange(
            len(annotation_text), 
            desc=f"Creating rationales for {feature_attribution} -> {thresholder_name} -> {data_split_name}", 
            leave=True
        )

        for annotation_id, sequence_text in annotation_text.items():

            temp_registry[annotation_id] = {}

            ## check if there is any padding which could affect our process and remove
            sequence_text = np.asarray(sequence_text)
            sos_eos = np.where(sequence_text == tokenizer.sep_token_id)[0]
            seq_length = sos_eos[0]

            full_doc = tokenizer.convert_ids_to_tokens(sequence_text[1:seq_length])
            full_doc = tokenizer.convert_tokens_to_string(full_doc)
            
            if args.query:

                query_end = sos_eos[1]
            
                query = tokenizer.convert_ids_to_tokens(sequence_text[seq_length + 1:query_end])
                query = tokenizer.convert_tokens_to_string(query)

            sequence_importance = importance_scores[annotation_id][feature_attribution][:seq_length]
            ## zero out cls and sep
            sequence_importance[0] = float("-inf")
            sequence_importance[-1] = float("-inf")
            sequence_text = sequence_text[:seq_length]

            ## untokenize sequence and sequence importance scores
            sequence_text, sequence_importance = wpiece2word(
                tokenizer = tokenizer, 
                sentence = sequence_text, 
                weights = sequence_importance
            )

            rationale_indxs = thresholder(
                scores = sequence_importance, 
                original_length = seq_length - 2,
                rationale_length = args["rationale_length"]
            )

            rationale = sequence_text[rationale_indxs]

            temp_registry[annotation_id]["text"] = " ".join(rationale)
            temp_registry[annotation_id]["full text doc"] = full_doc

            if args.query: 
                
                temp_registry[annotation_id]["query"]  = query

            pbar.update(1)

        if args.query:
            
            data["document"] = data.annotation_id.apply(lambda x : temp_registry[x]["text"])
            data["query"] = data.annotation_id.apply(lambda x : temp_registry[x]["query"])

        else:

            data["text"] = data.annotation_id.apply(lambda x : temp_registry[x]["text"])

        data["full text doc"] = data.annotation_id.apply(lambda x : temp_registry[x]["full text doc"])

        if feature_attribution == "scaled attention": feature_attribution = "scaled_attention"

        fname = os.path.join(
            os.getcwd(),
            args["extracted_rationale_dir"],
            thresholder_name,
            f"{feature_attribution}-{sal_scorer}{data_split_name}.csv"
        )

        print(f"saved in -> {fname}")

        data.to_csv(fname)


    return