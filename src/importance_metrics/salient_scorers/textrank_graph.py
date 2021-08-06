import numpy as np
import math 
import pandas as pd
from tqdm import trange, tqdm

import json

from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(25)


import config.cfg
from config.cfg import AttrDict
from transformers import AutoTokenizer

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))


def gg_text(inpt, embeds):

    local_vocab = set(inpt)

    if 0 in local_vocab: ## remove pad
        
        local_vocab.remove(0)

    local_vocab = dict(enumerate(local_vocab))

    encoded_embeds = embeds[inpt]

    sim_matr = np.ones([len(local_vocab), len(local_vocab)])

    for i in range(len(local_vocab)):
        for j in range(len(local_vocab)):
            if i != j:

                sim_matr[i][j] = cosine_similarity(encoded_embeds[i].reshape(1,768), encoded_embeds[j].reshape(1,768))[0,0]


    d = 0.85 # damping coefficient, usually is .85
    min_diff = 1e-5 # convergence threshold
    steps = 20 # iteration steps

    # Get normalized matrix
    g = sim_matr

    # Initialization for weight(pagerank value)
    pr = np.array([1] * len(local_vocab))

    # Iteration for TEXTGRAPH
    previous_pr = 0
    for epoch in range(steps):
        pr = (1-d) + d * np.dot(g, pr)*1e-1
        if abs(previous_pr - sum(pr))  < min_diff:
            break
        else:
            previous_pr = sum(pr)


    scores = dict(enumerate(pr))
    voc_score = {v: scores[k] for k,v in local_vocab.items()}
    voc_score[0] = 0.

    scored_seq  = [voc_score[j] for j in inpt]

    return scored_seq

import logging

def text_grapher(data, tokenizer = None, extract_rationales = False):
    
    import config.cfg
    from config.cfg import AttrDict

    with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
        args = AttrDict(json.load(f))
        
    """ 
    text-rank scorer
    returns the text-rank scores for words in the vocabulary  
    from the training set
    """

    embeds = np.load("bert_embeds/" + args.model_abbreviation + "-embeds.npy")

    data["text__"] = data.text.apply(lambda x : x["input_ids"])

    logging.info("calculating text graph scores")
    
    tqdm.pandas(desc = "calculating text-graph scores", position = 0, leave = True)

    data["salient_scores"] = data.text__.progress_apply(
        lambda x : gg_text(x, embeds = embeds),
    )

    logging.info("extracted textgraph scores")

    return {"scored_data":data.drop(columns = "text__")}