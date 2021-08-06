#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math 
import pandas as pd
from collections import Counter
import logging
from src.extractor.thresholders import thresholders
from src.utils.assistant import query_masker
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from src.importance_metrics.salient_scorers import linguist
from sklearn.feature_selection import  SelectKBest, chi2

import config.cfg
from config.cfg import AttrDict

def chi_scorer(data,tokenizer = None, train_vec = None, extract_rationales = False):

    with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
        args = AttrDict(json.load(f))


    """ 
    Tfidf scorer
    returns the tfidf scores for words in the vocabulary  
    from the training set
    """

    if extract_rationales:
    
        data["text__"] = data.text

    else:

        data["text__"] = data.text.apply(lambda x : " ".join(tokenizer.convert_ids_to_tokens(x["input_ids"])))


    if train_vec is None:
        # we want it to fit only on train
        vectorizer = TfidfVectorizer()
        chisquarer = SelectKBest(chi2, k="all")

        if extract_rationales:

            vectorizer.fit(data.text__)

        else:
            
            logging.info("fitting tfidf/chisquared")

            vectorizer.fit(data[data.exp_split == "train"].text__)

    else: 

        logging.info("preloading train_vec for tfidf/chisquared")
        vectorizer =  train_vec["tfidf_vectorizer"]
        chisquarer = train_vec["chisquarer"]
        

    tfidfs = vectorizer.transform(data.text__).toarray()

    if train_vec is None:

        chisquarer.fit(tfidfs, data.label)



    word2id = vectorizer.vocabulary_
    id2word = {v:k for k,v in word2id.items()}


    chi_scores = {}
    current_scores = chisquarer.scores_

    for indx, _ in id2word.items():

        chi_scores[indx] = current_scores[indx]

    chi_score_list = []

    for i in range(len(tfidfs)):
        
        # index sentences according to vocabulary entry
        # if the word does not exist place a -1 to recognise unkowns
        # as they will receive a 0 tfidf value    
        text = np.asarray(data.text__.values[i].split())

        # remove from text the linguistic fetures if any
        if args["linguistic_feature"]:
            
            text = np.asarray([x.split("_")[0] for x in text])

        indexed = [word2id[w] if w in word2id else -1 for w in text]
        
        # chiscore value if word exists in id2word
        # if its unkown then it receives a 0 word
        chi_score = np.asarray([chi_scores[indx] if indx in id2word else 0 for indx in indexed])
        
        chi_score_list.append(list(chi_score))


    data["salient_scores"] = chi_score_list

    logging.info("extracted chi_scores rationales")

    return {"scored_data":data.drop(columns = "text__"), "vectorizer": {"tfidf_vectorizer":vectorizer, "chisquarer": chisquarer}}
 
        

   

        
            