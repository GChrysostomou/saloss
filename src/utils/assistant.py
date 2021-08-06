# """
# contains functions for helping with the loading, processing, description and preparation of the datasets
# """

import pandas as pd
import numpy as np
import math
import json 

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))


def describe_data_stats(path = str):
    """ 
    returns dataset statistics such as : 
                                        - number of documens
                                        - average sequence length
                                        - average query length (if QA)
    """
    # ensure that datapath ends with "/"
    if path[-1] == "/":pass 
    else: path = path + "/"
    
    descriptions = {"train":{}, "dev":{}, "test":{}}

    train = pd.read_csv(path + "train.csv")
    dev = pd.read_csv(path + "dev.csv")
    test = pd.read_csv(path + "test.csv")

    if args["query"]:

        train = train.rename(columns = {"document":"text"})
        dev = dev.rename(columns = {"document":"text"})
        test = test.rename(columns = {"document":"text"})

    # load data and save them in descriptions dictionary
    
    descriptions["train"]["number_of_docs"] =len(train.text.values)
    descriptions["train"]["ave_doc_length"] =  math.ceil(np.asarray([len(x.split()) for x in train.text.values]).mean())

    descriptions["dev"]["number_of_docs"] =len(dev.text.values)
    descriptions["dev"]["ave_doc_length"] =  math.ceil(np.asarray([len(x.split()) for x in dev.text.values]).mean())
    
    descriptions["test"]["number_of_docs"] =len(test.text.values)
    descriptions["test"]["ave_doc_length"] =  math.ceil(np.asarray([len(x.split()) for x in test.text.values]).mean())

    majority_class = np.unique(np.asarray(test.label.values), return_counts = True)
   
    descriptions["train"]["label_distribution"] = {str(k):v for k, v in dict(np.asarray(majority_class).T).items()}
    descriptions["train"]["majority_class"] =  round(majority_class[-1].max() / majority_class[-1].sum() * 100,2)

    if args["query"]:

        descriptions["train"]["ave_query_length"] =  math.ceil(np.asarray([len(x.split()) for x in train["query"].values]).mean())
        descriptions["dev"]["ave_query_length"] =  math.ceil(np.asarray([len(x.split()) for x in dev["query"].values]).mean())
        descriptions["test"]["ave_query_length"] =  math.ceil(np.asarray([len(x.split()) for x in test["query"].values]).mean())
    
    return descriptions


def bert_padder(x, max_len):

    """
    pads documents to required length
    """

    if len(x) < max_len:
    
        x += [0]*(int(max_len) - len(x))
  
    return x

def query_masker(x, sep_token = 102):
    """
    returns a masked that indicates which words belongs to 
    the main text (1) and which to the query (0)
    """


    all_length = len(x)

    if sep_token in x:

        first_sep = x.index(sep_token) + 1 # to add sep

        return np.asarray([0]*first_sep + [1]*(all_length-first_sep))

    else:

        return np.asarray([1]*all_length)

def encode_it(tokenizer, max_length, *arguments):

    """
    returns token type ids, padded doc and 
    """

    if len(arguments) > 1:

        dic = tokenizer.encode_plus(arguments[0], arguments[1],
                                        add_special_tokens = True,
                                        max_length = max_length,
                                        padding = 'max_length',
                                        return_token_type_ids = True,
                                        truncation = True)

    else:
  
        dic = tokenizer.encode_plus(arguments[0],
                                        add_special_tokens = True,
                                        max_length = max_length,
                                        padding = 'max_length',
                                        return_token_type_ids = True,
                                        truncation = True)
       
    return dic


# import torch

# def wpiece2word(tokenizer, sentence, weights, print_err = False):    
    
#     """
#     converts word-piece ids to words and
#     importance scores/weights for word-pieces to importance scores/weights
#     for words by aggregating them
#     """

#     tokens = tokenizer.convert_ids_to_tokens(sentence)
#     strings = tokenizer.convert_tokens_to_string(tokens)

#     tokens = " ".join(tokens)

#     unique_words = {}
#     new_weights = {}
  
#     for i in range(len(tokens.split())):
        
#         if i < len(tokens.split())-1:
            
#             w = tokens.split()[i]
#             next_w = tokens.split()[i+1]
          
#             if "#" in next_w:

#                 if "#" not in w:
                    
#                     rec = i

#                     unique_words[rec] = w
#                     length = 1
#                     new_weights[rec] = weights[i].item()
                    
#             if "#" in w:
                
#                 unique_words[rec] += w.split("#")[-1] 
#                 new_weights[rec] += weights[i].item()
#                 length += 1
                
#         else:
            
#             w = tokens.split()[i]
            
#             if "#" in w:
                
#                 unique_words[rec] += w.split("#")[-1] 
#                 new_weights[rec] += weights[i].item()
#                 length += 1
                
#             else:
                
#                 pass
   
#     returned_scores = torch.zeros(len(strings.split()))
#     unique_words = {v:k for k,v in unique_words.items()}
 
#     for i, w in enumerate(strings.split()):
   
#         if w in tokens.split():

#             returned_scores[i] = weights[i]

#         else:

#             idx = unique_words[w] 
#             returned_scores[i] = new_weights[idx]

#     return np.asarray(strings.split()), returned_scores


import torch

def wpiece2word(tokenizer, sentence, weights, print_err = False):  

    """
    converts word-piece ids to words and
    importance scores/weights for word-pieces to importance scores/weights
    for words by aggregating them
    """

    tokens = tokenizer.convert_ids_to_tokens(sentence)

    new_words = {}
    new_score = {}

    position = 0

    for i in range(len(tokens)):

        word = tokens[i]
        score = weights[i].clone().detach().data

        if "##" not in word:
            
            position += 1
            new_words[position] = word
            new_score[position] = score
            
        else:
            
            new_words[position] += word.split("##")[1]
            new_score[position] += score

    return np.asarray(list(new_words.values())), torch.tensor(list(new_score.values()))


import re
def wpiece2word_r(tokenizer, sentence, weights, print_err = False):    
    
    """
    converts word-piece ids to words and
    importance scores/weights for word-pieces to importance scores/weights
    for words by aggregating them
    """

    tokens = tokenizer.convert_ids_to_tokens(sentence)
    strings = tokenizer.convert_tokens_to_string(tokens)
    strings = "<s> ".join(strings.split("<s>")).split("</s>")[0] + " </s>"

    unique_words = {}
    new_weights = {}

    import re

    for i in range(len(tokens)):
        
        if i < 2:
            
            w = tokens[i]
            
            if "Ġ" in w:
                    
                rec = i

                unique_words[rec] = re.sub("Ġ", "", w)
                length = 1
                new_weights[rec] = weights[i].item()
                
            else:
                
                rec = i

                unique_words[rec] = w
                length = 1
                new_weights[rec] = weights[i].item()
            
    
        elif i < len(tokens)-1:
            
            w = tokens[i]
            
            if "Ġ" in w:
                
                rec = i
                
                unique_words[rec] = re.sub("Ġ", "", w)
                length = 1
                new_weights[rec] = weights[i].item()
                
            elif w == "</s>":
        
                rec = i

                unique_words[rec] = w
                length = 1
                new_weights[rec] = weights[i].item()
                
            else:
                
                if tokens[i - 1] == "</s>":
                    
                    rec = i
                    
                    unique_words[rec] = w
                    length = 1
                    new_weights[rec] = weights[i].item()
                    
                else:
                
                    unique_words[rec] += w
                    length += 1
                    new_weights[rec] += weights[i].item()
                
        else:
            
            w = tokens[i]
            
            rec = i

            unique_words[rec] = w
            length = 1
            new_weights[rec] = weights[i].item()
                    
    returned_scores = torch.zeros(len(unique_words))
    unique_words = {v:k for k,v in unique_words.items()}

    sentence = []

    for _j, (word, indx) in enumerate(unique_words.items()):
        
        sentence.append(word)
        returned_scores[_j] = new_weights[indx]
        

    returned_scores = returned_scores[:len(sentence)]
        
    return np.asarray(sentence), returned_scores
