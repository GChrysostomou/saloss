from tqdm import trange, tqdm
from src.utils.assistant import encode_it
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers 
from transformers import AutoTokenizer
import json

import os
import os.path
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from src.importance_metrics.salient_methods import salient_scorer


import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

class classification_dataholder():
    """
    class that holds our data, pretrained tokenizer and set sequence length 
    for a classification task
    """
    def __init__(self, path : str, b_size :int , for_rationale : bool = False, return_as_dfs :bool = False):
        
        assert type(b_size) == int
        

        self.batch_size = b_size
        self.for_rationale = for_rationale

        # ensure that datapath ends with "/"
        if path[-1] == "/":pass 
        else: path = path + "/"

        if args.retrain:

            if args["saliency_scorer"] is None: sal_scorer = ""
            else: sal_scorer = args["saliency_scorer"] + "_"

            path = f"{args.data_dir}{args.thresholder}/{args.importance_metric}-{sal_scorer}"
           
            logging.info(" ** Loading rationales on dataholder form {} ** ".format(path))

        """
        loads data for a classification task from preprocessed .csv 
        files in the dataset/data folder
        and returns three dataholders : train, dev, test
        """
        # load splits
        train = pd.read_csv(path + "train.csv")#.sample(frac = 0.01, random_state = 1)
        dev = pd.read_csv(path + "dev.csv")#.sample(frac = 0.01, random_state = 1)
        test = pd.read_csv(path + "test.csv")#.sample(frac = 0.01, random_state = 1)

        # # if our dataset is part of a context-query task
        # # find max len of both context and query

        if args.query:
            
            max_len = round(max([len(x.split()) for x in train.document.values])) + \
                        max([len(x.split()) for x in train["query"].values])
            max_len = round(max_len)

        else:
            
            max_len = round(max([len(x.split()) for x in train.text.values]))

        max_len = min(max_len, 512)

        # load the pretrained tokenizer
        pretrained_weights = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

        self.nu_of_labels = len(train.label.unique())

        if args.query:
            
            train["text"] = train.apply(lambda x: encode_it(self.tokenizer, 
                            max_len, x["document"], x["query"]), axis = 1)
            dev["text"] = dev.apply(lambda x: encode_it(self.tokenizer, 
                            max_len, x["document"], x["query"]), axis = 1)
            test["text"] = test.apply(lambda x: encode_it(self.tokenizer, 
                            max_len, x["document"], x["query"]), axis = 1)

            # used only on rationale extraction
            train["query_mask"] = train.text.transform(lambda x: list((np.asarray(x["token_type_ids"]) == 0).astype(int)))
            dev["query_mask"] = dev.text.transform(lambda x: list((np.asarray(x["token_type_ids"]) == 0).astype(int)))
            test["query_mask"] = test.text.transform(lambda x: list((np.asarray(x["token_type_ids"]) == 0).astype(int)))

        else:

            train["text"] = train.apply(lambda x: encode_it(self.tokenizer, 
                            max_len,  x["text"]), axis = 1)
            dev["text"] = dev.apply(lambda x: encode_it(self.tokenizer, 
                            max_len,  x["text"]), axis = 1)
            test["text"] = test.apply(lambda x: encode_it(self.tokenizer, 
                            max_len, x["text"]), axis = 1)

            # used only on rationale extraction
            train["query_mask"] = train.text.transform(lambda x:list((np.asarray(x["input_ids"]) != 0).astype(int)) )
            dev["query_mask"] = dev.text.transform(lambda x:list((np.asarray(x["input_ids"]) != 0).astype(int)))
            test["query_mask"] = test.text.transform(lambda x:list((np.asarray(x["input_ids"]) != 0).astype(int)))

        if (args.saliency_scorer and args.train == True):
            
            ## check if we allready have saved the scores
            score_names = args["data_dir"] + "test_scores_" + args["saliency_scorer"] + ".json"

            ## check if they are saved and load them
            if os.path.isfile(score_names):
                
                print("Loading {} scores from file -- {}".format(
                    args.saliency_scorer,
                    score_names
                ))

                logging.info("Loading {} scores from file -- {}".format(
                    args.saliency_scorer,
                    score_names
                ))

                ## for train
                with open(args.data_dir + "train_scores_" + args.saliency_scorer + ".json", 'r') as f: 
                    files = json.load(f)
                    scores = {x["annotation_id"]:x["salient_scores"] for x in files}
                
                
                train["salient_scores"] = train.annotation_id.apply(lambda x : scores[x])

                ## for dev
                with open(args.data_dir + "dev_scores_" + args.saliency_scorer + ".json", 'r') as f: 
                    files = json.load(f)
                    scores = {x["annotation_id"]:x["salient_scores"] for x in files}
                
                dev["salient_scores"] = dev.annotation_id.apply(lambda x : scores[x])
                


                ## for test
                with open(args.data_dir + "test_scores_" + args.saliency_scorer + ".json", 'r') as f: 
                    files = json.load(f)
                    scores = {x["annotation_id"]:x["salient_scores"] for x in files}
                
                test["salient_scores"] = test.annotation_id.apply(lambda x : scores[x])
               
            ## calculate if not saved
            else:

                scorer = getattr(salient_scorer(args.saliency_scorer), args.saliency_scorer)

                out = scorer(train, tokenizer = self.tokenizer)
                train = out["scored_data"]

                ## save train scores
                with open(args.data_dir + "train_scores_" + args.saliency_scorer + ".json", 'w') as outfile:
                    json.dump(train[["annotation_id", "salient_scores"]].to_dict("r"), outfile)

                if "vectorizer" in out: vec = out["vectorizer"]
                else: vec = None
                out = scorer(dev, tokenizer = self.tokenizer, train_vec = vec)
                dev = out["scored_data"]

                ## save dev scores
                with open(args.data_dir + "dev_scores_" + args.saliency_scorer + ".json", 'w') as outfile:
                    json.dump(dev[["annotation_id", "salient_scores"]].to_dict("r"), outfile, indent = 4)

                out = scorer(test, tokenizer = self.tokenizer, train_vec = vec)
                test = out["scored_data"]

                ## save test scores
                with open(args.data_dir + "test_scores_" + args.saliency_scorer + ".json", 'w') as outfile:
                    json.dump(test[["annotation_id", "salient_scores"]].to_dict("r"), outfile, indent = 4)

        train["token_type_ids"] = train.text.transform(lambda x:x["token_type_ids"])
        dev["token_type_ids"] = dev.text.transform(lambda x:x["token_type_ids"])
        test["token_type_ids"] = test.text.transform(lambda x:x["token_type_ids"])


        train["attention_mask"] = train.text.transform(lambda x:x["attention_mask"])
        train["text"] = train.text.transform(lambda x:x["input_ids"])
        train["lengths"] = train.attention_mask.apply(lambda x: sum(x))

        
        dev["attention_mask"] = dev.text.transform(lambda x:x["attention_mask"])
        dev["text"] = dev.text.transform(lambda x:x["input_ids"])
        dev["lengths"] = dev.attention_mask.apply(lambda x: sum(x))

        test["attention_mask"] = test.text.transform(lambda x:x["attention_mask"])
        test["text"] = test.text.transform(lambda x:x["input_ids"])
        test["lengths"] = test.attention_mask.apply(lambda x: sum(x))

        if return_as_dfs:

            self.df_version = {
                "train" : train,
                "dev" : dev,
                "test" : test
            }
        # sort by length

        train = train.sort_values("lengths", ascending = True)
        dev = dev.sort_values("lengths", ascending = True)
        test = test.sort_values("lengths", ascending = True)
        
        # prepare data-loaders for training

        columns = ["text", "lengths", "label", "annotation_id", "query_mask", 
                    "token_type_ids", "attention_mask"]

        if (args.saliency_scorer and args.train == True):

            columns = ["text", "lengths", "label", "annotation_id", "query_mask", 
                        "token_type_ids", "attention_mask", "salient_scores"]

        self.train_loader = DataLoader(train[columns].values.tolist(),
                                batch_size = self.batch_size,
                                shuffle = False,
                                pin_memory = False)

        self.dev_loader = DataLoader(dev[columns].values.tolist(),
                                batch_size = self.batch_size,
                                shuffle = False,
                                pin_memory = False)

        self.test_loader = DataLoader(test[columns].values.tolist(),
                                batch_size = self.batch_size,
                                shuffle = False,
                                pin_memory = False)  
                                
        del train
        del dev
        del test


    def return_as_dfs_(self, data_split_name):

        return self.df_version[data_split_name]