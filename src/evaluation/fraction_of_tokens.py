
import torch
import pandas as pd
import json 
import glob 
import os
import spacy
from tqdm import tqdm
from tqdm import trange 
import numpy as np
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

from src.models.bert import bert
from src.evaluation.built_in.scorer import conduct_experiments_
from src.evaluation.built_in.importance_extractor import extractor
from src.extractor.rationalizer import register_importance_


class evaluate():

    """
    Class that contains method of rationale extraction as in:
        saliency scorer and thresholder approach
    Saves rationales in a csv file with their dedicated annotation_id 
    """

    def __init__(self, output_dims = 2):
        
        """
        loads and holds a pretrained model
        """

        if args["saliency_scorer"] is None: sal_scorer = ""
        else: sal_scorer = args["saliency_scorer"] + "_"

        assert args["saliency_scorer"] in {None, "tfidf", "textrank", "chisquared"}

        current_model = glob.glob(args["save_path"] + sal_scorer + args["model_abbreviation"] + "*.pt")[0]
        
        self.model = bert(masked_list=[0,101,102], output_dim = output_dims)

        logging.info("Loaded model -- {}".format(current_model))

        # loading the trained model
        self.model.load_state_dict(torch.load(current_model, map_location=device))

        self.model.to(device)
        
        self.results_dir = args["evaluation_dir"]

    def extract_importance_metrics(self, dataloader):

        extractor(
            model = self.model, 
            data = dataloader,
            save_path = self.results_dir
        )

    def fraction_of_experiments_(self, data):


        for data_split, dataloader in {"test" : data.test_loader , "dev" : data.dev_loader}.items():
            
            register_importance_(
                model = self.model,
                data = dataloader,
                data_split_name = data_split
            )

            conduct_experiments_(
                model = self.model, 
                data = dataloader,
                save_path = self.results_dir,
                data_split_name = data_split
            )

        return 


