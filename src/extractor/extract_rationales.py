
import torch
import json 
import glob 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import logging

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))
    
from src.models.bert import bert
from src.extractor.rationalizer import register_importance_, extractor_
from src.importance_metrics.imp_ex import extract_importance_scores_

from src.extractor.thresholders import thresholders

class extractor():

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

        current_model = glob.glob(args["save_path"] + sal_scorer + args["model_abbreviation"] + "*.pt")[0]
        
        self.model = bert(masked_list=[0,101,102], output_dim = output_dims)

        logging.info("Loaded model -- {}".format(current_model))

        # loading the trained model
        self.model.load_state_dict(torch.load(current_model, map_location=device))

        self.model.to(device)

        logging.info("Loaded model -- {}".format(current_model))

    def extract_importance(self, data):
        

        # create train, dev, test set with the attention and gradient rationales
        for key, loader in {"test":data.test_loader}.items():
            
            extract_importance_scores_(model = self.model, data = loader, 
                                    tokenizer = data.tokenizer, key = key)


        # empty cache - model no longer needed
        del self.model
        del data
            

    def _extract_rationales(self, data):

         # create train, dev, test set with the attention and gradient rationales
        for data_split_name, dataloader in {"train":data.train_loader, "dev":data.dev_loader, "test":data.test_loader}.items():

            register_importance_(
                model = self.model,
                data = dataloader,
                data_split_name = data_split_name
            )

            extractor_(
                data_as_df = data.return_as_dfs_(
                    data_split_name = data_split_name
                ),
                data_split_name = data_split_name,
                tokenizer = data.tokenizer,
                thresholder_name="topk"
            )

            extractor_(
                data_as_df = data.return_as_dfs_(
                    data_split_name = data_split_name
                ),
                data_split_name = data_split_name,
                tokenizer = data.tokenizer,
                thresholder_name="contigious"
            )
        # empty cache - model no longer needed
        del self.model
        del data

        return
        