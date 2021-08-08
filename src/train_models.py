"""
This module contains functions that:
train_and_save : function that will train on user defined runs
                 the predefined model on a user selected dataset. 
                 Will save details of model training and development performance
                 for each run and epoch. Will save the best model for each run
test_predictive_performance : function that obtains a trained model
                              on a user defined task and model. Will 
                              test on the test-dataset and will keep 
                              the best performing model, whilst also returning
                              statistics for model performances across runs, mean
                              and standard deviations
"""


import torch
import torch.nn as nn
from torch import optim
import json 
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from transformers.optimization import AdamW
import logging
import config.cfg

import gc


import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from src.models.bert import bert
from src.utils.train_test import train_model, test_model

def train_and_save(train_data_loader, dev_data_loader, for_rationale = False, output_dims = 2):

  
    """
    Trains the models depending on the number of random seeds
    a user supplied, saves the best performing models depending
    on dev loss and returns also stats
    """

    run_train = 0

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model_run_seed = args["seed"]
    ## sal_scorer
    
    if args["saliency_scorer"] is None: sal_scorer = ""
    else: sal_scorer = args["saliency_scorer"] + "_"

    torch.manual_seed(model_run_seed)
    torch.cuda.manual_seed(model_run_seed)
    np.random.seed(model_run_seed)

    model = bert(masked_list=[0,101,102], output_dim = output_dims)
        
    model.to(device)
    
    loss_function = nn.CrossEntropyLoss() 

    optimiser = AdamW([
        {'params': model.bert_model.parameters(), 'lr': args.lr_bert},
        {'params': model.output_layer.parameters(), 'lr': args.lr_classifier}], 
        correct_bias = False
        )

    if args.train_on_rat:
        
        saving_model = args["save_path"] + sal_scorer + args["importance_metric"] + "-" + args["model_abbreviation"] + str(model_run_seed) + ".pt"

    else:

        saving_model = args["save_path"] + sal_scorer + args["model_abbreviation"] + str(model_run_seed) + ".pt"

    dev_results, results_to_save = train_model(
        model,  
        train_data_loader, 
        dev_data_loader, 
        loss_function,
        optimiser,
        epochs = args["epochs"],
        cutoff = False, 
        save_folder = saving_model,
        run = run_train,
        seed = model_run_seed
    )

    if args.train_on_rat:
        
        text_file = open(args["save_path"] + "model_run_stats/" + sal_scorer + args["importance_metric"] + "-" + args["model_abbreviation"] + "_seed_" + str(model_run_seed) + ".txt", "w")

    else:
    
        text_file = open(args["save_path"] + "model_run_stats/" + sal_scorer + args["model_abbreviation"] + "_seed_" + str(model_run_seed) + ".txt", "w")

    text_file.write(results_to_save)
    text_file.close()

    df = pd.DataFrame.from_dict(dev_results)

    if args.train_on_rat:

        df.to_csv(args["save_path"] + "model_run_stats/" + sal_scorer + args["importance_metric"] + "-" + args["model_abbreviation"] + "_best_model_dev_seed_" + str(model_run_seed) + ".csv")

    else:

        df.to_csv(args["save_path"] + "model_run_stats/" + sal_scorer + args["model_abbreviation"] + "_best_model_dev_seed_" + str(model_run_seed) + ".csv")

    ## free up space 
    del model
    del optimiser
    gc.collect()
    torch.cuda.empty_cache()

import glob
import os 
import re

def test_predictive_performance(test_data_loader, for_rationale = False, output_dims = 2):    

    """
    Runs trained models on test set
    Also keeps the best model for experimentation
    and produces statistics    
    """
    
    if args["saliency_scorer"] is None: sal_scorer = ""
    else: sal_scorer = args["saliency_scorer"] + "_"

    if args.train_on_rat:

        saved_models = glob.glob(args["save_path"] + sal_scorer + args["importance_metric"] + "*.pt")
    
    else:

        saved_models = glob.glob(args["save_path"] + sal_scorer + "*.pt")
    
    stats_report = {}

    logging.info("-------------------------------------")
    logging.info("selecting best model")
    
    for current_model in saved_models:
        
        if args.train_on_rat:

            seed = re.sub(sal_scorer + args["importance_metric"] + "-bert", "", current_model.split(".pt")[0].split("/")[-1])

        else:

            seed = re.sub(sal_scorer + "bert", "", current_model.split(".pt")[0].split("/")[-1])

        model = bert(masked_list=[0,101,102], output_dim = output_dims)
     
        model.to(device)
        
        # loading the trained model
    
        model.load_state_dict(torch.load(current_model, map_location=device))
        
        model.to(device)
        
        loss_function = nn.CrossEntropyLoss()

        test_results,test_loss = test_model(model, loss_function, test_data_loader)
        
        df = pd.DataFrame.from_dict(test_results)
        
        df.to_csv(args["save_path"]  +"/model_run_stats/" + sal_scorer +  args["model_abbreviation"] + "_best_model_dev_seed_" + str(seed) + ".csv")

        stats_report["Macro F1 - avg:seed:" +str(seed)] = test_results["macro avg"]["f1-score"]
      
        logging.info(
            "seed: '{0}' -- Test loss: '{1}' -- Test accuracy: '{2}'".format(seed, round(test_loss, 3),round(test_results["macro avg"]["f1-score"], 3))
             )

        del model
        torch.cuda.empty_cache()
    
    
    """
    now to keep only the best model
    
    """
    
    performance_list = tuple(stats_report.items()) ## keeping the runs and acuracies

    performance_list = [(x.split(":")[-1], y) for (x,y) in performance_list]
    
    sorted_list = sorted(performance_list, key = lambda x: x[1])

    if args.train_on_rat:

        models_to_get_ridoff, _ = zip(*sorted_list) ### remove all models no need to store them

    else:

        models_to_get_ridoff, _ = zip(*sorted_list[:len(saved_models) - 1])

    for item in models_to_get_ridoff:

        if args.train_on_rat:
        
            os.remove(args["save_path"] +  sal_scorer + args["importance_metric"] + "-" + args["model_abbreviation"] + str(item) + ".pt")

        else:

            os.remove(args["save_path"] +  sal_scorer  + args["model_abbreviation"] + str(item) + ".pt")

    """
    saving the stats
    """
    
    stats_report["mean"] = np.asarray(list(stats_report.values())).mean()
    stats_report["std"] = np.asarray(list(stats_report.values())).std()
    stats_report = {k:[v] for k,v in stats_report.items()}
    
    df = pd.DataFrame(stats_report).T

    if args.train_on_rat:

        if args["saliency_scorer"] is None: sal_scorer = ""
        else: sal_scorer = args["saliency_scorer"]


        df.to_csv(args["save_path"] + args["importance_metric"] + "_" + sal_scorer+ "_" + args["model_abbreviation"] + "_predictive_performances.csv")

    else:

        df.to_csv(args["save_path"] + sal_scorer + args["model_abbreviation"] + "_predictive_performances.csv")
           

    torch.cuda.empty_cache()