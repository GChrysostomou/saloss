#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import argparse
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import datetime
import sys


date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "-dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "sst", 
    choices = ["sst", "agnews", "evinf", "adr", "multirc", "subj", "semeval"]
)

parser.add_argument(
    "-data_dir", 
    type = str, 
    help = "directory of saved processed data", 
    default = "datasets/"
)

parser.add_argument(
    "-model_dir",   
    type = str, 
    help = "directory to save models", 
    default = "full_text_models/"
)

parser.add_argument(
    "--saliency_scorer",
    type = str, 
    help = "saliency_scorer for loss", 
    default = None, 
    choices = [
        "textrank", "tfidf","chisquared", None,
        "textgraph", "random_alloc", "uniform_alloc"
    ]
)

### REST OF ARGS = to use for training rationales

parser.add_argument(
    "--thresholder", 
    type = str, 
    help = "thresholder for extracting rationales", 
    default = "topk",
    choices = ["contigious", "topk"]
)

parser.add_argument(
    "--train_on_rat",
    help = "train_on_rationales",
    action = "store_true"
)

parser.add_argument(
    "--importance_metric", 
    type = str, 
    help = "importance metric for ra.ext.", 
    default = "attention", 
    choices = ["attention", "scaled_attention", "gradients", "ig"]
)

parser.add_argument(
    "--seed", 
    type = int, 
    help = "random seed", 
    default = 100,
)

parser.add_argument(
    '--evaluate_models', 
    help='test predictive performance in and out of domain', 
    action='store_true'
)


user_args = vars(parser.parse_args())

if user_args["train_on_rat"]:

    log_dir = "experiment_logs/train_on_rat_" + user_args["dataset"] + "_" + date_time + "/"
    config_dir = "experiment_config/train_on_rat_" + user_args["dataset"] + "_" + date_time + "/"


else:
    
    log_dir = "experiment_logs/train_" + user_args["dataset"] + "_" + date_time + "/"
    config_dir = "experiment_config/train_" + user_args["dataset"] + "_" + date_time + "/"


os.makedirs(log_dir, exist_ok = True)
os.makedirs(config_dir, exist_ok = True)

import config.cfg

config.cfg.config_directory = config_dir



logging.basicConfig(
                    filename= log_dir + "/out.log", 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S'
                  )


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logging.info("Running on cuda : {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from src.utils.prep import initial_preparations, checks_on_local_args
import datetime
import sys

### training on rationales
if user_args["train_on_rat"]:

    bool_for_rationale = True

    # creating unique config from stage_config.json file and model_config.json file
    args = initial_preparations(user_args, stage = "retrain")

    args = checks_on_local_args("retrain", args)

else:

    bool_for_rationale = False

    # creating unique config from stage_config.json file and model_config.json file
    args = initial_preparations(user_args, stage = "train")

    args = checks_on_local_args("train", args)


logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")


from src.utils.dataholder import classification_dataholder as dataholder
from src.train_models import train_and_save, test_predictive_performance

from src.utils.assistant import describe_data_stats

# training the models and evaluating their predictive performance
# on the full text length

data = dataholder(
    args["data_dir"], 
    b_size = args["batch_size"],
    for_rationale = bool_for_rationale)


if args["evaluate_models"]:

    test_predictive_performance(
        test_data_loader = data.test_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels
    )

else:

    train_and_save(
        train_data_loader = data.train_loader, 
        dev_data_loader = data.dev_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels
    )

torch.cuda.empty_cache()