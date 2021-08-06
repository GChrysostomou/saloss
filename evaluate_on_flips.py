#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
import numpy as np
import pandas as pd
import argparse
import json
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
    "-evaluation_dir",   
    type = str, 
    help = "directory to save decision flips", 
    default = "decision_flip/"
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

parser.add_argument(
    "--devel_stage",
    help = "run evaluation on devel set",
    action = "store_true"
)


parser.add_argument(
    "--extract_importance_scores",
    help = "run evaluation on devel set",
    action = "store_true"
)




user_args = vars(parser.parse_args())

log_dir = "experiment_logs/evaluate_" + user_args["dataset"] + "_" + date_time + "/"
config_dir = "experiment_config/evaluate_" + user_args["dataset"] + "_" + date_time + "/"


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

logging.info("Running on cuda ? {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from src.utils.prep import initial_preparations, checks_on_local_args
import datetime
import sys

# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "evaluate")

args = checks_on_local_args("evaluate", args)

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")

# re-importing module to reset args if needed
from src.utils import dataholder
from src.utils.dataholder import classification_dataholder


data = classification_dataholder(
    args["data_dir"],
    b_size = args["batch_size"]
)

from src.evaluation import fraction_of_tokens

evaluator = fraction_of_tokens.evaluate(data.nu_of_labels)


logging.info("-- conducting flip experiments")

evaluator.fraction_of_experiments_(data)

logging.info("-- finished flip experiments")

# delete full data not needed anymore
del data
torch.cuda.empty_cache()

  