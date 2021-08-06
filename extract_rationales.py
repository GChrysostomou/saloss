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
    "-extracted_rationale_dir",   
    type = str, 
    help = "directory to save extracted_rationales", 
    default = "extracted_rationales/"
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
    "--thresholder", 
    type = str, 
    help = "thresholder for extracting rationales", 
    default = "topk",
    choices = ["contigious", "topk"]
)



user_args = vars(parser.parse_args())

log_dir = "experiment_logs/extract_" + user_args["dataset"] + "_" + date_time + "/"
config_dir = "experiment_config/extract_" + user_args["dataset"] + "_" + date_time + "/"


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
args = initial_preparations(user_args, stage = "extract")

args = checks_on_local_args("extract", args)

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")

# re-importing module to reset args if needed
from src.utils import dataholder
from src.utils.dataholder import classification_dataholder


data = classification_dataholder(args["data_dir"], b_size = args["batch_size"])

from src.extractor.extract_rationales import extractor

extractor = extractor(data.nu_of_labels)

extractor._extract_rationales(data)

# extractor.extract_importance(data)

# delete full data not needed anymore
del data
torch.cuda.empty_cache()

