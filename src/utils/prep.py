import json
import glob
import config
   
def prepare_config(user_args):

  # ensure that model directory ends with "/"
  if user_args["model_dir"][-1] == "/":pass 
  else: user_args["model_dir"] = user_args["model_dir"] + "/"
  # ensure that data directory ends with "/"
  if user_args["data_dir"][-1] == "/":pass 
  else: user_args["data_dir"] = user_args["data_dir"] + "/"
  
  # passed arguments
  dataset = user_args["dataset"]


  with open('config/model_config.json', 'r') as f:
      model_args = json.load(f)

  args = model_args[dataset]

  model_dir = user_args["model_dir"] 
  user_args["data_dir"] = user_args["data_dir"] + dataset + "/data/"
  data_dir = user_args["data_dir"]

  save_path = model_dir + dataset + "/"

  ## preparing an argument to help for data processing for different tasks
  if dataset in ["evinf", "multirc"]: user_args["query"] = True
  else: user_args["query"] = False

  # save rationales and rationale models in :  
  # save_dir 

  cwd = os.getcwd() + "/"


  if "evaluation_dir" in user_args:

    if user_args["evaluation_dir"][-1] == "/":pass 
    else: user_args["evaluation_dir"] = user_args["evaluation_dir"] + "/"
   
    evaluation_dir = cwd + user_args["evaluation_dir"] + user_args["dataset"] + "/"

  else: evaluation_dir = None

  if "extracted_rationale_dir" in user_args:

    if user_args["extracted_rationale_dir"][-1] == "/":pass 
    else: user_args["extracted_rationale_dir"] = user_args["extracted_rationale_dir"] + "/"

    extracted_rationale_dir = cwd + user_args["extracted_rationale_dir"] + user_args["dataset"] + "/data/"

  else: extracted_rationale_dir = None

  if "rationale_model_dir" in user_args:

    rationale_model_dir = cwd + user_args["model_dir"] + user_args["dataset"] + "/"
  
  else: rationale_model_dir = None


  args = dict(user_args, **args, **{
            "epochs":model_args["epochs"], 
            "save_path":save_path, 
            "data_directory" : cwd + data_dir, 
            "extracted_rationale_dir" : extracted_rationale_dir,
            "rationale_model_dir": rationale_model_dir,
            "model_abbreviation": model_args["model_abbreviation"][args["model"]],
            "evaluation_dir": evaluation_dir
  })

  return save_path, args

import os

def checks_on_local_args(stage, local_args):

  """
  setting some default args that user should not change
  """
  
  new_args = local_args

  new_args["linguistic_feature"] = None

  if stage == "train":
    
    new_args["retrain"] = False
    new_args["train"] = True
    
    new_args["epochs"] =  10

  if stage == "retrain":

    new_args["epochs"] = 5

    new_args["train"] = False
    new_args["retrain"] = True

  if stage == "evaluate":

    new_args["evaluate"] = True
    new_args["train"] = False
    new_args["retrain"] = False

  if stage == "extract":

    new_args["evaluate"] = False
    new_args["train"] = False
    new_args["retrain"] = False

  
  #### saving config file for this run
  with open(config.cfg.config_directory + 'instance_config.json', 'w') as file:
      file.write(json.dumps(new_args,  indent=4, sort_keys=True))

  return new_args



def make_folders(save_path, args, stage):

  assert stage in ["train", "extract", "retrain", "evaluate"]

  if stage == "train":

    os.makedirs(save_path + "/model_run_stats/", exist_ok=True)
    print("\nFull text models saved in: {}\n".format(save_path))

  if stage == "extract":

    os.makedirs(args["extracted_rationale_dir"], exist_ok=True)
    print("\nExtracted rationales saved in: {}\n".format(args["extracted_rationale_dir"]))

  if stage == "retrain":

    print(args["model_dir"])
    os.makedirs(args["model_dir"] + args["dataset"]  + "/model_run_stats/", exist_ok=True)
    print("\nRationale models saved in: {}\n".format(args["model_dir"] + args["dataset"] ))

  if stage == "evaluate":

    os.makedirs(args["evaluation_dir"], exist_ok=True)
    print("\Decision flip results saved in: {}\n".format(args["evaluation_dir"]))

  

def initial_preparations(user_args, stage):

    save_path, args = prepare_config(user_args)

    make_folders(save_path, args, stage)

    return args
