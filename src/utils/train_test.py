from tqdm import trange
import torch
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


import json 
import logging


class checkpoint_holder(object):

    """
    holds checkpoint information for the 
    training of models
    """

    def __init__(self, save_model_location):

        self.dev_loss = float("inf")
        self.save_model_location = save_model_location
        self.point = 0
        self.storer = {}

    def _store(self, model, point, epoch, dev_loss, dev_results):
        
        if self.dev_loss > dev_loss:
            
            self.dev_loss = dev_loss
            self.point = point
            self.storer = dev_results
            self.storer["epoch"] = epoch + 1
            self.storer["point"] = self.point
            self.storer["dev_loss"] = self.dev_loss

            torch.save(model.state_dict(), self.save_model_location)

        return self.storer
        
import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

def kl_div_loss(p, q) :

    log_p = torch.log(p + 1e-10)
    log_q = torch.log(q + 1e-10)
    kld = p * (log_p - log_q.float())

    return kld.sum()**2



def train_model(model, training, development, loss_function, optimiser, seed,
            run,epochs = 10, cutoff = True, save_folder  = None, 
            cutoff_len = 2):
    
    """ 
    Trains the model and saves it at required path
    Input: 
        "model" : initialised pytorch model
        "training" : training dataset
        "development" : development dataset
        "loss_function" : loss function to calculate loss at output
        "optimiser" : pytorch optimiser (Adam)
        "run" : which of the 5 training runs is this?
        "epochs" : number of epochs to train the model
        "cutoff" : early stopping (default False)
        "cutoff_len" : after how many increases in devel loss to cut training
        "save_folder" : folder to save checkpoints
    Output:
        "saved_model_results" : results for best checkpoint of this run
        "results_for_run" : analytic results for all epochs during this run
    """

    results = []
    
    results_for_run = ""
    
    pbar = trange(len(training) *epochs, desc='run ' + str(run+1), leave=True, 
    bar_format = "{l_bar}{bar}{elapsed}<{remaining}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    checkpoint = checkpoint_holder(save_model_location = save_folder)

    total_steps = len(training) * args["epochs"]
    
    if args.train_on_rat:
        warmup =  int(0.1*len(training))
    else:
        warmup = 0
    
    scheduler = get_linear_schedule_with_warmup(
                                                optimiser,
                                                num_warmup_steps=warmup,
                                                num_training_steps=total_steps
                                                )
    every = round(len(training) / 5)

    logging.info("-------------------------------------")
    logging.info("training on run {}".format(str(run)))
    logging.info("-saving checkpoint every {} iterations".format(every))

    if every == 0: every = 1

    model.train()

    loss_monitor = False

    if loss_monitor:
        act_loss = []
        kl_loss = []
    
    for epoch in range(epochs):
        
        total_loss = 0

        checks = 0       
        
        for batch in training:
            
            model.zero_grad()

            batch = [torch.stack(t).transpose(0,1) if type(t) is list else t for t in batch]
            
            inputs = {
                "sentences" : batch[0].to(device),
                "lengths" : batch[1].to(device),
                "labels" : batch[2].to(device),
                "token_type_ids" : batch[5].to(device),
                "attention_mask" : batch[6].to(device),
                "retain_gradient" : False
            }

            if len(batch) < 8: inputs["salient_scores"] = None
            else: 
                
                if len(batch) == 10:

                    sals = [x.float() for x in batch[-3:]]

                    inputs["salient_scores"] = torch.stack(sals).transpose(0,1).transpose(1,-1).to(device)

                else:

                    inputs["salient_scores"] = batch[7].to(device)
                    inputs["salient_scores"] = torch.masked_fill(inputs["salient_scores"], ~inputs["attention_mask"].bool(), float("-inf"))
                    inputs["salient_scores"] = torch.softmax(inputs["salient_scores"], dim = -1)

                    saliency_scores = inputs["salient_scores"].to(device)

                    inputs["salient_scores"] = None

            

            yhat, weights =  model(**inputs)

            if len(yhat.shape) == 1:
                
                yhat = yhat.unsqueeze(0)

            loss = loss_function(yhat, inputs["labels"]) 
            
            if (args.train == True and args.saliency_scorer):
                
                lambd = args.lr_sali

                weights = torch.masked_fill(weights, ~inputs["attention_mask"].bool(), float("-inf"))
                weights = torch.softmax(weights, -1)

                loss += lambd*kl_div_loss(weights, saliency_scores)

                if loss_monitor: 

                    act_loss.append(loss_function(yhat, inputs["labels"]))

                    kla = kl_div_loss(weights, saliency_scores)
                    kl_loss.append(kla)

            
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.)
            
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            pbar.update(1)
            pbar.refresh()
                
            if checks % every == 0:
                dev_results, dev_loss = test_model(model, loss_function, development)

                checkpoint_results = checkpoint._store(
                    model = model, 
                    point = checks, 
                    epoch = epoch, 
                    dev_loss = dev_loss, 
                    dev_results = dev_results
                )


            checks += 1


        dev_results, dev_loss = test_model(model, loss_function, development)    

        results.append([epoch, dev_results["macro avg"]["f1-score"], dev_loss, dev_results])
        
        logging.info("---epoch - {} | train loss - {} | dev f1 - {} | dev loss - {}".format(epoch + 1,
                                    round(total_loss * training.batch_size / len(training),2),
                                    round(dev_results["macro avg"]["f1-score"], 3),
                                    round(dev_loss, 2)))

        
        results_for_run += "epoch - {} | train loss - {} | dev f1 - {} | dev loss - {} \n".format(epoch + 1,
                                    round(total_loss * training.batch_size / len(training),2),
                                    round(dev_results["macro avg"]["f1-score"], 3),
                                    round(dev_loss, 2))


    
    ### monitor KL and Cross Entropy
    if (args.train == True and args.saliency_scorer and loss_monitor == True):

        import os
        os.makedirs("loss_monitor/" + config.cfg.config_directory.split("/")[-2], exist_ok = True)
        
        plt.figure(figsize = (14,10))
        plt.plot(range(len(act_loss)), act_loss, label = "CrossEntropyLoss")
        plt.plot(range(len(kl_loss)), kl_loss, label = "KLDivergence")
        plt.legend()

        plt.savefig("loss_monitor/" + config.cfg.config_directory.split("/")[-2] + "/monitoring_losses" + str(run) +".png", dpi = 100)

        a_file = "loss_monitor/" + config.cfg.config_directory.split("/")[-2] + "/" + "CrossEntropyLoss" + str(run) +".txt"

        np.savetxt(a_file, act_loss)

        k_file = "loss_monitor/" + config.cfg.config_directory.split("/")[-2] + "/" + "KLDivergence" + str(run) +".txt"
        
        np.savetxt(k_file, kl_loss)
        


    return checkpoint_results, results_for_run

from sklearn.metrics import classification_report

def test_model(model, loss_function, data):
    
    """ 
    Model predictive performance on unseen data
    Input: 
        "model" : initialised pytorch model
        "loss_function" : loss function to calculate loss at output
        "data" : unseen data (test)
    Output:
        "results" : classification results
        "loss" : normalised loss on test data
    """

    predicted = [] 
    
    actual = []
    
    total_loss = 0
  
    with torch.no_grad():
    
        for batch in data:
            
            model.eval()

            batch = [torch.stack(t).transpose(0,1) if type(t) is list else t for t in batch]
            
            inputs = {
                "sentences" : batch[0].to(device),
                "lengths" : batch[1].to(device),
                "labels" : batch[2].to(device),
                "token_type_ids" : batch[5].to(device),
                "attention_mask" : batch[6].to(device),
                "retain_gradient" : False
            }

            

            if len(batch) < 8: inputs["salient_scores"] = None
            else: 
                
                if len(batch) == 10:

                    sals = [x.float() for x in batch[-3:]]

                    inputs["salient_scores"] = torch.stack(sals).transpose(0,1).transpose(1,-1).to(device)

                else:

                    inputs["salient_scores"] = batch[7].to(device)
                    inputs["salient_scores"] = torch.masked_fill(inputs["salient_scores"], ~inputs["attention_mask"].bool(), float("-inf"))
                    inputs["salient_scores"] = torch.softmax(inputs["salient_scores"], dim = -1)

                    saliency_scores = inputs["salient_scores"].to(device)

                    inputs["salient_scores"] = None
            
            yhat, weights =  model(**inputs)

            if len(yhat.shape) == 1:
                
                yhat = yhat.unsqueeze(0)

            loss = loss_function(yhat, inputs["labels"])

            if (args.train == True and args.saliency_scorer):

                weights = torch.masked_fill(weights, ~inputs["attention_mask"].bool(), float("-inf"))
                weights = torch.softmax(weights, -1)

                loss += args.lr_sali * kl_div_loss(weights, saliency_scores)
        
            total_loss += loss.item()
            
            _, ind = torch.max(yhat, dim = 1)
    
            predicted.extend(ind.cpu().numpy())
    
            actual.extend(inputs["labels"].cpu().numpy())
   
        results = classification_report(actual, predicted, output_dict = True)

    
    return results, (total_loss * data.batch_size / len(data)) 
