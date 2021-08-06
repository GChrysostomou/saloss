from src.importance_metrics.salient_scorers.tfidf import *
from src.importance_metrics.salient_scorers.text_rank import *
from src.importance_metrics.salient_scorers.chi_sq import *
from src.importance_metrics.salient_scorers.textrank_graph import text_grapher
from src.importance_metrics.salient_scorers.linguist import collective_tagger

import scipy.sparse as sparse


import numpy as np
np.random.seed(15)

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

class salient_scorer():

    def __init__(self, importance_metric):
        
        self.importance_metric = importance_metric

    def tfidf(self, data, **kwargs):
      
        if len(kwargs) == 1: tokenizer, train_vec = kwargs["tokenizer"], None
        else: tokenizer, train_vec = kwargs["tokenizer"], kwargs["train_vec"]

        if "extract_rationales" not in kwargs: extract_rationales = False
        else: extract_rationales = kwargs["extract_rationales"]

        return tfidf_scorer(data, tokenizer = tokenizer, train_vec = train_vec, extract_rationales = extract_rationales)

    def chisquared(self, data, **kwargs):
      
        if len(kwargs) == 1: tokenizer, train_vec = kwargs["tokenizer"], None
        else: tokenizer, train_vec = kwargs["tokenizer"], kwargs["train_vec"]

        if "extract_rationales" not in kwargs: extract_rationales = False
        else: extract_rationales = kwargs["extract_rationales"]

        return chi_scorer(data, tokenizer = tokenizer, train_vec = train_vec, extract_rationales = extract_rationales)

    def textrank(self,data, **kwargs):

        if "extract_rationales" not in kwargs: extract_rationales = False
        else: extract_rationales = kwargs["extract_rationales"]

        return text_ranker(data, tokenizer = kwargs["tokenizer"], extract_rationales = extract_rationales)

    def textgraph(self,data, **kwargs):

        if "extract_rationales" not in kwargs: extract_rationales = False
        else: extract_rationales = kwargs["extract_rationales"]

        return text_grapher(data, tokenizer = kwargs["tokenizer"], extract_rationales = extract_rationales)


    def random_alloc(self,data,**kwargs):

        if "extract_rationales" not in kwargs: extract_rationales = False
        else: extract_rationales = kwargs["extract_rationales"]

        if extract_rationales:
            
            data["text__"] = data.text
        
        else:
            
            data["text__"] = data.text.apply(lambda x : x["input_ids"])

        logging.info("calculating random scores")
        
        tqdm.pandas(desc = "calculating random scores", position = 0, leave = True)

        data["salient_scores"] = data.text__.progress_apply(
            lambda x : list(np.random.rand(len(x), 5).mean(1)),
        )

        logging.info("extracted random rationales")

        return {"scored_data":data.drop(columns = "text__")}

    def uniform_alloc(self,data,**kwargs):

        if "extract_rationales" not in kwargs: extract_rationales = False
        else: extract_rationales = kwargs["extract_rationales"]

        if extract_rationales:
            
            data["text__"] = data.text
        
        else:
            
            data["text__"] = data.text.apply(lambda x : x["input_ids"])

        logging.info("calculating random scores")
        
        tqdm.pandas(desc = "calculating uniform scores", position = 0, leave = True)

        data["salient_scores"] = data.text__.progress_apply(
            lambda x : list(np.ones(len(x))),
        )

        logging.info("extracted uniform rationales")

        return {"scored_data":data.drop(columns = "text__")}

    

    def return_all(self, data, **kwargs):

        data = self.textrank(data, **kwargs)
        data = data["scored_data"].rename(columns = {"salient_scores": "textrank"})
        data = self.tfidf(data, **kwargs)
        data["scored_data"] = data["scored_data"].rename(columns = {"salient_scores": "tfidf"})

        other_tagger = collective_tagger(kwargs["tokenizer"])

        data["scored_data"] = other_tagger.return_all(data["scored_data"])

        return data