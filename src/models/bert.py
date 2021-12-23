import torch
import torch.nn as nn
import math 
from transformers import AutoModel, AutoConfig,  AutoModelForQuestionAnswering
import json 
from src.models.bert_components import BertModelWrapper#,  BertModelWrapperQA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

class bert(nn.Module):
    def __init__(self, masked_list = [0,101,102], output_dim = 2, dropout=0.1):
        
        super(bert, self).__init__()

        """
        BERT FOR CLASSIFICATION
        Args:
            output_dim : number of labels to classify
            mask_list : a list of tokens that we want to pad out (e.g. SEP, CLS tokens)
                        when normalising the attentions. 
                        **** WARNING mask list is not used on the input ****
        Input:
            **inputs : dictionary with encode_plus outputs + salient scores if needed + retain_arg 
        Output:
            yhat : the predictions of the classifier
            attention weights : the attenion weights corresponding to the tokens from the last layer
        """

        with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
            args = AttrDict(json.load(f))

        if "tfidf_asvec" in args:
       
            if args.tfidf_asvec:

                self.bert_hidden_dim = 768 + args.tfidf_vecdim

        else:  self.bert_hidden_dim = 768

        self.output_dim = output_dim        

        self.masked_list = masked_list
        self.dropout = dropout

        self.bert_config = AutoConfig.from_pretrained(args["model"], output_attentions = True)   
        
        bert_model = AutoModel.from_pretrained(args["model"], config=self.bert_config)        

        self.bert_model = BertModelWrapper(bert_model)

        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(768, self.output_dim)#self.bert_hidden_dim, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, **inputs):

        if "ig" not in inputs: inputs["ig"] = int(1)

        self.output, pooled_output, attention_weights = self.bert_model(inputs["sentences"], 
                                                                    attention_mask = inputs["attention_mask"],
                                                                    token_type_ids = inputs["token_type_ids"],
                                                                    ig = inputs["ig"]
                                                                    )

        # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = attention_weights[-1][:, :, 0, :].mean(1)
        # to retain gradients
        self.weights_or = attention_weights[-1]

        if inputs["retain_gradient"]:
            
            self.weights_or.retain_grad()
            self.bert_model.model.embeddings.word_embeddings.weight.retain_grad()

        logits = self.output_layer(self.dropout((pooled_output)))

        self.probs = torch.softmax(logits, dim = -1)

        return self.probs.to(device), self.weights


    def normalise_scores(self, scores, sequence):
        
        """
        returns word-piece normalised scores
        receives as input the scores {attention-weights, gradients} from bert and the sequence
        the sequence is used to create a mask with default tokens masked (101,102,0)
        which correspond to (CLS, SEP, PAD)
        """

        # mask from mask_list used to remove SOS, EOS and PAD tokens
        self.normalised_mask = torch.zeros_like(scores).bool()
    
        for item in self.masked_list:
        
            self.normalised_mask += (sequence == item).bool()

        # mask unwanted tokens
        scores = torch.masked_fill(scores, self.normalised_mask.to(device), 0)
   
        # return normalised word-piece scores       
        return scores / scores.sum(-1, keepdim = True)

    
    def integrated_grads(self, original_grad, original_pred, steps = 10, **inputs):

        lengths = inputs["lengths"]
        grad_list = [original_grad]
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):

            inputs["ig"] = x
            
            pred, _ = self.forward(**inputs)

            if len(pred.shape) == 1:

                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:

                baseline = pred[rows, original_pred[1]]

            pred[rows, original_pred[1]].sum().backward()

            #embedding gradients
            embed_grad = self.bert_model.model.embeddings.word_embeddings.weight.grad
            g = embed_grad[inputs["sentences"].long()][:,:max(inputs["lengths"])]

            grad_list.append(g)

        attributions = torch.stack(grad_list).mean(0)

        em = self.bert_model.model.embeddings.word_embeddings.weight[inputs["sentences"].long()][:,:max(inputs["lengths"])]

        ig = (attributions* em).sum(-1)[:,:max(lengths)]
        
        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))

        return ig
