## SaLoss

Repository for the paper *"Enjoy the Salience: Towards Better Transformer-based FaithfulExplanations with Word Salience", G.Chrysostomou and N.Aletras, to appear at EMNLP2021*. Pre-print available at this [link](https://arxiv.org/pdf/*)

## Prerequisites

Install necessary packages by using the files  [conda_reqs.txt](https://github.com/GChrysostomou/sal-loss_/blob/main/conda_reqs.txt) and  [pip_reqs.txt](https://github.com/GChrysostomou/sal-loss_/blob/main/pip_reqs.txt)  

```
conda create --name saloss --file  conda_reqs.txt
conda activate saloss
pip install -r pip_reqs.txt
python -m spacy download en
```

## Downloading Task Data
You can run the jupyter notebooks found under tasks/*task_name*/\*ipynb to generate a filtered, processed *csv* file used then by the dataloader to train the models.

## Training and Evaluating the models

You can train the models on full text using the following options: 

* dataset : *{"sst", "agnews", "evinf", "multirc", "semeval"}*
* data_dir : *directory where task data is* 
* model_dir : *directory for saved models*
* saliency_scorer : *{None, "textrank", "tfidf","chisquared"}*

and running the following script:

``` 
for seed in 100 200 300
do
python train_on_full.py -dataset $task_name -model_dir $model_dir --seed $seed 
done
python train_on_full.py -dataset $task_name -model_dir $model_dir --evaluate_models
```

simply add ```--saliency_scorer $sal_scorer``` if you want to add a particular saliency scorer

Example script (with Lin-TaSc):

```

python train_eval_bc.py -dataset sst 
			-encoder lstm 
			-mechanism dot 
			-data_dir data/ 
			-model_dir models/ 
			-lin
```

## Summarising results

Following the evaluation for multiple models / attention mechanisms , with and without TaSc, you can use [produce_reports.py](https://github.com/GChrysostomou/tasc/blob/master/produce_reports.py) to create tables in latex and as csv for the full stack of results (as a comparison), a table for comparing with other explanation techniques, results across attention mechanism, encoder and dataset. 

The script can be run with the following options:

* datasets: *list of datasets with permissible datasets listed above*
* encoders: *list of encoders with permissible encoders listed above*
* experiments_dir: *directory that contains saved results to use for summarising followed by /*
* mechanisms: *selection of attention mechanism with options {Tanh, Dot}*
* tasc_ver : *tasc version with options {lin, feat, conv}*

To generate radar plots you can run ```python produce_graphs.py```([produce_graphs.py](https://github.com/GChrysostomou/tasc/blob/master/produce_graphs.py)).

