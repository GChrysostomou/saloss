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

### Training on full text

You can train the models on full text using the following options: 

* dataset : *{"sst", "agnews", "evinf", "multirc", "semeval"}*
* data_dir : *directory where task data is* 
* model_dir : *directory for saved models*
* saliency_scorer : *{None, "textrank", "tfidf","chisquared", "uniform"}*

and running the following script:

``` 
for seed in 100 200 300
do
python train_on_full.py -dataset $task_name -model_dir $model_dir --seed $seed 
done
python train_on_full.py -dataset $task_name -model_dir $model_dir --evaluate_models
```

simply add ```--saliency_scorer $sal_scorer``` if you want to add a particular saliency scorer in the above and <b>ANY</b> of the following.

### Extracting rationales

* extracted_rationale_dir : *directory where to store extracted rationales and importance scores* 

```
python extract_rationales.py -dataset $task_name -model_dir $model_dir -extracted_rationale_dir $extract_rat_dir
```

### Evaluating on frac of tokens

* evaluation_dir : *directory where to save frac of results*

``` 
python evaluate_on_flips.py -dataset $task_name -model_dir $model_dir -evaluation_dir $evaluation_dir 
```

### Training on rationales 

It is important to train on rationales using the following argument ```--train_on_rat```

* importance_metric : *{"attention", "scaled_attention", "gradients", "ig"}*
* thresholder : *{topk, contigious}*

```
for seed in 100 200 300
do
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --seed $seed --train_on_rat  --importance_metric "attention" --thresholder $thresh
done
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --evaluate_models --train_on_rat  --importance_metric "attention" --thresholder $thresh
```


## Summarising results

To create the tables and figures seen in the paper you can run the [generate_results.py](https://github.com/GChrysostomou/sal-loss_/blob/main/src/utils/generate_results.py) script.
