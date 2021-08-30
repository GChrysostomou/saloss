#!/bin/bash

task_name="sst"
model_dir="models/" 
rat_data_dir="extracted_rationales/"
rat_model_dir="rationale_models/"
evaluation_dir="evaluation_results"
sal_scorer="textrank"


## train models on full text
for seed in 100 200 300
do
python train_on_full.py -dataset $task_name -model_dir $model_dir --seed $seed 
python train_on_full.py -dataset $task_name  -model_dir $model_dir --saliency_scorer $sal_scorer --seed $seed 
done
python train_on_full.py -dataset $task_name -model_dir $model_dir --evaluate_models
python train_on_full.py -dataset $task_name  -model_dir $model_dir --saliency_scorer $sal_scorer --evaluate_models

# evaluate on fraction of tokens
python evaluate_on_flips.py -dataset $task_name -model_dir $model_dir -evaluation_dir $evaluation_dir 
python evaluate_on_flips.py -dataset $task_name -model_dir $model_dir -evaluation_dir $evaluation_dir --saliency_scorer $sal_scorer

# extract rationales for all importance metrics
python extract_rationales.py -dataset $task_name -model_dir $model_dir --saliency_scorer $sal_scorer
python extract_rationales.py -dataset $task_name -model_dir $model_dir 

### retrain models on rationales for a particular feature attribution
### in this example attention
for seed in 100 200 300
do
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --seed $seed --train_on_rat  --importance_metric "attention"
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --saliency_scorer $sal_scorer --seed $seed --train_on_rat --importance_metric "attention"
done
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --evaluate_models --train_on_rat  --importance_metric "attention"
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --evaluate_models --saliency_scorer $sal_scorer --train_on_rat --importance_metric "attention"

