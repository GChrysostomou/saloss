#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=3:00:00

# set name of job
#SBATCH --job-name=sst

# request partition
#SBATCH --partition=devel

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL


module load cuda/10.1

module load torch/20Feb2018

module load python3/anaconda

source activate imp_rat

task_name="sst"
model_dir="models/" 
rat_data_dir="extracted_rationales/"
rat_model_dir="rationale_models/"
evaluation_dir="evaluation_results"
sal_scorer="tfidf"

for seed in 100 200 300
do
python train_on_full.py -dataset $task_name -model_dir $model_dir --seed $seed --saliency_scorer chisquared
python train_on_full.py -dataset $task_name  -model_dir $model_dir --saliency_scorer $sal_scorer --seed $seed 
done
python train_on_full.py -dataset $task_name -model_dir $model_dir  --saliency_scorer chisquared --evaluate_models
python train_on_full.py -dataset $task_name  -model_dir $model_dir --saliency_scorer $sal_scorer --evaluate_models

# eval on flips
python evaluate_on_flips.py -dataset $task_name -model_dir $model_dir -evaluation_dir $evaluation_dir --saliency_scorer chisquared
python evaluate_on_flips.py -dataset $task_name -model_dir $model_dir -evaluation_dir $evaluation_dir --saliency_scorer $sal_scorer

extract rationales
python extract_rationales.py -dataset $task_name -model_dir $model_dir --saliency_scorer $sal_scorer
python extract_rationales.py -dataset $task_name -model_dir $model_dir --saliency_scorer chisquared

for seed in 100 200 300
do
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --seed $seed --train_on_rat  --saliency_scorer chisquared
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --saliency_scorer $sal_scorer --seed $seed --train_on_rat
done
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --evaluate_models --train_on_rat  --saliency_scorer chisquared
python train_on_full.py -dataset $task_name -data_dir $rat_data_dir -model_dir $rat_model_dir --evaluate_models --saliency_scorer $sal_scorer --train_on_rat

