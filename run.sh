#!/bin/bash

#SBATCH --job-name=matching
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=20G

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate matching

srun --exclusive --ntasks=1 python -W ignore::RuntimeWarning train_vae.py --dataset=BALLS --max_epochs=250 --batch_size=100
srun --exclusive --ntasks=1 python -W ignore::RuntimeWarning main.py --dataset=BALLS --max_epochs=250 --batch_size=100
srun --exclusive --ntasks=1 python -W ignore::RuntimeWarning main.py --dataset=GEXADT --max_epochs=250 --batch_size=256
srun --exclusive --ntasks=1 python -W ignore::RuntimeWarning train_vae.py --dataset=GEXADT --max_epochs=250 --batch_size=256

