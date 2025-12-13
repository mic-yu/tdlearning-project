#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate roboClassEnv
export WANDB_API_KEY=
python run.py