#!/bin/bash

# sbatch --partition=single --gres="gpu:A40:1" --time=7:00:00 --mem=64gb --wrap "source $HOME/.bashrc && conda activate copa && python -m trident.run experiment=dialect_copa_clf"

for LANG in "en" "hr" "mk" "sl" "sr"
do
    sbatch --partition=single --gres="gpu:A40:1" --time=5:30:00 --mem=64gb --wrap "source $HOME/.bashrc && conda activate copa && python -m trident.run experiment=dialect_copa_clf run.lang=$LANG hydra.run.dir=logs/model-mixtral-instruct/lang-$LANG/shots-400/lr-1e5/"
    for SHOTS in 10 50 100
    do
        sbatch --partition=single --gres="gpu:A40:1" --time=5:30:00 --mem=64gb --wrap "source $HOME/.bashrc && conda activate copa && python -m trident.run experiment=dialect_copa_clf run.lang=$LANG run.shots=$SHOTS trainer.max_epochs=50 trainer.check_val_every_n_epoch=5 hydra.run.dir=logs/model-mixtral-instruct/lang-$LANG/shots-$SHOTS/lr-1e5/"
    done
done
