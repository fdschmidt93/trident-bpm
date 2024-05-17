# sbatch --partition=single --gres="gpu:A100:1"  --time=14:00:00 --mem=64gb --wrap "source $HOME/.bashrc && conda activate copa && env HYDRA_FULL_ERROR=1 python -m trident.run experiment=bpm_trace_clf +trainer.log_every_n_steps=10"
# env HYDRA_FULL_ERROR=1 python -m trident.run experiment=bpm_trace_activity_clf +trainer.log_every_n_steps=10 +trainer.val_check_interval=10 +trainer.limit_val_batches=25
sbatch --partition=single --gres="gpu:A100:1"  --time=72:00:00 --mem=64gb --wrap "source $HOME/.bashrc && conda activate copa && env HYDRA_FULL_ERROR=1 python -m trident.run experiment=bpm_trace_activity_clf +trainer.log_every_n_steps=10 run.seed=${1}"
