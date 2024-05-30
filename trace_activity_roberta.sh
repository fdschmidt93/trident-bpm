sbatch --partition=single --gres="gpu:1"  --time=48:00:00 --mem=64gb --wrap "source $HOME/.bashrc && conda activate copa && env HYDRA_FULL_ERROR=1 python -m trident.run experiment=bpm_trace_activity_seq_clf +trainer.log_every_n_steps=10 run.seed=${1}"

