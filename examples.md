experiment is one of

bpm_next_activity_pt_pred
bpm_next_activity_pred
(ignore names)

means logging on every step
+trainer.log_every_n_steps=1 
means checking validation on every 10 steps, by default, every 33.3% of an epoch
trainer.val_check_interval=10 

env HYDRA_FULL_ERROR=1 python -m trident.run  experiment=bpm_next_activity_pt_pred +trainer.log_every_n_steps=1 trainer.val_check_interval=10 


how to change model: use run.pretrained_model_name_or_path -- it should flow 1:1
env HYDRA_FULL_ERROR=1 python -m trident.run  experiment=bpm_next_activity_pt_pred run.pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2"

how to store run outputs nicely, as an example (you can set other variables from the config)
env HYDRA_FULL_ERROR=1 python -m trident.run  experiment=bpm_next_activity_pt_pred run.pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2" 'hydra.run.dir="./logs/${run.task}/${run.pretrained_model_name_or_path}"'

env HYDRA_FULL_ERROR=1 python -m trident.run  experiment=bpm_next_activity_pt_pred 'hydra.run.dir="./logs/${run.task}/${run.pretrained_model_name_or_path}"'

