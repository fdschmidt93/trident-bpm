#
# for LANG in "sw" "ur"
# do
#     for PARAM_STR in "null" "10M" "20M" "30M"
#     do
#         ./slurm_submit.py --time="8:00:00" --partition="single" --gres="gpu:1" --mem=32GB run_exp.py --embeds=wechsel --lang=$LANG --param_str=$PARAM_STR --steps=25000
#         ./slurm_submit.py --time="16:00:00" --partition="single" --gres="gpu:1" --mem=32GB run_exp.py --embeds=wechsel --lang=$LANG --param_str=$PARAM_STR --steps=50000
#         ./slurm_submit.py --time="24:00:00" --partition="single" --gres="gpu:1" --mem=32GB run_exp.py --embeds=wechsel --lang=$LANG --param_str=$PARAM_STR --steps=75000
#     done
# done

for LANG in "sw" # "ur"
do
    for PARAM_STR in "10M" "20M" # "30M"
    do
        ./slurm_submit.py --time="6:00:00" --partition="single" --gres="gpu:1" --mem=32GB run_exp.py --embeds=ident --lang=$LANG --param_str=$PARAM_STR --steps=25000
        ./slurm_submit.py --time="12:00:00" --partition="single" --gres="gpu:1" --mem=32GB run_exp.py --embeds=ident --lang=$LANG --param_str=$PARAM_STR --steps=50000
    done
done

# for LANG in "sw" "ur"
# do
#     for PARAM_STR in "10M" "20M" "30M"
#     do
#         ./slurm_submit.py --time="6:00:00" --partition="single" --gres="gpu:1" --mem=32GB run_exp.py --embeds=unsupervised --lang=$LANG --param_str=$PARAM_STR --steps=25000
#         ./slurm_submit.py --time="12:00:00" --partition="single" --gres="gpu:1" --mem=32GB run_exp.py --embeds=unsupervised --lang=$LANG --param_str=$PARAM_STR --steps=50000
#     done
# done
#
