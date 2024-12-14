


# Fine-tuning LMs on semantics-aware process mining tasks
- The sub-folder 'bpm' contains the necessary preprocessing and evaluation code.
- The individual tasks can be run using bash scripts, which submit a slurm job based on the parameters defined in configs/experiments:
    - 'pair.sh' for A-SAD using LLMs
    - 'trace.sh' for T-SAD using LLMs
    - 'activity.sh' for S-NAP using LLMs
    - 'next_activity.sh' for S-DFD using LLMs
    - 'pt.sh' for S-PTD using LLMs
    - 'trace_activity.sh' for multi-task T-SAD and S-NAP using LLMs
    - 'pair_roberta.sh' for A-SAD using RoBERTa
    - 'trace_roberta.sh' for T-SAD using RoBERTa
    - 'activity_roberta.sh' for S-NAP using RoBERTa
    - 'trace_activity_roberta.sh' for multi-task T-SAD and S-NAP using RoBERTa

- You need to adapt the parameters for each of the tasks, for the generation tasks, (currently) also the correct tokenizer must be imported in the src/bpm/processing.py file (line 375) 
 
 # trident-xtreme

`trident-xtreme` is a framework to declaratively run experiments for various natural language understanding tasks. The framework for now is geared towards evaluating cross-lingual transfer, but in principle seamlessly supports both pre-training and fine-tuning models. It uses [trident](https://fdschmidt93.github.io/trident/docs/readme.html) as its back-end, which end-to-end integrates [hydra](https://hydra.cc/) with [lightning](https://github.com/Lightning-AI/lightning).

`trident-xtreme` carves out the corresponding examples from the Huggingface `transformer` repositories into pipelines based modular components (trainer, modules, and datamodules).


## How To Best Use This Repository For Your Own Projects

1. Generate a new bare repository on Github or your service of choice
2. `git clone --mirror git@github.com:fdschmidt93/trident-xtreme.git temp`
3. `cd temp`
4. `git push --mirror git@github.com:$YOUR_USER/$YOUR_REPOSITORY.git`
5. `cd .. && rm -rf temp`
6. `git clone git@github.com:$YOUR_USER/$YOUR_REPOSITORY.git`
7. `git remote add origin git@github.com:$YOUR_USER/$YOUR_REPOSITORY.git`
8. `git remote add upstream git@github.com:fdschmidt93/trident-xtreme.git`

Any new features and updates go to branches on `origin`. Whenever you want to merge in updates from `trident-xtreme`, you can do `git fetch upstream` and then `git merge upstream/main`
