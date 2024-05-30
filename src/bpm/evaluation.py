import torch
from trident.core.module import TridentModule
import pandas as pd

from pathlib import Path


def store_predictions(
    trident_module: TridentModule,
    preds: torch.Tensor,
    labels: torch.Tensor,
    dataset_name: str,
    dir_: str,
    *args,
    **kwargs,
):
    trainer = trident_module.trainer
    epoch = trainer.current_epoch
    preds_ = pd.DataFrame(preds.cpu().numpy())
    preds_.columns = ["prediction"]
    labels_ = pd.DataFrame(labels.cpu().numpy())
    labels_.columns = ["labels"]
    df = pd.concat([preds_, labels_], axis=1)
    path = Path(dir_).joinpath(f"dataset={dataset_name}_epoch={epoch}.csv")
    df.to_csv(str(path), index_label="ids")
