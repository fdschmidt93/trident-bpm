from typing import cast
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from trident.core.module import TridentModule
from transformers.models.mixtral import MixtralForCausalLM
from omegaconf import DictConfig, ListConfig
from hydra.utils import instantiate


def get_labelled_tokens(tokenizer, label_tokens: list[str]) -> list[int]:
    """Returns the indices of label tokens in tokenizer vocabulary."""
    out = []
    name_or_path = tokenizer.name_or_path
    for token in label_tokens:
        # mistral has an empty string token before 0
        if name_or_path.startswith("mistralai/Mistral-7B") and token == "0":
            out.append(28734)
        else:
            input_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            assert len(input_ids) == 1, f"{token} is of length {len(input_ids)}"
            out.append(input_ids[0])
    return out


class LLMForSequenceClassification(TridentModule):
    def __init__(
        self,
        tokenizer: DictConfig | PreTrainedTokenizerFast,
        label_tokens: list[str] | dict[str, list[str]] = ["False", "True"],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = cast(
            PreTrainedTokenizerFast,
            instantiate(tokenizer) if isinstance(tokenizer, DictConfig) else tokenizer,
        )
        self.padding_side: str = tokenizer.padding_side
        self.label_tokens = label_tokens

    def setup(self, stage: str):
        if not hasattr(self, "_clf_head"):
            self.model = cast(MixtralForCausalLM, self.model)
            self.model.lm_head = torch.nn.Identity()
            # nn.Embedding
            # weight attribute is torch.Tensor of V by D,
            # where V is num tokens and D is hidden dimensionality
            embeddings = self.model.get_output_embeddings()
            if isinstance(self.label_tokens, (list, ListConfig)):
                ids = get_labelled_tokens(self.tokenizer, self.label_tokens)
                # index token ids of output embedding matrix
                # to fetch token embeddings in order of labelled tokens
                # resulting self.clf_head: D by len(label_tokens) dimensional torch.Tensor
                self._clf_head = nn.Parameter(embeddings.weight[ids].clone().T)
                self._clf_head.requires_grad = False
                self.clf_heads = None
            else:
                modules = []
                for values in self.label_tokens.values():
                    label_ids = get_labelled_tokens(self.tokenizer, values)
                    clf_head = nn.Parameter(embeddings.weight[label_ids].clone().T)
                    clf_head.requires_grad = False
                    modules.append(clf_head)
                self.clf_heads = modules
                self.dataset2idx = {
                    k: i for i, k in enumerate(self.label_tokens.keys())
                }
                self._clf_head = self.clf_heads[0]

    def forward(self, batch: dict) -> dict[str, None | torch.Tensor]:
        self.model = cast(MixtralForCausalLM, self.model)
        hidden_states_NLD = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
        ).hidden_states[-1]
        # left padding
        if self.padding_side == "left":
            last_token_states_ND = hidden_states_NLD[:, -1, :]
        else:
            # TODO: make sure no end-of-sequence token is in there
            last_token_ids = (batch["attention_mask"].sum(1) - 1).long()
            last_token_states_ND = hidden_states_NLD[:, last_token_ids, :]
        logits_NC = last_token_states_ND @ self._clf_head
        if (labels := batch.get("labels")) is not None:
            loss = F.cross_entropy(logits_NC, labels)
        else:
            loss = None
        return {"logits": logits_NC, "loss": loss}

    def training_step(
        self, batch: dict, batch_idx: int
    ) -> dict[str, None | torch.Tensor]:
        if "input_ids" in batch:
            output = self(batch)
            self.log("train/loss", output["loss"])
            return output
        else:
            assert self.clf_heads is not None
            losses = []
            for k, v in batch.items():
                self._clf_head = self.clf_heads[self.dataset2idx[k]]
                outputs = self(v)
                loss = outputs["loss"]
                self.log(f"train/{k}_loss", loss)
                losses.append(loss)
            loss = torch.stack(losses).mean()
            self.log("train/loss", loss)
            return {"loss": loss}

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.clf_heads is not None:
            # 0 and 2 are trace dataloaders
            # 1 and 3 are activity dataloaders
            self._clf_head = self.clf_heads[dataloader_idx % 2]

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.clf_heads is not None:
            # 0 and 2 are trace dataloaders
            # 1 and 3 are activity dataloaders
            self._clf_head = self.clf_heads[dataloader_idx % 2]


class RobertaForMultipleSequenceClassification(TridentModule):
    def __init__(
        self,
        num_labels: dict[str, int],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_labels = num_labels

    def setup(self, stage: str):
        if not hasattr(self, "_clf_head"):
            clf_heads = []
            for num_labels in self.num_labels.values():
                cfg = deepcopy(cast(RobertaConfig, self.model.config))
                cfg.update({"num_labels": num_labels})
                clf_head = RobertaClassificationHead(cfg)
                clf_heads.append(clf_head)

            self.clf_heads = nn.ModuleList(clf_heads)
            self._clf_head = self.clf_heads[0]
            self.dataset2idx = {k: i for i, k in enumerate(self.num_labels.keys())}
        #     else:
        #         modules = []
        #         for values in self.label_tokens.values():
        #             label_ids = get_labelled_tokens(self.tokenizer, values)
        #             clf_head = nn.Parameter(embeddings.weight[label_ids].clone().T)
        #             clf_head.requires_grad = False
        #             modules.append(clf_head)
        #         self.clf_heads = modules
        #         self.dataset2idx = {
        #             k: i for i, k in enumerate(self.label_tokens.keys())
        #         }
        # self._clf_head = self.clf_heads[0]

    def forward(self, batch: dict) -> dict[str, None | torch.Tensor]:
        outputs = self.model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        sequence_output = outputs[0]
        logits = self._clf_head(sequence_output)
        V = logits.shape[-1]
        loss = None
        if (labels := batch.get("labels", None)) is not None:
            loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
        }

    def training_step(
        self, batch: dict, batch_idx: int
    ) -> dict[str, None | torch.Tensor]:
        if "input_ids" in batch:
            return self(batch)
        else:
            losses = []
            for k, v in batch.items():
                self._clf_head = self.clf_heads[self.dataset2idx[k]]
                outputs = self(v)
                loss = outputs["loss"]
                self.log(f"train/{k}_loss", loss)
                losses.append(loss)
            loss = torch.stack(losses).mean()
            self.log("train/loss", loss)
            return {"loss": loss}

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # self._clf_head = self.clf_heads[dataloader_idx]
        self._clf_head = self.clf_heads[dataloader_idx % 2]

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # self._clf_head = self.clf_heads[dataloader_idx]
        self._clf_head = self.clf_heads[dataloader_idx % 2]


#
# class RobertaForMultipleChoice(TridentModule):
#     def __init__(
#         self,
#         tokenizer: DictConfig | PreTrainedTokenizerFast,
#         label_tokens=["False", "True"],
#         *args,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.tokenizer = cast(
#             PreTrainedTokenizerFast,
#             instantiate(tokenizer) if isinstance(tokenizer, DictConfig) else tokenizer,
#         )
#         self.padding_side: str = tokenizer.padding_side
#         self.label_tokens: list[str] = label_tokens
#
#     def setup(self, stage: str):
#         if not hasattr(self, "clf_head"):
#             self.model = cast(MixtralForCausalLM, self.model)
#             self.model.lm_head = torch.nn.Identity()
#             embeddings = self.model.get_input_embeddings()
#             ids = get_labelled_tokens(self.tokenizer, self.label_tokens)
#             # index token ids of output embedding matrix
#             # to fetch token embeddings in order of labelled tokens
#             # resulting self.clf_head: len(label tokens) by D dimensional torch.Tensor
#             self.clf_head = nn.Parameter(embeddings.weight[ids].clone().T)
#             self.clf_head.requires_grad = False
#
#     def forward(self, batch: dict) -> dict[str, None | torch.Tensor]:
#         self.model = cast(MixtralForCausalLM, self.model)
#         hidden_states_NLD = self.model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch.get("attention_mask"),
#             output_hidden_states=True,
#         ).hidden_states[-1]
#         # left padding
#         if self.padding_side == "left":
#             last_token_states_ND = hidden_states_NLD[:, -1, :]
#         else:
#             # TODO: make sure no end-of-sequence token is in there
#             last_token_ids = (batch["attention_mask"].sum(1) - 1).long()
#             last_token_states_ND = hidden_states_NLD[:, last_token_ids, :]
#         logits_NC = last_token_states_ND @ self.clf_head
#         if (labels := batch.get("labels")) is not None:
#             loss = F.cross_entropy(logits_NC, labels)
#         else:
#             loss = None
#         return {"logits": logits_NC, "loss": loss}
