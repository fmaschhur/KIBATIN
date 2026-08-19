"""Define the serialized multitask model and Hugging Face training hooks."""

import torch
import numpy as np
from torch.optim import AdamW
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    TrainerCallback,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
)
from transformers.tokenization_utils import BatchEncoding
from typing import Union


class CustomModelConfig(PretrainedConfig):
    """Serializable configuration for the three-head KIBATIN architecture.

    The private label fields represent independent diagnosis, indication, and
    triage heads; inherited single-task label metadata is not their contract.
    """

    model_type = "kibatin-multitask"

    def __init__(
        self,
        type: str = "unknown",
        pretrained_model_name: str = "GerMedBERT/medbert-512",
        num_extra_features: int = 0,
        extra_features_scale: int = 0,
        _num_labels: list[int] = None,
        _label2id: list[dict] = None,
        _id2label: list[dict] = None,
        backbone_config: dict | None = None,
        target_head: str = "tri",
        **kwargs,
    ):
        """Initialize architecture, label, backbone, and target metadata.

        ``target_head`` controls loss scheduling and validation selection, not
        which heads are present: every configured model contains all three.

        Raises:
            ValueError: If ``target_head`` is not ``dia``, ``ind``, or ``tri``.
        """
        if target_head not in {"dia", "ind", "tri"}:
            raise ValueError("target_head must be one of: dia, ind, tri.")
        if _num_labels is not None:
            self._num_labels = _num_labels
        if _id2label is not None:
            self._id2label = _id2label
        if _label2id is not None:
            self._label2id = _label2id
        self.pretrained_model_name = pretrained_model_name
        self.backbone_config = backbone_config
        self.target_head = target_head
        self.name_or_path = type + "/" + pretrained_model_name
        self.num_extra_features = num_extra_features
        self.extra_features_scale = extra_features_scale
        super().__init__(**kwargs)


# multi-task BERT model (one head each for diagnose, indication, triage)
# CAUTION: the inherited config.num_labels/id2label/label2id fields do not
# represent the three independent task heads.
class MultiTaskModel(PreTrainedModel):
    """BERT and numeric-feature model with three classification heads.

    The architecture transforms the BERT CLS representation and numeric
    features separately, joins them through a shared projection, and emits
    diagnosis, indication, and triage logits.
    """

    config_class = CustomModelConfig

    def __init__(
        self,
        config: config_class,
        use_auth_token: str | bool = False,
        initialize_backbone_from_pretrained: bool = False,
    ):
        """Construct the shared backbone, numeric MLP, and three heads.

        The backbone is reconstructed from serialized configuration and is
        initialized either from local layer construction or pretrained model
        weights. Authentication is forwarded only for pretrained loading.

        Raises:
            ValueError: If backbone metadata or numeric dimensions are invalid.
        """
        # config initialization
        super().__init__(config)
        if not isinstance(config.backbone_config, dict):
            raise ValueError(
                "MultiTaskModel requires a serialized backbone_config."
            )
        backbone_config_values = dict(config.backbone_config)
        model_type = backbone_config_values.pop("model_type", None)
        if not model_type:
            raise ValueError("backbone_config requires model_type.")
        backbone_config = AutoConfig.for_model(
            model_type, **backbone_config_values
        )
        auth_kwargs = {"use_auth_token": use_auth_token} if use_auth_token else {}
        if initialize_backbone_from_pretrained:
            self.bert = AutoModel.from_pretrained(
                config.pretrained_model_name,
                config=backbone_config,
                **auth_kwargs,
            )
        else:
            self.bert = AutoModel.from_config(backbone_config)
        bert_dim = self.bert.config.hidden_size
        self.base_model_prefix = "bert"

        nxf = config.num_extra_features
        xfs = config.extra_features_scale
        xfo = nxf * xfs
        nl = config._num_labels
        if nxf <= 0 or xfs <= 0:
            raise ValueError("MultiTaskModel requires extra features.")

        # BERT output normalization
        self.bert_norm = nn.Sequential(
            nn.LayerNorm(bert_dim),
            nn.Linear(bert_dim, bert_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # shared projection head
        self.projection_head_base = nn.Sequential(
            # nn.LayerNorm(bert_dim + xfo), - may give higher influence to extra features
            nn.Linear(bert_dim + xfo, bert_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        classification_head_in = bert_dim
        self.diagnoses_head = nn.Sequential(
            nn.LayerNorm(classification_head_in),
            nn.Linear(classification_head_in, nl[0]),
        )
        self.indications_head = nn.Sequential(
            nn.LayerNorm(classification_head_in),
            nn.Linear(classification_head_in, nl[1]),
        )
        self.triage_head = nn.Sequential(
            nn.LayerNorm(classification_head_in),
            nn.Linear(classification_head_in, nl[2]),
        )

        # extra feature MLP
        self.extra_feature_mlp = nn.Sequential(
            nn.BatchNorm1d(nxf),
            nn.Linear(nxf, int(xfo / 2), bias=False),
            nn.BatchNorm1d(int(xfo / 2)),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(int(xfo / 2), xfo, bias=True),
            nn.LayerNorm(xfo),
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        extra_features=None,
        labels=None,
    ):
        """Return diagnosis, indication, and triage logits for one batch.

        ``extra_features`` must match the configured numeric width. The returned
        mapping contains ``dia``, ``ind``, and ``tri`` tensors with shape
        ``(batch_size, task_label_count)``.

        Raises:
            ValueError: If the batch omits ``extra_features``.
        """
        # get BERT embeddings
        bert_complete_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        bert_output = bert_complete_output.last_hidden_state[:, 0, :]

        bert_output = self.bert_norm(bert_output)

        # scale extra features
        if extra_features is None:
            raise ValueError("MultiTaskModel requires extra_features in each batch.")
        extra_features_scaled = self.extra_feature_mlp(extra_features)

        # combine BERT and statistical feature representations
        head_input = torch.cat([bert_output, extra_features_scaled], dim=-1)

        # apply shared projection head
        head_input = self.projection_head_base(head_input)

        # return classification head outputs
        return {
            "dia": self.diagnoses_head(head_input),
            "ind": self.indications_head(head_input),
            "tri": self.triage_head(head_input),
        }


# dataset class needed for training (multiple heads)
class MultiTaskReportDataset(torch.utils.data.Dataset):
    """Dataset joining token encodings, three task labels, and numeric inputs."""

    def __init__(
        self,
        encodings: BatchEncoding,
        labels: list[list[str]],
        label2id: list[dict],
        extra_features: list[list[float]] = None,
    ):
        """Store aligned encodings, per-head labels, maps, and numeric rows."""
        self.encodings = encodings
        self.labels = labels
        self.label2id = label2id
        self.extra_features = extra_features

    def __getitem__(self, idx: int) -> dict:
        """Return one tensor-ready item with labels ordered dia, ind, then tri."""
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(
                [
                    self.label2id[0][self.labels[0][idx]],
                    self.label2id[1][self.labels[1][idx]],
                    self.label2id[2][self.labels[2][idx]],
                ],
                dtype=torch.long,
            )
        if self.extra_features is not None:
            item["extra_features"] = torch.tensor(
                self.extra_features[idx], dtype=torch.float32
            )
        return item

    def __len__(self):
        """Return the number of rows in the first tokenizer encoding field."""
        return len(list(self.encodings.values())[0])


# Trainer subclass with support for multiple heads and scheduled custom weighted XEnt loss
class WeightedMultiTaskTrainer(Trainer):
    """Trainer with weighted three-head loss and parameter-specific optimization."""

    def __init__(
        self,
        class_weights: list[float],
        loss_weights: list[float] = None,
        loss_schedule_fn: callable = lambda x: x,
        lr_bert: float | None = None,
        lr_pooler: float | None = None,
        lr_other: float | None = None,
        loss_weight_min_max: tuple[float, float] = (0.0, 1.0),
        *args,
        **kwargs,
    ):
        """Initialize class weights, scheduled head weights, and learning rates.

        When no component-specific rates are supplied, only layer-normalization
        parameters are excluded from weight decay; custom rates also exclude
        batch-normalization parameters.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_weights = (
            np.array(loss_weights) if loss_weights else np.array([1 / 3, 1 / 3, 1 / 3])
        )  # loss weights start at 1/3 for all heads
        self.loss_schedule_fn = loss_schedule_fn
        self.loss_weight_min_max = loss_weight_min_max
        print("_" * 20)
        print(f"model initialized with loss_weights: {self.loss_weights}")
        print("_" * 20)

        self.lr_bert = lr_bert if lr_bert is not None else self.args.learning_rate
        self.lr_pooler = lr_pooler if lr_pooler is not None else self.lr_bert
        self.lr_other = lr_other if lr_other is not None else self.args.learning_rate

        if lr_bert is None and lr_other is None and lr_pooler is None:
            self.exclude_decay = (nn.LayerNorm,)
        else:
            self.exclude_decay = (nn.LayerNorm, nn.BatchNorm1d)

    def compute_loss(
        self, model: PreTrainedModel, inputs: dict, return_outputs=False, **kwargs
    ) -> Union[tuple[list, list], list]:
        """Compute the scheduled sum of three class-weighted cross-entropy losses.

        The method mutates ``inputs`` by removing ``labels`` before forwarding
        the remaining batch to the model. Each task uses its own class weights
        and label-smoothing loss.

        Returns:
            The combined loss, optionally paired with the three-head logits.
        """
        labels = inputs.pop("labels")
        # forward pass
        logits = model(**inputs)
        # compute custom loss (suppose one has 3 labels with different weights)
        dia_loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(
                self.class_weights[0],
                device=next(model.parameters()).device,
                dtype=torch.float,
            ),
            label_smoothing=0.001,
        )
        ind_loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(
                self.class_weights[1],
                device=next(model.parameters()).device,
                dtype=torch.float,
            ),
            label_smoothing=0.001,
        )
        tri_loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(
                self.class_weights[2],
                device=next(model.parameters()).device,
                dtype=torch.float,
            ),
            label_smoothing=0.001,
        )
        dia_loss = dia_loss_fct(
            logits["dia"].view(-1, self.model.config._num_labels[0]), labels.T[0]
        )
        ind_loss = ind_loss_fct(
            logits["ind"].view(-1, self.model.config._num_labels[1]), labels.T[1]
        )
        tri_loss = tri_loss_fct(
            logits["tri"].view(-1, self.model.config._num_labels[2]), labels.T[2]
        )
        loss = (
            dia_loss * self.loss_weights[0]
            + ind_loss * self.loss_weights[1]
            + tri_loss * self.loss_weights[2]
        )
        return (loss, logits) if return_outputs else loss

    def get_train_dataloader(self):
        """Build a training loader that drops the final incomplete batch.

        Dropping the remainder avoids unsafe singleton batches in the numeric
        BatchNorm layers while restoring the caller's TrainingArguments value
        after loader construction.
        """
        original_drop_last = self.args.dataloader_drop_last
        self.args.dataloader_drop_last = True
        try:
            return super().get_train_dataloader()
        finally:
            self.args.dataloader_drop_last = original_drop_last

    def create_optimizer(self):
        """Create or return AdamW with component-aware learning-rate groups.

        Backbone, pooler, and remaining parameters receive their configured
        rates. Biases and normalization-module parameters are grouped with zero
        decay, while all other trainable parameters use configured weight decay.
        """
        if self.optimizer is not None:
            return self.optimizer

        id_to_lr = {}
        decay = {}
        no_decay = {}
        seen = set()

        if hasattr(self.model, "bert") and self.model.bert is not None:
            for param in self.model.bert.parameters():
                id_to_lr[id(param)] = self.lr_bert
            if (
                hasattr(self.model.bert, "pooler")
                and self.model.bert.pooler is not None
            ):
                for param in self.model.bert.pooler.parameters():
                    id_to_lr[id(param)] = self.lr_pooler

        for module in self.model.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)

                lr = id_to_lr.get(pid, self.lr_other)

                if name == "bias" or isinstance(module, self.exclude_decay):
                    no_decay.setdefault(lr, []).append(param)
                else:
                    decay.setdefault(lr, []).append(param)

        for param in self.model.parameters():
            if param.requires_grad and id(param) not in seen:
                lr = id_to_lr.get(id(param), self.lr_other)
                decay.setdefault(lr, []).append(param)

        param_groups = []
        for lr, ps in decay.items():
            if ps:
                param_groups.append(
                    {"params": ps, "lr": lr, "weight_decay": self.args.weight_decay}
                )
        for lr, ps in no_decay.items():
            if ps:
                param_groups.append({"params": ps, "lr": lr, "weight_decay": 0.0})

        self.optimizer = AdamW(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def update_loss_weights(self):
        """Advance head weights through the schedule and clip each bound."""
        self.loss_weights = np.array(self.loss_schedule_fn(self.loss_weights)).clip(
            self.loss_weight_min_max[0], self.loss_weight_min_max[1]
        )


class WeightLossScheduleCallback(TrainerCallback):
    """Epoch callback that advances a trainer's head-loss weights."""

    def __init__(
        self,
        trainer: WeightedMultiTaskTrainer,
        holdout: int = 0,
    ):
        """Bind a trainer and number of initial epochs to leave unchanged."""
        self.trainer = trainer
        self.holdout = holdout
        self.epoch_counter = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update scheduled loss weights after the configured holdout epochs."""
        self.epoch_counter += 1
        if self.epoch_counter > self.holdout:
            self.trainer.update_loss_weights()
            print("_" * 20)
            print(
                f"Updated loss weights at epoch {self.epoch_counter}: {self.trainer.loss_weights}"
            )
            print("_" * 20)
        return control


def data_collator_factory(tokenizer):
    """Build a dynamic-padding collator for text, labels, and numeric features.

    The returned collator removes labels and numeric tensors before tokenizer
    padding, then restores their stacked batches afterward.
    """
    base_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    def multitask_collate(features):
        """Pad a batch after popping non-text values from feature dictionaries.

        The supplied dictionaries are mutated by removing ``labels`` and
        ``extra_features`` when present.
        """
        # pull out non-text fields first so tokenizer.pad doesn't touch them
        labels = (
            torch.stack([f.pop("labels") for f in features])
            if "labels" in features[0]
            else None
        )
        extra = (
            torch.stack([f.pop("extra_features") for f in features])
            if "extra_features" in features[0]
            else None
        )

        batch = base_collator(features)  # pads input_ids/attention_mask dynamically

        if labels is not None:
            batch["labels"] = labels
        if extra is not None:
            batch["extra_features"] = extra
        return batch

    return multitask_collate
