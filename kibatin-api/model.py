"""Reconstruct the serialized three-head architecture for API inference."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class MultiTaskModel(nn.Module):
    """Inference-only copy of the serialized KIBATIN training architecture.

    The model joins the BERT CLS representation with transformed numeric
    features and applies shared projection plus diagnosis, indication, and
    triage classification heads.
    """

    def __init__(self, manifest: dict[str, Any]):
        """Construct all layers from a validated artifact manifest.

        The caller supplies the serialized backbone configuration, numeric
        feature dimensions, and the three label counts in diagnosis,
        indication, and triage order.
        """
        super().__init__()
        backbone_values = dict(manifest["backbone_config"])
        model_type = backbone_values.pop("model_type")
        backbone_config = AutoConfig.for_model(model_type, **backbone_values)
        self.bert = AutoModel.from_config(backbone_config)
        bert_dim = self.bert.config.hidden_size
        self.base_model_prefix = "bert"

        num_extra_features = int(manifest["num_extra_features"])
        extra_features_scale = int(manifest["extra_features_scale"])
        extra_feature_output = num_extra_features * extra_features_scale
        num_labels = [int(value) for value in manifest["num_labels"]]

        self.bert_norm = nn.Sequential(
            nn.LayerNorm(bert_dim),
            nn.Linear(bert_dim, bert_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.projection_head_base = nn.Sequential(
            nn.Linear(bert_dim + extra_feature_output, bert_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.diagnoses_head = nn.Sequential(
            nn.LayerNorm(bert_dim), nn.Linear(bert_dim, num_labels[0])
        )
        self.indications_head = nn.Sequential(
            nn.LayerNorm(bert_dim), nn.Linear(bert_dim, num_labels[1])
        )
        self.triage_head = nn.Sequential(
            nn.LayerNorm(bert_dim), nn.Linear(bert_dim, num_labels[2])
        )
        self.extra_feature_mlp = nn.Sequential(
            nn.BatchNorm1d(num_extra_features),
            nn.Linear(num_extra_features, int(extra_feature_output / 2), bias=False),
            nn.BatchNorm1d(int(extra_feature_output / 2)),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(int(extra_feature_output / 2), extra_feature_output, bias=True),
            nn.LayerNorm(extra_feature_output),
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

        ``extra_features`` must contain the numeric feature vector expected by
        the artifact. The returned mapping has ``dia``, ``ind``, and ``tri``
        tensors with one row per input and one column per task label.

        Raises:
            ValueError: If the batch omits ``extra_features``.
        """
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state[:, 0, :]
        bert_output = self.bert_norm(bert_output)
        if extra_features is None:
            raise ValueError("MultiTaskModel requires extra_features in each batch.")
        extra_features_scaled = self.extra_feature_mlp(extra_features)
        head_input = self.projection_head_base(
            torch.cat([bert_output, extra_features_scaled], dim=-1)
        )
        return {
            "dia": self.diagnoses_head(head_input),
            "ind": self.indications_head(head_input),
            "tri": self.triage_head(head_input),
        }
