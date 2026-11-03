"""
MMRT: Sequence Knowledge Distillation Loss
Knowledge distillation for temporal sequence modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceKnowledgeDistillation(nn.Module):
    """Knowledge distillation loss with temperature scaling."""

    def __init__(self, T=1):
        super().__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(
            prediction_logits[:, :, start_idx:] / self.T, dim=-1
        ).view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(
            ref_logits[:, :, start_idx:] / self.T, dim=-1
        ).view(-1, ref_logits.shape[2] - start_idx)
        return self.kdloss(prediction_logits, ref_probs) * self.T * self.T


# Alias for state_dict compatibility
SeqKD = SequenceKnowledgeDistillation
