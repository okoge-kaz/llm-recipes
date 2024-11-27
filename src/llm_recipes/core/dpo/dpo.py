from typing import Tuple
import torch
from torch import nn


CROSS_ENTROPY_IGNORE_IDX = -100


def get_batch_log_probs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = CROSS_ENTROPY_IGNORE_IDX,
) -> torch.FloatTensor:
    """
    Calculate log probabilities based on provided logits and labels.

    Args:
        logits (torch.FloatTensor): direct logits output of the model of shape (b, s, v)
        labels (torch.LongTensor): ground-truth labels to compute log probs with, shape (b, s).
            Label tokens with a value of label_pad_token_id are ignored.
        label_pad_token_id (int): token id to ignore in labels.

    Returns:
        Calculated log probs of shape (b, )

    Raises:
        ValueError: If logits and labels have different shapes.
    """

    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "Logits (batch and sequence length dim) and labels must have the same shape."
        )

    labels = labels[:, 1:].clone()  # type: ignore
    logits = logits[:, :-1, :]  # type: ignore
    loss_mask = labels != label_pad_token_id

    labels[labels == label_pad_token_id] = 0
    # take log-likelihood of the labels given our model
    per_token_log_probs = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    return (per_token_log_probs * loss_mask).sum(-1)  # type: ignore


def concatenated_forward(
    model: nn.Module, batch: dict[str, torch.Tensor], local_rank: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run forward pass of the model with chosen and rejected samples concatenated.

    Args:
        model (nn.Module): The model to be used for the forward pass.
        batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of input_ids and labels.

    Returns:
        Tuple of chosen log probs, rejected log probs, chosen logits, rejected logits.
    """
    concatenated_input_ids = torch.cat(
        [batch['chosen_input_ids'], batch['rejected_input_ids']], dim=0
    )
    concatenated_labels = torch.cat(
        [batch['chosen_labels'], batch['rejected_labels']], dim=0
    )
    concatenated_input_ids = concatenated_input_ids.to(local_rank)
    concatenated_labels = concatenated_labels.to(local_rank)

    # formed by concatenating an equal number of "chosen" and "rejected".
    len_chosen = concatenated_input_ids.shape[0] // 2

    all_logits = model(concatenated_input_ids).logits

    all_log_probs = get_batch_log_probs(
        all_logits, concatenated_labels  # type: ignore
    )

    chosen_log_probs = all_log_probs[:len_chosen]
    rejected_log_probs = all_log_probs[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    return (chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits)
