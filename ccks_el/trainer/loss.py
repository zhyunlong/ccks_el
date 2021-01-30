import torch
import logging

def loss_fn(kb_score, type_logits, kb_label, type, alpha=0.5):
    loss_1 = torch.nn.BCELoss()(kb_score, kb_label)
    loss_2 = torch.nn.CrossEntropyLoss()(type_logits, type)
    loss = alpha*loss_1 + (1-alpha)*loss_2
    return loss

