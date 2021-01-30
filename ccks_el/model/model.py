import torch.nn as nn
from transformers import *
import logging
import torch

class Model(nn.Module):
    def __init__(self, pretrain_path, type_nums):
        super(Model, self).__init__()
        self.pretrain_name = pretrain_path
        self.pretrain_model = AutoModel.from_pretrained(pretrain_path)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.kb_linear = nn.Linear(768*2, 1)
        self.kb_head = nn.Sigmoid()
        self.type_head = nn.Linear(768, type_nums)

    def forward(self, text_a_ids, text_a_attention_mask, text_b_ids, text_b_attention_mask):
        # text_a_feat is pooled feature
        _, text_a_feat = self.pretrain_model(text_a_ids, attention_mask=text_a_attention_mask)
        text_a_feat = self.drop1(text_a_feat)
        _, text_b_feat = self.pretrain_model(text_b_ids, attention_mask=text_b_attention_mask)
        text_b_feat = self.drop2(text_b_feat)
        kb_query = torch.cat((text_a_feat, text_b_feat), -1)
        kb_query = self.kb_linear(kb_query)
        kb_score = self.kb_head(kb_query)
        type_logits = self.type_head(text_a_feat)
        return kb_score, type_logits




