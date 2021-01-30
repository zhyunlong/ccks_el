import torch
from transformers import *
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", do_lower_case=False)

def data_collator(features):
    text_a  = [sample["text_a"] for sample in features]
    text_b = [sample["text_b"] for sample in features]
    label = [sample["label"] for sample in features]
    type = [sample["type"] for sample in features]
    ent_id = [sample["ent_id"] for sample in features]
    batch = {}
    text_a_feats = tokenizer.batch_encode_plus(text_a, padding='longest')
    batch["text_a_ids"] = torch.tensor(text_a_feats['input_ids'])
    batch["text_a_attention_mask"] = torch.tensor(text_a_feats['attention_mask'])
    text_b_feats = tokenizer.batch_encode_plus(text_b, padding='longest')
    batch["text_b_ids"] = torch.tensor(text_b_feats['input_ids'])
    batch["text_b_attention_mask"] = torch.tensor(text_b_feats['attention_mask'])
    batch["label"] = torch.tensor(label)
    batch["type"] = torch.tensor(type)
    batch["ent_id"] = ent_id
    return batch