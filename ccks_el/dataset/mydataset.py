import torch
import pandas as pd
import json
import torchtext
from transformers import *



class Mydataset(torch.utils.data.Dataset):
    def __init__(self, path, model_name, type_json_map_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.dataframe = pd.read_csv(path, sep='\t', low_memory=False, lineterminator="\n")
        self.text_a = self.dataframe['text_a']
        self.text_b = self.dataframe['text_b']
        self.label = self.dataframe['label']
        self.type = self.dataframe['type']
        self.ent_id = self.dataframe['ent_id']
        json_file = open(type_json_map_path,"r")
        self.type_map = json.load(json_file)
        self.max_len = 512
        self.pad_tok = self.tokenizer.pad_token

    def __getitem__(self, idx):
        sample = dict()
        sample["text_a"] = self.text_a[idx]
        sample["text_b"] = self.text_b[idx]
        sample["label"] = self.label[idx]
        sample["type"] = self.type_map[self.type[idx]]
        sample["ent_id"] = self.ent_id[idx]
        return sample

    def __len__(self):
        return self.dataframe.shape[0]
