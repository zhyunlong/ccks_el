import os
import torch
import argparse
import logging
from torch import cuda
from ccks_el.dataset import Mydataset
from ccks_el.model import Model
from ccks_el.trainer import Trainer
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s' )

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', action='store', dest='batch_size', default=1,
                    help='train set batch size')
opt = parser.parse_args()


train_set = Mydataset('data/tiny_train.txt', 'bert-base-chinese', 'data/type_label_map.json')
eval_set = Mydataset('data/tiny_dev.txt', 'bert-base-chinese', 'data/type_label_map.json')
train_params = {
    'batch_size': opt.batch_size,
    'save_steps': 10000000,
    'do_eval': True,
    'eval_dataset':eval_set,
    'evaluate_steps':5,
    'device': 'cpu',
    #'device': 'cuda' if cuda.is_available() else 'cpu',
    'epoch': 1
}
bert_path = os.path.join("bert", "bert_chinese")
type_num = len(train_set.type_map)
model = Model(bert_path, type_num)
trainer = Trainer(model, train_set, **train_params)
trainer.train()

