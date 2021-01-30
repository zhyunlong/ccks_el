from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.optim import Adam
import torch
import os
import logging
from .loss import loss_fn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ccks_el.util import Checkpoint
import torch.nn.functional as F
from .data_collate import data_collator

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, model, train_dataset, device='cpu',
                 epoch=3, batch_size=16, learning_rate=0.01, expt_dir="expt/",
                 do_eval=False, eval_dataset=None, eval_batch_size=16,
                 gradient_accumulation_steps=5, logging_steps=10, evaluate_steps=500, save_steps=500):
        logger.info("batch size {}".format(batch_size))
        logger.info("save steps {}".format(save_steps))
        logger.info("logging steps {}".format(logging_steps))
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.do_eval = do_eval
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.logging_steps = logging_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.evaluate_steps = evaluate_steps
        self.save_steps = save_steps
        self.expt_dir = expt_dir

    def get_train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        dataloader = DataLoader(self.train_dataset, sampler=train_sampler, collate_fn=data_collator, batch_size=self.batch_size)
        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        eval_sampler = RandomSampler(eval_dataset)
        dataloader = DataLoader(eval_dataset, sampler=eval_sampler, collate_fn=data_collator, batch_size=self.eval_batch_size)
        return dataloader

    def get_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def evaluate(self, model, eval_dataset):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self._prediction_loop(model, eval_dataloader, description="Evaluation")

    def _prediction_loop(self, model, dataloader, description):
        model.eval()
        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        all_predict_kb_label = []
        all_truth_kb_label = []
        all_predict_type = []
        all_truth_type = []
        for inputs in dataloader:
            text_a_ids = inputs['text_a_ids'].to(self.device, dtype=torch.long)
            text_a_attention_mask = inputs['text_a_attention_mask'].to(self.device, dtype=torch.long)
            text_b_ids = inputs['text_b_ids'].to(self.device, dtype=torch.long)
            text_b_attention_mask = inputs['text_b_attention_mask'].to(self.device, dtype=torch.long)
            truth_kb_label = inputs['label'].to(self.device, dtype=torch.float)
            truth_type = inputs['type'].to(self.device, dtype=torch.long)
            eval_losses: List[float] = []
            with torch.no_grad():
                kb_score, type_logits = model(text_a_ids, text_a_attention_mask, text_b_ids, text_b_attention_mask)
                #get loss
                truth_kb_label = truth_kb_label.reshape(-1,1)
                mask = truth_kb_label.ne(-1)
                kb_score_select = torch.masked_select(kb_score, mask)
                kb_label_select = torch.masked_select(truth_kb_label, mask)
                loss = loss_fn(kb_score_select, type_logits, kb_label_select, truth_type)
                eval_losses += [loss.mean().item()]
                
                #logger.info("text ids {}".format(text_a_ids[:10]))
                logger.info("type logits {}".format(type_logits))

                #get type  score
                type_score = F.log_softmax(type_logits, dim=-1) 

                #logger.info("type score {}".format(type_score.shape))

                type_score = type_score.argmax(dim=-1)
                
                
                kb_score_np = kb_score_select.flatten().cpu().numpy()
                kb_score_np[kb_score_np>=0.5]=1
                kb_score_np[kb_score_np<0.5]=0
                kb_score_np = kb_score_np.astype(int)
                all_predict_kb_label.extend(kb_score_np.tolist())
                all_truth_kb_label.extend(kb_label_select.flatten().cpu().numpy().astype(int).tolist())
                all_predict_type.extend(type_score.flatten().cpu().numpy().tolist())
                all_truth_type.extend(truth_type.flatten().cpu().numpy().tolist())

        from sklearn.metrics import accuracy_score
        #logger.info(all_truth_type[:10])
        #logger.info(all_predict_type[:10])
        #logger.info(all_truth_kb_label[:10])
        #logger.info(all_predict_kb_label[:10])
        kb_acc = accuracy_score(all_truth_kb_label, all_predict_kb_label)
        type_acc = accuracy_score(all_truth_type, all_predict_type)
        metrics = {}
        metrics["kb_acc"] = kb_acc
        metrics["type_acc"] = type_acc
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)
        msg = ""
        for key in metrics:
            msg = msg + "{} : {};   ".format(key, metrics[key])
        logger.info(msg)
        return metrics

    def train(self):
        train_dataloader = self.get_train_dataloader()
        optimizer = self.get_optimizers()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", self.epoch)
        model = self.model
        tr_loss = 0
        logging_loss = 0
        model.zero_grad()
        self.global_step = 0
        for epoch in range(self.epoch):
            for step, inputs in enumerate(train_dataloader):
                tr_loss += self._training_step(self.model, inputs)
                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(train_dataloader) <= self.gradient_accumulation_steps
                        and (step + 1) == len(train_dataloader)
                ):
                    #logger.info("update")
                    optimizer.step()
                    model.zero_grad()
                self.global_step += 1
                if self.global_step % self.logging_steps == 0:
                    logging_loss = (tr_loss - logging_loss) / self.logging_steps
                    logger.info("epoch {} step {}, train loss : {}".format(str(epoch), str(step), logging_loss))
                    logging_loss = tr_loss
                if self.do_eval and self.global_step % self.evaluate_steps == 0: 
                    self.evaluate(model, self.eval_dataset)
                if self.global_step % self.save_steps == 0:
                    logger.info("saving  model")
                    Checkpoint(model=model,
                               optimizer=optimizer,
                               epoch=epoch, step=self.global_step).save(self.expt_dir)



    def _training_step(self, model, inputs):
        model.train()
        text_a_ids = inputs['text_a_ids'].to(self.device, dtype=torch.long)
        text_a_attention_mask = inputs['text_a_attention_mask'].to(self.device, dtype=torch.long)
        text_b_ids = inputs['text_b_ids'].to(self.device, dtype=torch.long)
        text_b_attention_mask = inputs['text_b_attention_mask'].to(self.device, dtype=torch.long)
        truth_kb_label = inputs['label'].to(self.device, dtype=torch.float)
        truth_type = inputs['type'].to(self.device, dtype=torch.long)
        kb_score, type_logits = self.model(text_a_ids, text_a_attention_mask, text_b_ids, text_b_attention_mask)
        #logger.info(type_logits.shape)
        #logger.info(type_logits[:5])
        truth_kb_label = truth_kb_label.reshape(-1,1)
        mask = truth_kb_label.ne(-1)
        kb_score_select = torch.masked_select(kb_score, mask)
        kb_label_select = torch.masked_select(truth_kb_label, mask)
        loss = loss_fn(kb_score_select, type_logits, kb_label_select, truth_type)
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        return loss.item()
