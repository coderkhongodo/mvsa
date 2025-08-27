import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import TxtOnly_Dataset
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from text_processing import Tweet_Preprocessing
from transformers import AutoTokenizer, AutoModel
import argparse
from config import *
from utils import prepare_data, prepare_text_data, agg_metrics_val, get_optimizer_params, loss_correction
from transformers import get_linear_schedule_with_warmup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ------ LOGGING-----------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )
logger = logging.getLogger(__name__)
# ------ MODELS-----------------------------------------------------


class BERT(nn.Module):
    def __init__(self,model_dir, num_labels, dropout=0.1):
        super(BERT, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        hidden_size = getattr(self.bert_model.config, 'hidden_size', txt_feat_size)
        self.linear = nn.Linear(hidden_size, num_labels)
        
    def forward(self,ids,mask,token_type_ids):
        outputs = self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        last_hidden = outputs[0]
        cls_output = last_hidden[:,0,:]
        dropout_output = self.dropout(cls_output)
        linear_output = self.linear(dropout_output)
        return linear_output
   
class BERNICE(nn.Module):
    def __init__(self,model_dir, num_labels, dropout=0.1):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        hidden_size = getattr(self.bert_model.config, 'hidden_size', txt_feat_size)
        self.linear = nn.Linear(hidden_size, num_labels)
        
    def forward(self,ids,mask):
        outputs = self.bert_model(ids,attention_mask=mask, return_dict=False)
        last_hidden = outputs[0]
        cls_output = last_hidden[:,0,:]
        dropout_output = self.dropout(cls_output)
        linear_output = self.linear(dropout_output)
        return linear_output

class RoBERTa(nn.Module):
    def __init__(self,model_dir, num_labels, dropout=0.1):
        super(RoBERTa, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        hidden_size = getattr(self.bert_model.config, 'hidden_size', txt_feat_size)
        self.linear = nn.Linear(hidden_size, num_labels)
        
    def forward(self,ids,mask):
        outputs = self.bert_model(ids,attention_mask=mask, return_dict=False)
        last_hidden = outputs[0]
        cls_output = last_hidden[:,0,:]
        dropout_output = self.dropout(cls_output)
        linear_output = self.linear(dropout_output)
        return linear_output
   
class TextModel(object):
    """
    TextModel class
    """
    def __init__(self, config, model_name, freeze = False):
        """ Initialization """
        self.batch_size = config.batch_size
        self.num_labels = config.num_labels
        self.model_name = model_name
        self.model_dir = MODEL_DIR_DICT[self.model_name]
        self.max_length = config.max_length
        self.dropout = config.dropout
        self.use_loss_correction = config.use_loss_correction
      
        # tokenizer
        if self.model_name == "bernice":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, model_max_length=self.max_length) 
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir) 
        
        # model
        if self.model_name == "roberta":
            self.model = RoBERTa(self.model_dir, self.num_labels, dropout=self.dropout)
        elif self.model_name in {"bernice","phobert"}:
            self.model = BERNICE(self.model_dir, self.num_labels, dropout=self.dropout)
        else:
            self.model = BERT(self.model_dir, self.num_labels, dropout=self.dropout)
        
        if freeze:
            for param in self.model.bert_model.parameters():
                param.requires_grad = False
        
        # Initialize weights properly
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=0.1)  # Use smaller gain
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        # Apply weight initialization to the classification head
        self.model.linear.apply(init_weights)
        
        self.model.to(device)
        print(self.model)

        self.softmax = nn.Softmax(dim=1)
    
    def load_data(self,data, testing=False ,eval_txt_test=False, task_name=None):

        train, y_vector_tr, val, y_vector_val, test, y_vector_te, class_weights, image_adds = prepare_data(data, self.num_labels, testing=testing)
        use_norm = False if task_name == "viclick" else True
        tr_dataset = TxtOnly_Dataset(self.model_name, train.tweet_id.values,train.text.values,y_vector_tr,self.tokenizer, self.max_length,task_name, normalization=use_norm)
        train_loader = DataLoader(tr_dataset, batch_size=self.batch_size,shuffle=True)
        val_dataset = TxtOnly_Dataset(self.model_name, val.tweet_id.values, val.text.values,y_vector_val,self.tokenizer, self.max_length,task_name, normalization=use_norm)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,shuffle=False)
        te_dataset = TxtOnly_Dataset(self.model_name, test.tweet_id.values, test.text.values,y_vector_te,self.tokenizer, self.max_length,task_name, normalization=use_norm)
        test_loader = DataLoader(te_dataset, batch_size=self.batch_size,shuffle=False)
        if eval_txt_test:
            # text_only
            txt_test, y_txt_te = prepare_text_data(num_labels=self.num_labels, testing=testing)
            txt_te_dataset = TxtOnly_Dataset(self.model_name, txt_test.tweet_id.values, txt_test.text.values, y_txt_te, self.tokenizer, self.max_length, task_name)
            txt_te_loader = DataLoader(txt_te_dataset, batch_size=self.batch_size,shuffle=False) 
        else:
            txt_te_loader= None
        return train_loader, val_loader, test_loader, class_weights, txt_te_loader
    

    def train(self,dataloader,val_dataloader,epochs,loss_fn,lr,weight_decay,
    te_dataloader=None,model_path=None,val_filename=None,te_filename=None, 
    warmup_steps=100, gradient_clip=1.0, scheduler_type='linear'):  

        #Initialize Optimizer
        named_parameters = self.model.named_parameters()
        optimizer_params = get_optimizer_params(named_parameters, weight_decay, lr)
        optimizer = optim.AdamW(optimizer_params, lr=lr)
        
        # Initialize scheduler per batch
        total_steps = len(dataloader) * epochs
        if scheduler_type == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        elif scheduler_type == 'warmup_cosine':
            # Simple warmup + cosine scheduler
            # Start with very low LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.01  # Start with 1% of target LR
            
            # Use OneCycleLR for better control
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=lr,
                total_steps=total_steps,
                pct_start=warmup_steps/total_steps,
                anneal_strategy='cos'
            )
        elif scheduler_type == 'warmup_linear':
            # Backward-compat custom warmup-linear
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.01
            if isinstance(warmup_steps, float) and warmup_steps < 1.0:
                warmup_steps_int = max(1, int(total_steps * warmup_steps))
            else:
                warmup_steps_int = int(warmup_steps)
            scheduler = {
                'type': 'warmup_linear', 
                'warmup_steps': warmup_steps_int, 
                'target_lr': lr, 
                'total_steps': total_steps,
                'decay_steps': max(1, total_steps - warmup_steps_int)
            }
        elif scheduler_type == 'hf_warmup_linear':
            # HuggingFace warmup linear scheduler
            if isinstance(warmup_steps, float) and warmup_steps < 1.0:
                warmup_steps_int = max(1, int(total_steps * warmup_steps))
            else:
                warmup_steps_int = int(warmup_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps_int, num_training_steps=total_steps)
        elif scheduler_type == 'onecycle':
            # Use OneCycleLR for better control
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=lr,
                total_steps=total_steps,
                pct_start=warmup_steps/total_steps,
                anneal_strategy='cos'
            )
        else:
            scheduler = None
        
        self.model.train()
        res_val, res_te = [], []
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        global_step = 0
        
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch, dl in loop:
                ids=dl['ids'].to(device)
                mask= dl['mask'].to(device)
                label=dl['target'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if self.model_name not in {"roberta", "bernice", "phobert"}:
                    token_type_ids=dl['token_type_ids'].to(device)
                    output=self.model(
                        ids=ids,
                        mask=mask,
                        token_type_ids=token_type_ids)
                else:
                    # roberta, bernice
                    output=self.model(
                        ids=ids,
                        mask=mask)
                # targets: move to device and build class indices for CE
                label = label.to(output.device)
                if label.dim() == 2 and label.size(1) == self.num_labels:
                    label_idx = torch.argmax(label, dim=1)
                else:
                    label_idx = label
                # compute loss
                if self.use_loss_correction:
                    # loss_correction expects one-hot labels
                    loss = loss_correction(T, loss_fn, output, label)
                else:
                    loss = loss_fn(output, label_idx)
                
                # Check for NaN or inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or inf loss detected: {loss.item()}, skipping batch")
                    continue
                
                # Check for extremely high loss
                if loss.item() > 10.0:
                    logger.warning(f"Very high loss detected: {loss.item():.4f}")
                # backward loss
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                # Check for gradient explosion
                if grad_norm > gradient_clip * 0.9:  # If gradient norm is close to clip value
                    logger.warning(f"Large gradients detected: {grad_norm:.4f}")
                
                optimizer.step()
                
                # Update learning rate after each batch
                global_step += 1
                if scheduler is not None:
                    if isinstance(scheduler, dict) and scheduler.get('type') == 'warmup_linear':
                        # WarmupLinear scheduler
                        if global_step <= scheduler['warmup_steps']:
                            # Linear warmup from 1% to 100% of target LR
                            warmup_factor = global_step / scheduler['warmup_steps']
                            current_lr = scheduler['target_lr'] * (0.01 + 0.99 * warmup_factor)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = current_lr
                        else:
                            # After warmup, linear decay to 10% of target LR
                            decay_step = global_step - scheduler['warmup_steps']
                            decay_factor = max(0.1, 1.0 - (decay_step / scheduler['decay_steps']) * 0.9)
                            current_lr = scheduler['target_lr'] * decay_factor
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = current_lr
                    else:
                        scheduler.step()
                
                # predictions (per-batch) used only for epoch aggregates
                pred = torch.argmax(self.softmax(output), dim=1)
                target = label_idx
                num_correct = torch.sum(pred == target).item()
                num_samples = label_idx.size(0)

                # Update epoch statistics
                epoch_loss += loss.item()
                epoch_correct += num_correct
                epoch_total += num_samples

                # Update progress bar with epoch-level running averages only
                current_lr = optimizer.param_groups[0]['lr']
                running_avg_loss = epoch_loss / float(batch + 1)
                running_avg_acc = epoch_correct / float(epoch_total) if epoch_total > 0 else 0.0
                loop.set_description(f'Epoch={epoch+1}/{epochs} (Step {global_step})')
                loop.set_postfix(loss=f'{running_avg_loss:.4f}', acc=f'{running_avg_acc:.3f}', lr=f'{current_lr:.2e}')
            
            # Log epoch summary
            avg_epoch_loss = epoch_loss / len(dataloader)
            avg_epoch_acc = epoch_correct / epoch_total
            logger.info(f"Epoch {epoch+1} Summary - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_acc:.4f}")
            
            # Learning rate is now updated per batch
            
            # predict val
            print("val")
            res_val_d = self.eval(val_dataloader,loss_fn)
            res_val_d["epoch"] = epoch
            res_val.append(res_val_d)
            
            # Early stopping
            current_val_loss = res_val_d["loss"]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                # Save best model
                if model_path != None:
                    torch.save(self.model.state_dict(), model_path.replace('.pth', '_best.pth'))
                    logger.info("Best model saved with val_loss: {:.4f}".format(best_val_loss))
            else:
                patience_counter += 1
                logger.info("Patience counter: {}/{}".format(patience_counter, patience))
            
            if patience_counter >= patience:
                logger.info("Early stopping triggered after {} epochs".format(epoch + 1))
                break
                
            if val_filename != None and (epoch%2 == 0 or epoch==epochs-1):
                logger.info("Compute metrics (val)")
                metrics_val = agg_metrics_val(res_val, metric_names, self.num_labels)
                pd.DataFrame(metrics_val).to_csv(val_filename,index=False)
                logger.info("{} saved!".format(val_filename))

            if te_dataloader != None:
                # predict test
                print("test")
                res_te_d = self.eval(te_dataloader,loss_fn)
                res_te_d["epoch"] = epoch
                res_te.append(res_te_d)
                if te_filename != None and (epoch%2 == 0 or epoch==epochs-1):
                    logger.info("Compute metrics (test)")
                    metrics_te = agg_metrics_val(res_te, metric_names, self.num_labels)
                    pd.DataFrame(metrics_te).to_csv(te_filename,index=False)
                    logger.info("{} saved!".format(te_filename))

        if model_path != None:
            torch.save(self.model.state_dict(), model_path)
            logger.info("{} saved".format(model_path))

    def eval(self, dataloader, loss_fn):
        predictions = []
        labels = []
        data_ids = []
        self.model.eval()

        total_correct = 0
        total_count = 0
        total_loss_sum = 0.0

        loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for batch, dl in loop:
            ids = dl['ids'].to(device)
            mask = dl['mask'].to(device)
            label = dl['target'].to(device)
            data_id = dl['data_id'].to(device)
            if self.model_name not in {"roberta", "bernice", "phobert"}:
                token_type_ids = dl['token_type_ids'].to(device)
            # Compute logits
            with torch.no_grad():
                if self.model_name not in {"roberta", "bernice", "phobert"}:
                    output = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                else:
                    # roberta, bernice
                    output=self.model(
                        ids=ids,
                        mask=mask,
                        )
            # Build class indices for CE
            if label.dim() == 2 and label.size(1) == self.num_labels:
                label_idx = torch.argmax(label, dim=1)
            else:
                label_idx = label
            # Compute loss
            if self.use_loss_correction:
                # loss_correction expects one-hot (float) labels
                label_onehot = label.to(dtype=output.dtype)
                loss = loss_correction(T, loss_fn, output, label_onehot)
                batch_loss = loss.item()
                num_samples = label.size(0)
                total_loss_sum += batch_loss * num_samples
            else:
                loss = loss_fn(output, label_idx)
                num_samples = label_idx.size(0)
                total_loss_sum += loss.item() * num_samples
            # Predictions and accuracy
            pred = torch.argmax(self.softmax(output), dim=1)
            target = label_idx
            total_correct += torch.sum(pred == target).item()
            total_count += num_samples
            # Save predictions and targets
            predictions += pred
            labels += target
            data_ids += data_id

        # Compute dataset-level averages
        eval_loss = total_loss_sum / float(total_count) if total_count > 0 else 0.0
        eval_acc = (total_correct / float(total_count)) * 100 if total_count > 0 else 0.0

        print(f'loss: {eval_loss:.4f} acc: {eval_acc:.4f}\n')

        y_pred = torch.stack(predictions)
        y = torch.stack(labels)
        data_ids = torch.stack(data_ids)

        res = {
            "data_id": data_ids,
            "loss": eval_loss,
            "predictions": y_pred,
            "labels": y
        }

        return res




        










