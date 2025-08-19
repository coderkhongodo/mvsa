import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import ImgOnly_Dataset, ImgOnlyCNN_Dataset
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel
import argparse
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics.classification import BinaryF1Score
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from config import *
from utils import get_conv_model, prepare_data, agg_metrics_val, get_optimizer_params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ------ LOGGING-----------------------------------------------------
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO,
                   )
logger = logging.getLogger(__name__)
# ------ MODELS------------------------------------------------------

class BEiT(nn.Module):
    def __init__(self, model_dir, num_labels, dropout=0.1):
        super(BEiT, self).__init__()
        self.model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(img_feat_size, num_labels)
        
    def forward(self,pixel_values):
        _,pooled_output= self.model(pixel_values, return_dict=False)
        linear_output = self.linear(pooled_output)
        return linear_output

class DEiT(nn.Module):
    def __init__(self, model_dir, num_labels, dropout=0.1):
        super(DEiT, self).__init__()
        self.model = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(img_feat_size, num_labels)
        
    def forward(self,pixel_values):
        _,pooled_output = self.model(pixel_values, return_dict=False)
        linear_output = self.linear(pooled_output)
        return linear_output

class CNN(nn.Module):
    def __init__(self, model_dir, model_name, num_labels, feature_extract = False):
        super(CNN, self).__init__()
        self.model_name = model_name
        self.feature_extract = feature_extract
        self.net = get_conv_model(self.model_name)
        self.net.load_state_dict(torch.load(model_dir))
        self.set_parameter_requires_grad()
        num_ftrs = self.net.fc.in_features # 2048
        self.net.fc = nn.Linear(num_ftrs, num_labels)
        # params to optimize
        print("Params to learn:")
        params_to_update = self.net.parameters()
        if self.feature_extract:
            params_to_update = []
            for name,param in self.net.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in self.net.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        self.params_to_update = params_to_update
        self.net = self.net.to(device)
        
    
    def set_parameter_requires_grad(self):
        if self.feature_extract:
            for param in self.net.parameters():
                param.requires_grad = False

    def forward(self,x):
        x_out = self.net(x)
        return x_out
  
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class ImageModel(object):
    """
    ImageModel class
    """
    def __init__(self, batch_size, num_labels, model_name, conv_att = False, feature_extract=False):
        """ Initialization """
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.model_name = model_name
        self.cnn = self.model_name in {"resnet50","resnet152"}
        self.model_dir = MODEL_DIR_DICT[self.model_name]
        if not self.cnn:
            # feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_dir)
        # model
        self.set_model(conv_att=conv_att, feature_extract=feature_extract)
        self.softmax = nn.Softmax(dim=1)
    
    def set_model(self, conv_att = False, feature_extract=False):
        if self.cnn:
            if conv_att:
                self.model = CNNAtt(model_dir=self.model_dir, model_name=self.model_name, num_labels=self.num_labels,
                feature_extract=feature_extract)   
            else:
                self.model = CNN(model_dir=self.model_dir, model_name=self.model_name, num_labels=self.num_labels,
                feature_extract=feature_extract) 
            
        else:
            if self.model_name == "vit":
                self.model = AutoModelForImageClassification.from_pretrained(self.model_dir, 
                        num_labels=self.num_labels)
            elif self.model_name == "beit":
                self.model = BEiT(self.model_dir, self.num_labels)
            else:
                self.model = DEiT(self.model_dir, self.num_labels)
        self.model.to(device)
        print(self.model)       
    
    def load_data(self, data, img_file_fmt, testing=False, task_name=None):
        train, y_vector_tr, val, y_vector_val, test, y_vector_te, class_weights, _ = prepare_data(data, self.num_labels, testing=testing)
        if self.cnn:
            tr_dataset = ImgOnlyCNN_Dataset(train.tweet_id.values,y_vector_tr,img_file_fmt,task_name=task_name)
            val_dataset = ImgOnlyCNN_Dataset(val.tweet_id.values,y_vector_val,img_file_fmt,task_name=task_name)
            te_dataset = ImgOnlyCNN_Dataset(test.tweet_id.values,y_vector_te,img_file_fmt,task_name=task_name)
        else:
            tr_dataset = ImgOnly_Dataset(train.tweet_id.values,y_vector_tr,self.feature_extractor,img_file_fmt,
                                         task_name=task_name)
            val_dataset = ImgOnly_Dataset(val.tweet_id.values,y_vector_val,self.feature_extractor,img_file_fmt,
                                          task_name=task_name)
            te_dataset = ImgOnly_Dataset(test.tweet_id.values,y_vector_te,self.feature_extractor,img_file_fmt,
                                         task_name=task_name)

        train_loader = DataLoader(tr_dataset, batch_size=self.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(te_dataset, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader, class_weights
    
    def train(self,dataloader,val_dataloader,epochs,loss_fn,lr,weight_decay,
              te_dataloader=None,model_path=None,val_filename=None,te_filename=None,
              warmup_steps=100, gradient_clip=0.5, scheduler_type='warmup_linear'):
        #Initialize Optimizer
        named_parameters = self.model.named_parameters()
        optimizer_params = get_optimizer_params(named_parameters, weight_decay, lr)
        optimizer = optim.AdamW(optimizer_params, lr=lr)

        # Scheduler per-batch (like text)
        total_steps = len(dataloader) * epochs
        if scheduler_type == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        elif scheduler_type == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=warmup_steps/total_steps, anneal_strategy='cos')
        elif scheduler_type == 'warmup_linear':
            scheduler = {'type': 'warmup_linear', 'warmup_steps': warmup_steps, 'target_lr': lr, 'total_steps': total_steps, 'decay_steps': max(1, total_steps - warmup_steps)}
            for pg in optimizer.param_groups:
                pg['lr'] = lr * 0.01
        else:
            scheduler = None

        self.model.train()
        res_val, res_te = [], []
        best_val_loss = float('inf')
        patience, patience_counter = 3, 0
        global_step = 0
        for  epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            for batch, dl in loop:
                pixel_values = torch.squeeze(dl['pixel_values'])
                if len(pixel_values.size())<4:
                    pixel_values = torch.unsqueeze(pixel_values,0)
                pixel_values = pixel_values.to(device)
                #print("pixel values",pixel_values.size())
                label=dl['labels'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if self.cnn:
                    output = self.model(pixel_values)
                else:
                    output=self.model(
                            pixel_values=pixel_values,
                            )
                    if self.model_name == "vit":
                        output = output.logits

                # Targets as class indices for CrossEntropyLoss
                if label.dim() == 2 and label.size(1) == self.num_labels:
                    label_idx = torch.argmax(label, dim=1)
                else:
                    label_idx = label
                # compute loss
                loss = loss_fn(output, label_idx)
                # backward pass
                loss.backward()
                # clip & step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()
                # scheduler step per-batch
                global_step += 1
                if scheduler is not None:
                    if isinstance(scheduler, dict) and scheduler.get('type') == 'warmup_linear':
                        if global_step <= scheduler['warmup_steps']:
                            warmup_factor = global_step / scheduler['warmup_steps']
                            current_lr = scheduler['target_lr'] * (0.01 + 0.99 * warmup_factor)
                            for pg in optimizer.param_groups:
                                pg['lr'] = current_lr
                        else:
                            decay_step = global_step - scheduler['warmup_steps']
                            decay_factor = max(0.1, 1.0 - (decay_step / scheduler['decay_steps']) * 0.9)
                            current_lr = scheduler['target_lr'] * decay_factor
                            for pg in optimizer.param_groups:
                                pg['lr'] = current_lr
                    else:
                        scheduler.step()
                # predict
                pred = torch.argmax(self.softmax(output),dim=1)  
                target = label_idx         
                num_correct = torch.sum(pred==target).item()
                num_samples = label_idx.size(0)
                epoch_loss += loss.item()
                epoch_correct += num_correct
                epoch_total += num_samples
                # progress
                current_lr = optimizer.param_groups[0]['lr']
                running_avg_loss = epoch_loss / float(batch + 1)
                running_avg_acc = epoch_correct / float(epoch_total) if epoch_total>0 else 0.0
                loop.set_description(f'Epoch={epoch+1}/{epochs} (Step {global_step})')
                loop.set_postfix(loss=f'{running_avg_loss:.4f}', acc=f'{running_avg_acc:.3f}', lr=f'{current_lr:.2e}')
            
            # epoch summary
            avg_epoch_loss = epoch_loss / len(dataloader)
            avg_epoch_acc = epoch_correct / epoch_total
            logger.info(f"Epoch {epoch+1} Summary - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_acc:.4f}")

            # predict val
            res_val_d = self.eval(val_dataloader,loss_fn)
            res_val_d["epoch"] = epoch
            res_val.append(res_val_d)
            if val_filename != None and (epoch%2 == 0 or epoch==epochs-1):
                logger.info("Compute metrics (val)")
                metrics_val = agg_metrics_val(res_val, metric_names, self.num_labels)
                pd.DataFrame(metrics_val).to_csv(val_filename,index=False)
                logger.info("{} saved!".format(val_filename))
            # early stopping
            if res_val_d["loss"] < best_val_loss:
                best_val_loss = res_val_d["loss"]
                patience_counter = 0
                if model_path is not None:
                    torch.save(self.model.state_dict(), model_path.replace('.pth','_best.pth'))
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    logger.info("Early stopping")
                    break

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
        #return res_val

    def eval(self, dataloader, loss_fn):
        predictions = []
        labels = []
        data_ids = []
        loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
        self.model.eval()

        total_correct = 0
        total_count = 0
        total_loss_sum = 0.0

        for batch, dl in loop:
            pixel_values = torch.squeeze(dl['pixel_values']).to(device)
            if len(pixel_values.size()) < 4:
                pixel_values = torch.unsqueeze(pixel_values, 0)
            label = dl['labels'].to(device)
            data_id = dl['data_id'].to(device)
            # Compute logits
            with torch.no_grad():
                if self.cnn:
                    output = self.model(pixel_values)
                else:
                    output=self.model(pixel_values=pixel_values)
                    if self.model_name == "vit":
                        output = output.logits
            # Targets as class indices for CrossEntropyLoss
            if label.dim() == 2 and label.size(1) == self.num_labels:
                label_idx = torch.argmax(label, dim=1)
            else:
                label_idx = label
            # Compute loss (weighted by batch size)
            loss = loss_fn(output, label_idx)
            num_samples = label_idx.size(0)
            total_loss_sum += loss.item() * num_samples
            # Predictions and accuracy
            pred = torch.argmax(self.softmax(output), dim=1)
            total_correct += torch.sum(pred == label_idx).item()
            total_count += num_samples
            # Save predictions and targets
            predictions += pred
            labels += label_idx
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

    
