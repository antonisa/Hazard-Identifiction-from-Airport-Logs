import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
import random
import metrics
import warnings
import time
import datetime
warnings.simplefilter('ignore')

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
batch_size = 48


#Function for training our model
def train(model, optimizer, train_dataloader,scheduler):
    

    t0 = time.time()
    total_loss = 0

    model.train()
    for step, batch in enumerate(train_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()        

        outputs = model(b_input_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)


        loss = outputs.loss

        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            



    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(utils.format_time(time.time() - t0)))

    print("")
    
    return avg_train_loss
    


#Function for validating our model using the F-Score as our validation metric    
def validation(model,validation_dataloader):
    print("Running Validation...")
    t0 = time.time()
    predictions,actual = [],[]
    model.eval()

    total_loss, eval_recall = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch


        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            attention_mask=b_input_mask,
                           labels = b_labels)

            logits = outputs.logits

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            loss = outputs.loss

            total_loss += loss.item()
            predictions.append(pred_flat)
            actual.append(labels_flat)
    predictions = np.concatenate(predictions)
    actual = np.concatenate(actual)
    avg_val_loss = total_loss / len(train_dataloader)            
    p,r,f,a,fp,fr = metrics.metrics(actual,predictions)
    
    

    print("")
    print("  Average val loss: {0:.2f}".format(avg_val_loss))

    print("  F-Score: {0:.2f}".format(f))
    print("  Validation took: {:}".format(utils.format_time(time.time() - t0)))
    
    return f, avg_val_loss, nb_eval_steps
    

#Function to predict the results over the test set after training
def predict(test_dataloader, model):

    predictions,actual = [],[]
    model.eval()
    
    for batch in test_dataloader:

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch


        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            attention_mask=b_input_mask)

            logits = outputs.logits

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            predictions.append(pred_flat)
            actual.append(labels_flat)
    
    predictions = np.concatenate(predictions)
    actual = np.concatenate(actual)
    return predictions,actual
            

    