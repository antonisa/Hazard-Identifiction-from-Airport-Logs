import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
from sklearn.model_selection import KFold, train_test_split
import seaborn as sns
import statistics
import random
import warnings
warnings.simplefilter('ignore')

import utils, preproc, train, metrics

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
batch_size = 48

df = pd.read_pickle('df_new_training.pkl')
df_test = pd.read_pickle('/scratch/vparadk/df_test_1.pkl')
df_test = preproc.drop_col(df,df_test)
tokenizer, seq_len_max = preproc.tokenization(df)
device = utils.gpu_call()
results = pd.DataFrame(columns=['Precision','Recall','F1-Score','Accuracy','F-Score for Precision', 'F-Score for Recall', 'Count-POS'], dtype=float)


for col in df_test.columns[1:]:
    print('Column: ', col)
    X_test = df_test['ADAB SPREADSHEET NARRATIVES'].values
    X = df['ADAB SPREADSHEET NARRATIVES'].values
    epochs = 10
    y = df[col].values
    count_POS = len(np.where(y==1)[0])
    y_test = df_test[col].values
    model_name = '/scratch/vparadk/best_model_state.pt'
    kf = KFold(n_splits=5)
    dic = {}
    
    #Splitting the data into 5 train-test folds
    for train_index, test_index in kf.split(X):
        flag = 1        #This flag checks whether there were any positive samples for the given class in the test set 
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        res = np.where(y_train==1)
        if len(res[0]) < 10 and len(res[0]) > 1:
            duplicate = int((len(y_train)*.05)/len(res[0]))
            if duplicate > 0:
                X_train_new = np.array([])
                y_train_new = np.array([])
                X_train_new = np.repeat(X_train[res],duplicate)
                y_train_new = np.repeat(y_train[res], duplicate)
                X_train = np.append(X_train,X_train_new)
                y_train = np.append(y_train,y_train_new)
                
        if len(res[0]) == 0: 
            flag = 0
            old_eval = float('inf')
        else: old_eval = 0
        
        
        early_stopping_counter = 0
        train_dataloader, validation_dataloader = utils.dataset(X_train, X_val, y_train, y_val)
        model,optimizer, scheduler = utils.model_load(train_dataloader, epochs)
        
        for epoch_i in range(0, epochs):
            train_loss = train.train(model, optimizer, train_dataloader, scheduler)
            avg_eval_f1_score, avg_val_loss, nb_eval_steps = train.validation(model, validation_dataloader)
            if flag == 1:
                if avg_eval_f1_score > old_eval:
                    old_eval = avg_eval_f1_score
                    torch.save(model.state_dict(), model_name)   
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                if early_stopping_counter == 2:
                    break
            elif flag == 0:
                if avg_val_loss < old_eval:
                    old_eval = avg_val_loss
                    torch.save(model.state_dict(), model_name)   
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                if early_stopping_counter == 2:
                    break
        model.load_state_dict(torch.load(model_name))
        print("")
        print("Training complete!")
        test_dataloader = utils.test_dataset(X_test, y_test)
        predictions, actual = train.predict(test_dataloader, model)
        
        
        #Code that performs prediction for the given test set using five different train-validation sets i.e. bagging
        for i in range(len(predictions)):
            if i not in dic:
                dic[i] = [predictions[i]]
            else:
                dic[i].append(predictions[i])
                
    predictions = []            
    for i in dic:
        predictions.append(max(dic[i],key=dic[i].count))
        
        

    print("Testing complete!")
    p,r,f,a,fp,fr = metrics.metrics(actual,predictions)
    print('Column: ', col)
    print('Precision: ',p)
    print('Recall: ',r)
    print('F1-Score: ',f), 
    print('Accuracy: ',a)
    print('F-Score for Precision: ',fp)
    print('F-Score for Recall: ',fr)
    ser = pd.Series([p,r,f,a,fp,fr,count_POS], name=col, index=results_new.columns)
    results_new = results_new.append(ser)
    results_new['Count-POS'] = results['Count-POS'].apply(lambda x: len(x[0]))
    results_new.to_csv('/scratch/vparadk/results_new_train_oversample_avg_eval_loss_f1_score_test.csv', index = True, header=True)
    df_test.drop(col,axis=1,inplace=True)
    df_test.to_pickle("/scratch/vparadk/df_test_1.pkl")







