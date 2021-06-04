import torch
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import time
import datetime    
from sklearn.metrics import precision_score, recall_score, fbeta_score


#Using the GPU for modelling if available else CPU
def gpu_call():
    if torch.cuda.is_available():      
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = t-a  # free inside reserved

    print('Reserved: ', r)
    print('Allocated: ', a)
    print('Total :', t)
    print('Memory : ', f)
    
    return device


#Calculating time completed for an epoch
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
    

#Creating the Train and Validation Dataloaders
def dataset(X_train, X_val, y_train, y_val):
    print("Preprocessing Begins...")
    train_inputs, train_masks = preprocessing_for_bert(X_train)
    val_inputs, val_masks = preprocessing_for_bert(X_val)
    print("Preprocessing Ends...")
    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_labels = torch.tensor(y_val,dtype=torch.long)
    print("Dataloader Creation Begins...")
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    validation_data = TensorDataset(val_inputs, val_masks, val_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
    print("Dataloader Creation Ends...")
    return train_dataloader, validation_dataloader
    
#Creating the Test DataLoader    
def test_dataset(X_test, y_test):
    print("Preprocessing Begins...")
    test_inputs, test_masks = preprocessing_for_bert(X_test)
    print("Preprocessing Ends...")
    test_labels = torch.tensor(y_test,dtype=torch.long)
    print("Dataloader Creation Begins...")
    test_data = TensorDataset(test_inputs, test_masks,test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    print("Dataloader Creation Ends...")
    return test_dataloader
    

#Loading the BertForSequenceClassification Model and the optimizer     
def model_load(train_dataloader, epochs):
    model = BertForSequenceClassification.from_pretrained( "textattack/bert-base-uncased-SST-2",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification.
        )
    print("Model is Loaded")
    model.cuda()
    optimizer = AdamW(model.parameters(),
        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
        )
        
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
         optimizer,
         num_warmup_steps=0,
         num_training_steps=total_steps
            )
        
    return model, optimizer, scheduler
    
