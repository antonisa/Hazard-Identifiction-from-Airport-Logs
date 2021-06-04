import pandas as pd


#Dropping columns manually based on the train set
def drop_col(df, df_test):
	cols = df.columns[1:]
	cols_to_drop = []
	for col in df_test.columns[1:]:
		if col not in cols:
			cols_to_drop.append(col)
			
	cols_to_drop.append('MINIMUM VECTOR ALTITUDE ISSUE (MVA)')
    
    
    df_test.drop(cols_to_drop,axis=1,inplace=True)
    
    return df_test
    
#Creating the tokenizer and the max_sequence_length
def tokenization(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    seq_lens = df['ADAB SPREADSHEET NARRATIVES'].apply(lambda x:len(x.split()))

    seq_len_max = max(seq_lens)
    
    return tokenizer, seq_len_max
    
#Function that uses the encode_plus method to transform the data into the required format for BERT
def preprocessing_for_bert(data):
    
    input_ids = []
    attention_masks = []

    
    for sent in data:
        
        encoded_sent = tokenizer.encode_plus(
            text = sent,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=seq_len_max+5,       # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            truncation=True
            )
        
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks