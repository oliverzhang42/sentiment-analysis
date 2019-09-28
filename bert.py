#https://colab.research.google.com/drive/1ZQvuAVwA3IjybezQOXnrXMGAnMyZRuPU#scrollTo=VVJDXVZRJF13

import h5py
import numpy as np
import torch
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_parser(text):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    return indexed_tokens, segments_ids

def bert_converter(model, indexed_tokens, segments_ids):
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    
    # TODO: Currently only considering the last embedding layer!!
    encoded_layers = encoded_layers[-1].detach().cpu().numpy()
    return encoded_layers

def save(name, obj):
    hf = h5py.File('{}.h5'.format(name), 'w')
    dset = hf.create_dataset('dataset_1', (25000, 150, 768), dtype='float32')

    for i in range(len(obj)):
        dset[i] = obj[i]
    
    hf.close()

import pudb; pudb.set_trace()

'''
train = pd.read_csv("labeledTrainData.tsv", sep='\t')
test = pd.read_csv("testData.tsv", sep='\t')

train_labels = train['sentiment']
np.save("train_labels", train_labels)

clean_train = []
clean_test = []

for i in range(len(train["review"])):
    cleaned = bert_parser(train["review"][i][0:512])
    clean_train.append(cleaned)

for i in range(len(test["review"])):
    cleaned = bert_parser(test["review"][i][0:512])
    clean_test.append(cleaned)

np.save("bert_pretrain", clean_train)
np.save("bert_pretest", clean_test)
#'''

clean_train = np.load("bert_pretrain.npy", allow_pickle=True)
clean_test = np.load("bert_pretest.npy", allow_pickle=True)

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

bert_train = []
bert_test = []

'''
for i in range(len(clean_train)):
    indexed_tokens, segments_ids = clean_train[i]
    encoding = bert_converter(model, indexed_tokens, segments_ids)
    encoding = encoding[0, 0:150, :]
    
    if encoding.shape[0] < 150:
        padding = np.zeros((150-encoding.shape[0], 768))
        encoding = np.concatenate((padding, encoding))

    bert_train.append(encoding)

save("bert_train", bert_train)
del bert_train
'''

for i in range(len(clean_test)):
    indexed_tokens, segments_ids = clean_test[i]
    encoding = bert_converter(model, indexed_tokens, segments_ids)
    encoding = encoding[0, 0:150, :]

    if encoding.shape[0] < 150:
        padding = np.zeros((150-encoding.shape[0], 768))
        encoding = np.concatenate((padding, encoding))

    bert_test.append(encoding)

np.save("bert_test", bert_test)
