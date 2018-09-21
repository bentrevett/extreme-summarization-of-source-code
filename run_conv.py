import torch
import torch.optim as optim
import torch.nn as nn

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator

import os
import argparse
import random

import models
import utils

parser = argparse.ArgumentParser(description='Implemention of \'A Convolutional Attention Network for Extreme Summarization of Source Code\'', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--project', default='cassandra', type=str, help='Which project to run on')
parser.add_argument('--data_dir', default='data', type=str, help='Where to find the training data')
parser.add_argument('--checkpoints_dir', default='checkpoints', type=str, help='Where to save the model checkpoints')
parser.add_argument('--no_cuda', action='store_true', help='Use this flag to stop using the GPU')
parser.add_argument('--min_freq', default=2, help='Minimum times a token must appear in the dataset to not be unk\'d')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--emb_dim', default=128, type=int)
parser.add_argument('--k1', default=8, type=int)
parser.add_argument('--k2', default=8, type=int)
parser.add_argument('--w1', default=24, type=int)
parser.add_argument('--w2', default=29, type=int)
parser.add_argument('--w3', default=10, type=int)
parser.add_argument('--dropout', default=0.25, type=float)
parser.add_argument('--clip', default=1.0, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--seed', default=1234, type=int)

args = parser.parse_args()

assert os.path.exists(f'{args.data_dir}/{args.project}_train.json')
assert os.path.exists(f'{args.data_dir}/{args.project}_test.json')

if not os.path.exists(f'{args.checkpoints_dir}'):
    os.mkdir(f'{args.checkpoints_dir}')

#make deterministic
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

#get available device
device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')

#set up fields
BODY = Field()
NAME = Field()
fields = {'name': ('name', NAME), 'body': ('body', BODY)}

#get data from json
train, test = TabularDataset.splits(
                path = 'data',
                train = f'{args.project}_train.json',
                test = f'{args.project}_test.json',
                format = 'json',
                fields = fields
              )

#build the vocabulary
BODY.build_vocab(train.body, train.name, min_freq=args.min_freq)
NAME.build_vocab(train.body, train.name, min_freq=args.min_freq)

# make iterator for splits
train_iter, test_iter = BucketIterator.splits(
    (train, test), 
    batch_size=args.batch_size, 
    sort_key=lambda x: len(x.name),
    repeat=False,
    device=-1 if device == 'cpu' else None)

#calculate these for the model
vocab_size = len(BODY.vocab)
pad_idx = BODY.vocab.stoi['<pad>']
unk_idx = BODY.vocab.stoi['<unk>']

#initialize model
model = models.ConvAttentionNetwork(vocab_size, args.emb_dim, args.k1, args.k2, args.w1, args.w2, args.w3, args.dropout, pad_idx)

#place on GPU if available
model = model.to(device)

#initialize optimizer and loss function
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
optimizer = optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9)

criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion, clip):
    
    #turn on dropout/bn
    model.train()
    
    epoch_loss = 0
    n_examples = 0
    precision = 0
    recall = 0
    f1 = 0

    for i, batch in enumerate(iterator):
        
        bodies = batch.body
        names = batch.name
        
        optimizer.zero_grad()
        
        output = model(bodies, names)
        
        #take highest probability token as prediction
        preds = output.max(2)[1]

        examples = names.shape[1]
        n_examples += examples

        #calculate precision, recall and f1
        #this is probably very inefficient
        for ex in range(examples):
            actual = [n.item() for n in names[:,ex][1:]]
            predicted = [p.item() for p in preds[:,ex][1:]]
            _precision, _recall, _f1 = utils.token_precision_recall(predicted, actual, unk_idx, pad_idx)
            precision += _precision
            recall += _recall
            f1 += _f1

        #calculate loss
        loss = criterion(output[1:].view(-1, output.shape[2]), names[1:].view(-1))
        
        #calculate gradients wrt loss
        loss.backward()
        
        #clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        #update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator), precision/n_examples, recall/n_examples, f1/n_examples

def evaluate(model, iterator, criterion):
    
    #turn off bn/dropout
    model.eval()
    
    epoch_loss = 0
    n_examples = 0
    precision = 0
    recall = 0
    f1 = 0
    
    #ensures no gradients are calculated, speeds up calculations
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            bodies = batch.body.to(device)
            names = batch.name.to(device)

            output = model(bodies, names, 0) #set teacher forcing to zero

            preds = output.max(2)[1]

            examples = names.shape[1]
            n_examples += examples

            for ex in range(examples):
                actual = [n.item() for n in names[:,ex][1:]]
                predicted = [p.item() for p in preds[:,ex][1:]]
                _precision, _recall, _f1 = utils.token_precision_recall(predicted, actual, unk_idx, pad_idx)
                precision += _precision
                recall += _recall
                f1 += _f1

            loss = criterion(output[1:].view(-1, output.shape[2]), names[1:].view(-1))

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator), precision/n_examples, recall/n_examples, f1/n_examples

best_test_loss = float('inf')

if not os.path.isdir(f'{args.checkpoints_dir}'):
    os.makedirs(f'{args.checkpoints_dir}')
    
for epoch in range(args.epochs):
    
    train_loss, train_precision, train_recall, train_f1 = train(model, train_iter, optimizer, criterion, args.clip)
    test_loss, test_precision, test_recall, test_f1 = evaluate(model, test_iter, criterion)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), f'{args.checkpoints_dir}/{args.project}-conv-model.pt')    
    
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train F1: {train_f1:.3f} | Test Loss: {test_loss:.3f} | Test F1: {test_f1:.3f}')
