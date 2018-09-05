import torch
import torch.optim as optim
import torch.nn as nn

from torchtext.data import Field, LabelField
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator

import os
import argparse

import models

parser = argparse.ArgumentParser(description='Implemention of \'A Convolutional Attention Network for Extreme Summarization of Source Code\'', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--project', default='cassandra', type=str, help='Which project to run on')
parser.add_argument('--data_dir', default='data', type=str, help='Where to find the training data')
parser.add_argument('--checkpoints_dir', default='checkpoints', type=str, help='Where to save the model checkpoints')
parser.add_argument('--no_cuda', action='store_true', help='Use this flag to stop using the GPU')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--emb_dim', default=128, type=int)
parser.add_argument('--k1', default=8, type=int)
parser.add_argument('--k2', default=8, type=int)
parser.add_argument('--w1', default=24, type=int)
parser.add_argument('--w2', default=29, type=int)
parser.add_argument('--w3', default=10, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--clip', default=1.0, type=float)
parser.add_argument('--epochs', default=100, type=int)

args = parser.parse_args()

assert os.path.exists(f'{args.data_dir}/{args.project}_train.json')
assert os.path.exists(f'{args.data_dir}/{args.project}_test.json')
                    
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
BODY.build_vocab(train.body, train.name)
NAME.build_vocab(train.body, train.name)

# make iterator for splits
train_iter, test_iter = BucketIterator.splits(
    (train, test), 
    batch_size=args.batch_size, 
    sort_key=lambda x: len(x.name),
    repeat=False,
    device=-1)

#calculate these for the model
vocab_size = len(BODY.vocab)
pad_idx = BODY.vocab.stoi['<pad>']

#initialize model
model = models.CopyAttentionNetwork(vocab_size, args.emb_dim, args.k1, args.k2, args.w1, args.w2, args.w3, args.dropout, pad_idx)

#place on GPU if available
model = model.to(device)

#initialize optimizer, scheduler and loss function
criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters())

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        bodies = batch.body.to(device)
        names = batch.name.to(device)

        copy = torch.zeros(names.shape[0], names.shape[1], bodies.shape[0])
       
        _ones = torch.ones(bodies.shape[0])
        _zeros = torch.zeros(bodies.shape[0])

        for j, (name, body) in enumerate(zip(names, bodies)):
            for k, token in enumerate(name):
                copy[j,k,:] = torch.where(bodies[:,k] == token, _ones, _zeros)
        
        optimizer.zero_grad()
        
        output, kappas = model(bodies, names)
       
        output = output[1:].view(-1, output.shape[2])
        kappas = kappas[1:].view(-1, kappas.shape[2])

        names = names[1:].view(-1,1)
        one_hot_names = torch.zeros(names.shape[0], output.shape[1])
        one_hot_names.scatter_(1, names, 1)
        copy = copy[1:].view(-1, copy.shape[2])
 
        pred = torch.cat((output, kappas), dim=1)
        target = torch.cat((one_hot_names, copy), dim=1)

        loss = criterion(pred, target)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
     
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            bodies = batch.body.to(device)
            names = batch.name.to(device)
            
            
            copy = torch.zeros(names.shape[0], names.shape[1], bodies.shape[0])
       
            _ones = torch.ones(bodies.shape[0])
            _zeros = torch.zeros(bodies.shape[0])

            for j, (name, body) in enumerate(zip(names, bodies)):
                for k, token in enumerate(name):
                    copy[j,k,:] = torch.where(bodies[:,k] == token, _ones, _zeros)

            output, kappas = model(bodies, names, 0) #set teacher forcing to zero

            output = output[1:].view(-1, output.shape[2])
            kappas = kappas[1:].view(-1, kappas.shape[2])

            one_hot_names = torch.zeros(names.shape[0], output.shape[1])
            one_hot_names.scatter_(1, names, 1)
            copy = copy[1:].view(-1, copy.shape[2])

            pred = torch.cat((output, kappas), dim=1)
            target = torch.cat((one_hot_names, copy), dim=1)

            loss = criterion(pred, target)

            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

best_test_loss = float('inf')

if not os.path.isdir(f'{args.checkpoints_dir}'):
    os.makedirs(f'{args.checkpoints_dir}')
    
for epoch in range(args.epochs):
    
    train_loss = train(model, train_iter, optimizer, criterion, args.clip)
    test_loss = evaluate(model, test_iter, criterion)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), f'{args.checkpoints_dir}/{args.project}-copy-model.pt')    
    
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} |')
