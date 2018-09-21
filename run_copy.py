import torch
import torch.optim as optim
import torch.nn as nn

from torchtext.data import Field, LabelField
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
parser.add_argument('--k1', default=32, type=int)
parser.add_argument('--k2', default=16, type=int)
parser.add_argument('--w1', default=18, type=int)
parser.add_argument('--w2', default=19, type=int)
parser.add_argument('--w3', default=2, type=int)
parser.add_argument('--dropout', default=0.4, type=float)
parser.add_argument('--clip', default=0.75, type=float)
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
model = models.CopyAttentionNetwork(vocab_size, args.emb_dim, args.k1, args.k2, args.w1, args.w2, args.w3, args.dropout, pad_idx)

#place on GPU if available
model = model.to(device)

#initialize optimizer
optimizer = optim.RMSprop(model.parameters(), momentum=0.9, lr=1e-3)

def train(model, iterator, optimizer, clip):
    
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

        I = torch.zeros(names.shape[0], names.shape[1], bodies.shape[0]).to(device)
       
        _ones = torch.ones(bodies.shape[0]).to(device)
        _zeros = torch.zeros(bodies.shape[0]).to(device)

        #create the I tensor
        #the length of the method body where elements are:
        #  1 in the position where the current token you are trying to predict are in the body
        #  0 otherwise
        for j, name in enumerate(names):
            for k, token in enumerate(name):
                I[j,k,:] = torch.where(bodies[:,k] == token, _ones, _zeros)
        
        #output is predictions
        #kappas are copy-attention over the body
        #lambdas are probability of copy over generate from vocab
        output, kappas, lambdas = model(bodies, names)
       
        examples = names.shape[1]
        n_examples += examples
        
        copy_preds = kappas.max(2)[1]
        vocab_preds = output.max(2)[1]
                
        for ex in range(examples):
            predicted = []
            actual = [n.item() for n in names[:,ex][1:]]
            for n, l in enumerate(lambdas[:,ex][1:], start=1):
                if l.item() >= 0.5: #do copy
                    copied_token_position = copy_preds[n,ex]
                    predicted.append(bodies[copied_token_position, ex].item())
                else:
                    predicted.append(vocab_preds[n,ex].item())
            _precision, _recall, _f1 = utils.token_precision_recall(predicted, actual, unk_idx)
            precision += _precision
            recall += _recall
            f1 += _f1
    
        #reshape parameters
        output = output[1:].view(-1, output.shape[2])
        kappas = kappas[1:].view(-1, kappas.shape[2])
        lambdas = lambdas[1:].view(-1)
        I = I[1:].view(-1, I.shape[2])
        names = names[1:].view(-1, 1)

        #probability of using copy and model predictions from vocab
        use_copy = torch.log(lambdas + 10e-8) + torch.sum(I * kappas, dim=1)
        use_model = torch.log(1 - lambdas + 10e-8) + torch.gather(output, 1, names).squeeze(1)

        #calculate loss
        loss = torch.mean(utils.logsumexp(use_copy, use_model))

        #calculate gradients
        loss.backward()
        
        #clip to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        #update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
     
    return epoch_loss / len(iterator), precision/n_examples, recall/n_examples, f1/n_examples

def evaluate(model, iterator):
    
    #turn off bn/dropout
    model.eval()
    
    epoch_loss = 0
    n_examples = 0
    precision = 0
    recall = 0
    f1 = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            bodies = batch.body
            names = batch.name

            I = torch.zeros(names.shape[0], names.shape[1], bodies.shape[0]).to(device)
       
            _ones = torch.ones(bodies.shape[0]).to(device)
            _zeros = torch.zeros(bodies.shape[0]).to(device)

            for j, name in enumerate(names):
                for k, token in enumerate(name):
                    I[j,k,:] = torch.where(bodies[:,k] == token, _ones, _zeros)

            output, kappas, lambdas = model(bodies, names, 0) #set teacher forcing to zero

            examples = names.shape[1]
            n_examples += examples

            copy_preds = kappas.max(2)[1]
            vocab_preds = output.max(2)[1]

            for ex in range(examples):
                predicted = []
                actual = [n.item() for n in names[:,ex][1:]]
                for n, l in enumerate(lambdas[:,ex][1:], start=1):
                    if l.item() >= 0.5: #do copy
                        copied_token_position = copy_preds[n,ex]
                        predicted.append(bodies[copied_token_position, ex].item())
                    else:
                        predicted.append(vocab_preds[n,ex].item())
                _precision, _recall, _f1 = utils.token_precision_recall(predicted, actual, unk_idx)
                precision += _precision
                recall += _recall
                f1 += _f1
            
            output = output[1:].view(-1, output.shape[2])
            kappas = kappas[1:].view(-1, kappas.shape[2])
            lambdas = lambdas[1:].view(-1)
            I = I[1:].view(-1, I.shape[2])
            names = names[1:].view(-1,1)

            use_copy = torch.log(lambdas + 10e-8) + torch.sum(I * kappas, dim=1)
            use_model = torch.log(1 - lambdas + 10e-8) + torch.gather(output, 1, names).squeeze(1)

            loss = torch.mean(utils.logsumexp(use_copy, use_model))

            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator), precision/n_examples, recall/n_examples, f1/n_examples

best_test_loss = float('inf')

if not os.path.isdir(f'{args.checkpoints_dir}'):
    os.makedirs(f'{args.checkpoints_dir}')
    
for epoch in range(args.epochs):
    
    train_loss, train_precision, train_recall, train_f1 = train(model, train_iter, optimizer, args.clip)
    test_loss, test_precision, test_recall, test_f1 = evaluate(model, test_iter)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), f'{args.checkpoints_dir}/{args.project}-copy-model.pt')    
    
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train F1: {train_f1:.3f} | Test Loss: {test_loss:.3f} | Test F1: {test_f1:.3f}')