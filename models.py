import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class AttentionFeatures(nn.Module):
    """
    Page 3 of the paper
    attention_features (code tokens c, context h_{t-1})
     C <- lookupandpad(c, E)
     L1 <- ReLU(Conv1d(C, K_{l1}))
     L2 <- Conv1d(L1, K_{l2}) * h_{t-1}
     Lfeat <- L2/||L2||_2
     return Lfeat
    """
    def __init__(self, embedding_dim, k1, w1, k2, w2, w3, dropout):
        super().__init__()
                
        self.w1 = w1
        self.k1 = k1

        self.w2 = w2
        self.k2 = k2

        #self.w3 = w3 #use this to calculate padding

        self.conv1 = nn.Conv1d(embedding_dim, k1, w1)
        self.conv2 = nn.Conv1d(k1, k2, w2)
        self.do = nn.Dropout(dropout)
        self.activation = nn.PReLU()
    def forward(self, C, h_t):
        
        #C = embedded body tokens
        #h_t = previous hidden state used to predict name token
        
        #C = [bodies len, batch size, emb dim]
        #h_t = [1, batch size, k2]
        
        C = C.permute(1, 2, 0) #input to conv needs n_channels as dim 1
        
        #C = [batch size, emb dim, bodies len]
        
        h_t = h_t.permute(1, 2, 0) #from [1, batch size, k2] to [batch size, k2, 1]
        
        #h_t = [batch size, k2, 1]
        
        L_1 = self.do(self.activation(self.conv1(C)))
        
        #L_1 = [batch size, k1, bodies len - w1 + 1]
        
        L_2 = self.do(self.conv2(L_1)) * h_t
                
        #L_2 = [batch size, k2, bodies len - w1 - w2 + 2]
        
        L_feat = F.normalize(L_2, p=2, dim=1)
                
        #L_feat = [batch size, k2, bodies len - w1 - w2 + 2]
                
        return L_feat
    
class AttentionWeights(nn.Module):
    """
    Page 3 of the paper
    attention_features (attention features Lfeat, kernel K)
     return Softmax(Conv1d(Lfeat, K))
    """
    def __init__(self, k2, w3, dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(k2, 1, w3)
        self.do = nn.Dropout(dropout)

    def forward(self, L_feat):
                
        #L_feat = [batch size, k2, bodies len - w1 - w2 + 2]
        
        x = self.do(self.conv1(L_feat))
        
        #x = [batch size, 1, bodies len - w1 - w2 - w3 + 3]
        
        x = x.squeeze(1)
        
        #x = [batch size, bodies len - w1 - w2 - w3 + 3]
        
        x = F.softmax(x, dim=1)
                
        #x = [batch size, bodies len - w1 - w2 - w3 + 3]
                
        return x
    
class ConvAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, k1, k2, w1, w2, w3, dropout, pad_idx):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.k1 = k1
        self.k2 = k2
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.dropout = dropout
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.do = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, k2)
        self.attn_feat = AttentionFeatures(embedding_dim, k1, w1, k2, w2, w3, dropout)
        self.attn_weights = AttentionWeights(k2, w3, dropout)
        self.bias = nn.Parameter(torch.ones(vocab_size))
        
        n_padding = w1 + w2 + w3 - 3
        self.padding = torch.zeros(n_padding, 1).fill_(pad_idx).long()
        
    def forward(self, bodies, names, tf=None):
        
        if tf is None:
            tf = self.dropout
        
        #bodies = [bodies len, batch size]
        #names = [names len, batch size]  
        
        #stores the probabilities generated for each token
        outputs = torch.zeros(names.shape[0], names.shape[1], self.vocab_size).to(names.device)
        
        #outputs = [name len, batch size, vocab dim]
        
        #need to pad the function body so after it has been fed through
        #the convolutional layers it is the same size as the original function body
        bodies_padded = torch.cat((bodies, self.padding.expand(-1, bodies.shape[1]).to(bodies.device)))
        
        #bodies_padded = [bodies len + w1 + w2 + w3 - 3, batch_size]
        
        #from now on when we refer to bodies len, we mean the padded version
        
        #convert function body tokens into their embeddings
        emb_b = self.embedding(bodies_padded)
        
        #emb_b = [bodies len, batch size, emb dim]
                
        #first input to the gru is the first token of the function name
        #which is a start of sentence token
        output = names[0]
            
        #generate predicted function name tokens one at a time
        for i in range(1, names.shape[0]):
                        
            #initial hidden state is start of sentence token passed through gru
            #subsequent hidden states from either the previous token predicted by the model
            #or the ground truth token the model should have predicted
            _, h_t = self.gru(self.embedding(output).unsqueeze(0))

            #h_t = [1, batch size, k2]

            #computes `k2` features for each token which are scaled by h_t
            L_feat = self.attn_feat(emb_b, h_t)

            #L_feat = [batch size, k2, bodies len - w1 - w2 + 2]

            #computes the attention values for each token in the function body
            #the second dimension is now equal to the original unpadded `bodies len` size
            alpha = self.attn_weights(L_feat)

            #alpha = [batch size, bodies len - w1 - w2 - w3 + 3]

            #emb_b also contains the padding tokens so we slice these off
            emb_b_slice = emb_b.permute(1, 0, 2)[:, :bodies.shape[0], :]

            #emb_b = [batch_size, bodies len, emb dim]

            #apply the attention to the embedded function body tokens
            n_hat = torch.sum(alpha.unsqueeze(2) * emb_b_slice, dim=1)

            #n_hat = [batch size, emb dim]

            #E is the embedding layer weights
            E = self.embedding.weight.unsqueeze(0).expand(bodies.shape[1],-1,-1)

            #E = [batch size, vocab size, emb dim]

            #matrix multiply E and n_hat and apply a bias
            #n is the probability distribution over the vocabulary for the predicted next token
            n = torch.bmm(E, n_hat.unsqueeze(2)).squeeze(2) + self.bias.unsqueeze(0).expand(bodies.shape[1], -1)
            
            #n = [batch size, vocab size]
            
            #store prediction probability distribution in large tensor that holds 
            #predictions for each token in the function name
            outputs[i] = n
            
            #with probability of `tf`, use the model's prediction of the next token
            #as the next token to feed into the model (to become the next h_t)
            #with probability 1-`tf`, use the actual ground truth next token as
            #the next token to feed into the model
            #teacher forcing ratio is equal to dropout during training and 0 during inference
            if random.random() < tf:
                
                #model's predicted token highest value in the probability distribution
                top1 = n.max(1)[1]
                output = top1
                
            else:
                output = names[i]
                
        #outputs = [name len, batch size, vocab dim]
                
        return outputs
    
class CopyAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, k1, k2, w1, w2, w3, dropout, pad_idx):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.k1 = k1
        self.k2 = k2
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.dropout = dropout
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.do = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, k2)
        self.attn_feat = AttentionFeatures(embedding_dim, k1, w1, k2, w2, w3, dropout)
        self.attn_weights_alpha = AttentionWeights(k2, w3, dropout)
        self.attn_weights_kappa = AttentionWeights(k2, w3, dropout)
        self.conv1 = nn.Conv1d(k2, 1, w3)
        self.bias = nn.Parameter(torch.ones(vocab_size))
        
        n_padding = w1 + w2 + w3 - 3
        self.padding = torch.zeros(n_padding, 1).fill_(pad_idx).long()
        
    def forward(self, bodies, names, tf=None):
        
        if tf is None:
            tf = self.dropout
            
        #bodies = [bodies len, batch size]
        #names = [names len, batch size]  
        
        #stores the probabilities generated for each token
        outputs = torch.zeros(names.shape[0], names.shape[1], self.vocab_size).to(names.device)
        
        #outputs = [names len, batch size, vocab dim]
        
        #stores the copy attention generated for each token
        kappas = torch.zeros(names.shape[0], names.shape[1], bodies.shape[0])

        #kappas = [name len, batch size, bodies len]

        #need to pad the function body so after it has been fed through
        #the convolutional layers it is the same size as the original function body
        bodies_padded = torch.cat((bodies, self.padding.expand(-1, bodies.shape[1]).to(bodies.device)))
        
        #bodies_padded = [bodies len + w1 + w2 + w3 - 3, batch_size]
        
        #from now on when we refer to bodies len, we mean the padded version
        
        #convert function body tokens into their embeddings
        emb_b = self.embedding(bodies_padded)
        
        #emb_b = [bodies len, batch size, emb dim]
                
        #first input to the gru is the first token of the function name
        #which is a start of sentence token
        output = names[0]
            
        #generate predicted function name tokens one at a time
        for i in range(1, names.shape[0]):
                        
            #initial hidden state is start of sentence token passed through gru
            #subsequent hidden states from either the previous token predicted by the model
            #or the ground truth token the model should have predicted
            _, h_t = self.gru(self.embedding(output).unsqueeze(0))

            #h_t = [1, batch size, k2]

            #computes `k2` features for each token which are scaled by h_t
            L_feat = self.attn_feat(emb_b, h_t)

            #L_feat = [batch size, k2, bodies len - w1 - w2 + 2]

            #alpha is the attention values for each token in the function body
            #kappa is the probability that each token in the function body is copied
            #the second dimension is now equal to the original unpadded `bodies len` size
            alpha = self.attn_weights_alpha(L_feat)
            kappa = self.attn_weights_kappa(L_feat)          
        
            #alpha = [batch size, bodies len - w1 - w2 - w3 + 3]
            #kappa = [batch size, bodies len - w1 - w2 - w3 + 3]

            #calculate the weight of predicting by copying from body vs. predicting by guessing from vocab
            lambd = F.max_pool1d(torch.sigmoid(self.do(self.conv1(L_feat))), alpha.shape[1]).squeeze(2)
            
            #emb_b also contains the padding tokens so we slice these off
            emb_b_slice = emb_b.permute(1, 0, 2)[:, :bodies.shape[0], :]

            #emb_b = [batch_size, bodies len, emb dim]

            #apply the attention to the embedded function body tokens
            n_hat = torch.sum(alpha.unsqueeze(2) * emb_b_slice, dim=1)

            #n_hat = [batch size, emb dim]

            #E is the embedding layer weights
            E = self.embedding.weight.unsqueeze(0).expand(bodies.shape[1],-1,-1)

            #E = [batch size, vocab size, emb dim]

            #matrix multiply E and n_hat and apply a bias
            #n is the probability distribution over the vocabulary for the predicted next token
            n = torch.bmm(E, n_hat.unsqueeze(2)).squeeze(2) + self.bias.unsqueeze(0).expand(bodies.shape[1], -1)
            
            #n = [batch size, vocab size]
            
            #store prediction probability distribution in large tensor that holds 
            #predictions for each token in the function name   
            outputs[i] = (1 - lambd) * F.softmax(n,dim=1)
            
            #store copy probability distribution
            kappas[i] = lambd * kappa

            #with probability of `tf`, use the model's prediction of the next token
            #as the next token to feed into the model (to become the next h_t)
            #with probability 1-`tf`, use the actual ground truth next token as
            #the next token to feed into the model
            #teacher forcing ratio is equal to dropout during training and 0 during inference
            if random.random() < tf:
                
                #model's predicted token highest value in the probability distribution
                top1 = n.max(1)[1]
                output = top1
                
            else:
                output = names[i]
                
        #outputs = [name len, batch size, vocab dim]
                
        return outputs, kappas
