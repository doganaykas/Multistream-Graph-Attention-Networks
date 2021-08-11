import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class GraphAttentionLayer(nn.Module):   
    def __init__(self, in_features, out_features, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.B = nn.Parameter(torch.zeros(7, 7) + 1e-6)
        self.A = Variable(torch.eye(7), requires_grad=False)      
        
        
    def forward(self, h):     
        if len(h.size())==4:
            N, C, T, V = h.size()       
            h = h.permute(0, 3, 1, 2).contiguous().view(N, V, C*T)
        else:    
            N,V,C = h.size()
        
        #GAT
        Wh = torch.matmul(h, self.W)
        a_input = self.batch_prepare_attentional_mechanism_input(Wh)
        e = torch.matmul(a_input, self.a)    
        e = self.leakyrelu(e.squeeze(-1))
        attention = F.softmax(e, dim=-1)
        
        #Learnable Adjacency Matrix
        adj_mat = None
        self.A = self.A.cuda(h.get_device())
        adj_mat = self.B[:,:] + self.A[:,:]
        adj_mat_min = torch.min(adj_mat)
        adj_mat_max = torch.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(torch.diag(torch.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = torch.sqrt(torch.inverse(D))
        adj_mat_norm_d12 = torch.matmul(torch.matmul(D_12, adj_mat), D_12)
        
        #Updating the features of vertices
        attention = torch.matmul(adj_mat_norm_d12, attention)
        h_prime = torch.matmul(attention, Wh)  
            
        return F.elu(h_prime)
        
        
       
    def batch_prepare_attentional_mechanism_input(self, Wh):
        B, M, E = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, M, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(B, M, M, 2 * E)
    
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
      
    
class GAT_Multi_Head(nn.Module):
    def __init__(self, nfeat, nhid, alpha, nheads):
        super(GAT_Multi_Head, self).__init__()
        self.attentions = [GraphAttentionLayer(in_features=nfeat, out_features=nhid, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
      

    def forward(self, x):
        N, C, T, V = x.size()
        x = torch.cat([att(x) for att in self.attentions], dim=-1)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)         
        return x


class GraphAttentionLayer2D(nn.Module):   
    def __init__(self, in_features, out_features, alpha):
        super(GraphAttentionLayer2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.B = nn.Parameter(torch.zeros(7, 7) + 1e-6)
        self.A = Variable(torch.eye(7), requires_grad=False)
                   
        
    def forward(self, h):     
        if len(h.size())==4:     
            N, C, T, V = h.size()
            h = h.permute(0, 3, 1, 2)           
        else:   
            N,V,C = h.size()
                       
        #GAT
        Wh = torch.matmul(h, self.W)
        a_input = self.batch_prepare_attentional_mechanism_input(Wh)
        e = torch.matmul(a_input, self.a)
        e = self.leakyrelu(e.squeeze(-1))
        attention = F.softmax(e, dim=-1)
        
        #Learnable Adjacency Matrix
        adj_mat = None
        self.A = self.A.cuda(h.get_device())
        adj_mat = self.B[:,:] + self.A[:,:]
        adj_mat_min = torch.min(adj_mat)
        adj_mat_max = torch.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(torch.diag(torch.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = torch.sqrt(torch.inverse(D))
        adj_mat_norm_d12 = torch.matmul(torch.matmul(D_12, adj_mat), D_12)
        
        #2D Attention
        Wh = Wh.permute(0, 1, 3, 2) 
        attention = torch.diag_embed(attention)      
        Wh_=[]
        for i in range(V):
            at = torch.zeros(N, self.out_features, C).to(dev)
            for j in range(V):               
                at+=torch.matmul(Wh[:,j,:,:].to(dev), attention[:,i,j,:,:].to(dev))
            Wh_.append(at)
            
        h_prime = torch.stack((Wh_))         
        h_prime = h_prime.permute(1,3,2,0).contiguous().view(N, C * self.out_features, V)
        h_prime = torch.matmul(h_prime.double(), adj_mat_norm_d12).view(N, C, self.out_features, V)
    
        return F.elu(h_prime)

               
    def batch_prepare_attentional_mechanism_input(self, Wh):    
        B, M, E, T = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1) 
        Wh_repeated_alternating = Wh.repeat(1, M, 1, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(B, M, M, E, 2*T)
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
       
class GAT_Multi_Head2D(nn.Module):
    def __init__(self, nfeat, nhid, alpha, nheads):
        super(GAT_Multi_Head2D, self).__init__()    
        self.attentions = [GraphAttentionLayer2D(in_features=nfeat, out_features=nhid, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        

    def forward(self, x):
        N, C, T, V = x.size()
        x = torch.cat([att(x) for att in self.attentions], dim=2)           
        return x
     
    
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(         
            input_size=84,
            hidden_size=21,         
            num_layers=1,           
            batch_first=True)

        self.out = nn.Linear(21, 7)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(N, T, C*V)    
        r_out, (h_n, h_c) = self.rnn(x, None)

        out = self.out(r_out[:, 0, :])
        return out
    
class Model(nn.Module):
    def __init__(self):  
        super(Model, self).__init__()
        self.data_bn = nn.BatchNorm1d(6 * 7)
        self.l1 = GAT_Multi_Head(180, 9, 0.2, 20)
        self.l2 = GAT_Multi_Head2D(30,10, 0.2, 3)    
        self.l3 = RNN()

    def forward(self, x):    
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)
        #First Stream
        x_1 = self.l1(x)
        #Second Stream
        x_2 = self.l2(x)
        #Concetanation along weather variables
        x = torch.cat([x_1, x_2], dim=1)
        x = self.l3(x)
        
        return x