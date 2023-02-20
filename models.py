import torch
import torch.nn as nn
from utils import sample_uniform, get_mask

def get_device(x):
    device= x.get_device()
    if device == -1:
        device= 'cpu'
    return device

def input_padded(x, model, mask, pad_val):
    if mask is None:
        return model(x)
    
    device= x.get_device()
    sq_x= x[mask]
    sq_out= model(sq_x)
    
    base_shape= list(x.shape)
    base_shape[-1] = sq_out.shape[-1]
    base= pad_val * torch.ones(base_shape, device= device)
    
    base[mask] = sq_out
    return base

class Generator(nn.Module):
    def __init__(self, Z_dim, S_dim, F_dim, 
                 eta_dim, inp_dim, rsample, nhidden, layers, pad_val):
        super(Generator, self).__init__()
        
        self.Z_dim= Z_dim
        self.S_dim= S_dim
        self.etadim= eta_dim
        self.padval= pad_val
        self.layers= layers
        self.rsample = rsample
        
        self.Zcat= Z_dim + S_dim + eta_dim
        
        
        self.G_rnn = nn.GRU(
            input_size= inp_dim, 
            hidden_size= nhidden, 
            num_layers=layers, 
            batch_first=True
        )
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden, nhidden),
                                        nn.LeakyReLU(0.1),
                                        nn.Linear(nhidden, F_dim))
        if Z_dim != 0:
            self.in_linear= nn.Sequential(nn.Linear(self.Zcat, nhidden),
                                          nn.LeakyReLU(0.1),
                                          nn.Linear(nhidden, inp_dim))
        
            self.emb_hidden=nn.Sequential(nn.Linear(Z_dim, nhidden),
                                          nn.LeakyReLU(0.1),
                                          nn.Linear(nhidden, nhidden*layers))
        
    def forward(self, z, T_in, mask=None, s= None):
        seqlen= max(T_in)
        
        if mask is None and min(T_in) < seqlen:
            mask= get_mask(T_in)
        
        device= get_device(z)
        
        noise= self.rsample([z.shape[0], seqlen, self.etadim]).to(device)
        
        if self.Z_dim > 0:
            init_hidden= self.emb_hidden(z).reshape(self.layers, z.shape[0],-1)
            z= z.unsqueeze(1).tile([1, seqlen, 1])
            z= torch.cat([z,noise], dim=-1)
            z= self.in_linear(z)
        else:
            z= noise
        
        if self.S_dim > 0:
            z= torch.cat([z, s], dim=-1)
            
        z_pack = nn.utils.rnn.pack_padded_sequence(
            input=z, 
            lengths=T_in, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        out, _ = self.G_rnn(z_pack, init_hidden)
        
        out, T_out = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out, 
            batch_first=True,
            padding_value=self.padval,
            total_length=seqlen
        )
        
        return input_padded(out, self.out_linear, mask, self.padval)
    
class Encoder(nn.Module):
    def __init__(self, Z_dim, S_dim, F_dim, 
                 eta_dim, inp_dim, rsample, nhidden, layers, pad_val):
        super(Encoder, self).__init__()
        
        self.S_dim= S_dim
        self.etadim= eta_dim
        self.Zcat= F_dim + S_dim + eta_dim
        self.padval= pad_val
        self.rsample= rsample

        
        self.E_rnn = nn.GRU(
            input_size= inp_dim, 
            hidden_size=nhidden,
            num_layers=layers, 
            batch_first=True,
        )
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden, nhidden),
                                        nn.LeakyReLU(0.1),
                                        nn.Linear(nhidden, Z_dim))
        
        self.in_linear= nn.Sequential(nn.Linear(self.Zcat, nhidden),
                                      nn.LeakyReLU(0.1),
                                      nn.Linear(nhidden, inp_dim))
        
        
    def forward(self, x,T_in, s= None):
        device= get_device(x)
        seqlen= max(T_in)
        
        noise= self.rsample([x.shape[0], seqlen, self.etadim]).to(device)
        x= torch.cat([x,noise], dim=-1)
        
        if self.S_dim > 0:
            x= torch.cat([x, s], dim=-1)
            
        x= self.in_linear(x)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=T_in, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        _, out = self.E_rnn(x_pack)
        
        return self.out_linear(out[-1])
    
class Discriminator(nn.Module):
    def __init__(self, F_dim,S_dim, 
                 inp_dim= 50,nhidden=32, layers=2, pad_val= -2):
        super(Discriminator, self).__init__()
        
        self.S_dim= S_dim
        self.Zcat= F_dim + S_dim
        self.padval= pad_val
        
        self.D_rnn = nn.GRU(
            input_size= inp_dim,
            hidden_size=nhidden, 
            num_layers=layers, 
            batch_first=True
        )
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden, nhidden),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(nhidden, 1))
        
        self.in_linear= nn.Sequential(nn.Linear(self.Zcat, nhidden),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(nhidden, inp_dim))
        
    def forward(self, x,T_in, mask= None, s= None):
        seqlen= max(T_in)
        
        if self.S_dim != 0:
            x= torch.cat([x, s], dim=-1)
        
        x= self.in_linear(x)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=T_in, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        out, _= self.D_rnn(x_pack)
        
        out, T_out = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out, 
            batch_first=True,
            padding_value=self.padval,
            total_length=seqlen
        )
        
        return input_padded(out, self.out_linear, mask, self.padval)
    
class linearDis(nn.Module):
    def __init__(self, Z_dim,nhidden=50, layers=3):
        super(linearDis, self).__init__()
        
        net= [nn.Linear(Z_dim, nhidden),nn.Dropout(0.1),nn.LeakyReLU(0.2)]
        for _ in range(layers):
            net.extend([nn.Linear(nhidden, nhidden),nn.Dropout(0.1),nn.LeakyReLU(0.2)])
        net.extend([nn.Linear(nhidden, 1)])
        
        self.net= nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)
    
class fetsGan(nn.Module):
    def __init__(self, Z_dim, S_dim, F_dim,
                 eta_dim=4, inp_dim=50, rsample= sample_uniform, nhidden=100,layers=1, pad_val= -2):
        super(fetsGan, self).__init__()
        
        self.fets= Z_dim > 0
        self.F_dim= F_dim
        self.padval= pad_val
        self.sampler= rsample
        
        self.G= Generator(Z_dim, S_dim, F_dim, eta_dim, inp_dim, rsample, nhidden, layers, pad_val)
        self.D= Discriminator(F_dim, S_dim)
        
        if self.fets:
            self.E= Encoder(Z_dim, S_dim, F_dim, eta_dim, inp_dim, rsample, nhidden, layers, pad_val)
            self.LD= linearDis(Z_dim)
            
    # def forward(self,net,inp,T=None, mask= None, s= None):
    #     if net == 'G':
    #         return self.G(inp,T, mask, s)
        
    #     elif net == 'E':
    #         return self.E(inp,T, s)
        
    #     elif net == 'D':
    #         return self.D(inp,T, s)
        
    #     elif net == 'LD':
    #         return self.LD(inp)
        
        
        
        
    
if __name__ == '__main__':
    # gen= Generator(10, 0, 4, eta_dim=4, inp_dim=50, rsample= sample_uniform, nhidden=100,layers=1, pad_val= -2)
    # inp= torch.ones([64,10])
    # gen(inp, [10]*64)
    
    # enc= Encoder(10,0,4)
    # inp= torch.ones([64, 10, 4])
    # enc(inp, [10]*64)
    
    # dis= Discriminator(4,0)
    # inp= torch.ones([64, 10, 4])
    # dis(inp, [10]*64)
    
    # dis= linearDis(10)
    # inp= torch.ones([64,10])
    # dis(inp)
    
    model= fetsGan(10, 0, 4)