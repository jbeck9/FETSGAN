import torch
import torch.nn as nn
from utils import sample_uniform, get_mask

def get_device(x):
    device= x.get_device()
    if device < 0:
        device= 'cpu'
    return device



def input_padded(x, model, mask, pad_val, device=None):
    if mask is None:
        return model(x)
    
    if device is None:
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
        self.inpdim= inp_dim
        self.rsample = rsample
        
        self.Zcat= Z_dim + S_dim
        
        
        self.G_rnn = nn.GRU(
            input_size= self.Zcat, 
            hidden_size= nhidden, 
            num_layers=layers, 
            batch_first=True
        )
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden + eta_dim + self.Zcat, nhidden),
                                        nn.ReLU(),
                                        nn.Linear(nhidden, F_dim))
        if Z_dim > 0:
            self.emb_hidden=nn.Sequential(nn.Linear(Z_dim, nhidden),
                                          nn.ReLU(),
                                          nn.Linear(nhidden, nhidden*layers))
        
    def forward(self, z, T_in, mask=None, s= None):
        seqlen= max(T_in)
        
        if mask is None and min(T_in) < seqlen:
            mask= get_mask(T_in)
        
        device= get_device(z)
        noise= self.rsample([z.shape[0], seqlen, self.etadim]).to(device)
        
        if self.Z_dim > 0:
            init_hidden= self.emb_hidden(z).reshape(z.shape[0],self.layers, -1).permute((1,0,2))
            
            zi= z.unsqueeze(1).tile([1, seqlen, 1])
        else:
            zi= self.rsample([z.shape[0], seqlen, self.inpdim]).to(device)
        
        if self.S_dim > 0:
            zi= torch.cat([zi, s[:,:seqlen]], dim=-1)
        
        z_pack = nn.utils.rnn.pack_padded_sequence(
            input=zi, 
            lengths=T_in, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        if self.Z_dim > 0:
            out, _ = self.G_rnn(z_pack, init_hidden)
        else:
            out, _ = self.G_rnn(z_pack)
            
        
        out, T_out = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out, 
            batch_first=True,
            padding_value=self.padval,
            total_length=seqlen
        )
        out= torch.cat([out, noise, zi], dim=-1)
        
        return input_padded(out, self.out_linear, mask, self.padval, device=device)
    
class TEncoder(nn.Module):
    def __init__(self, Z_dim, S_dim, F_dim, 
                 eta_dim, inp_dim, rsample, nhidden, layers, pad_val, nhead=2):
        super(TEncoder, self).__init__()
        
        self.layers= layers
        self.nhidden= nhidden
        self.S_dim= S_dim
        self.etadim= eta_dim
        self.padval= pad_val
        self.rsample= rsample
        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, batch_first= True)
        
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.in_linear= nn.Linear(F_dim + S_dim, nhidden)
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden + eta_dim, nhidden),
                                        nn.ReLU(),
                                        nn.Linear(nhidden, Z_dim))
        
        
    def forward(self, x, T_in, s=None):
        device= get_device(x)
        mask= get_mask(T_in)
        
        noise= self.rsample([x.shape[0], self.etadim]).to(device)
        
        if self.S_dim > 0:
            x= torch.cat([x, s], dim=-1)
        
        x= input_padded(x, self.in_linear, mask, self.padval)
        enc_out= self.model(x, src_key_padding_mask= ~mask.to(device))
        
        out= torch.cat([enc_out[:,-1,:], noise], dim=-1)
        
        return self.out_linear(out)
    
class Encoder(nn.Module):
    def __init__(self, Z_dim, S_dim, F_dim, 
                 eta_dim, inp_dim, rsample, nhidden, layers, pad_val):
        super(Encoder, self).__init__()
        
        self.S_dim= S_dim
        self.etadim= eta_dim
        self.Zcat= F_dim + S_dim
        self.padval= pad_val
        self.rsample= rsample

        
        self.E_rnn = nn.GRU(
            input_size= self.Zcat, 
            hidden_size=nhidden,
            num_layers=layers, 
            batch_first=True,
        )
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden + eta_dim, nhidden),
                                        nn.LeakyReLU(0.1),
                                        nn.Linear(nhidden, Z_dim))
        
        
    def forward(self, x,T_in, s= None):
        device= get_device(x)
        
        noise= self.rsample([x.shape[0], self.etadim]).to(device)
        # x= torch.cat([x,noise], dim=-1)
        
        if self.S_dim > 0:
            x= torch.cat([x, s], dim=-1)
            
        # x= self.in_linear(x)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=T_in, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        _, out = self.E_rnn(x_pack)
        
        out= torch.cat([out[-1], noise], dim=-1)
        
        return self.out_linear(out)
    
class Discriminator(nn.Module):
    def __init__(self, F_dim,S_dim, 
                 inp_dim=50, nhidden=32, layers=2, pad_val= -2):
        super(Discriminator, self).__init__()
        
        self.S_dim= S_dim
        self.Zcat= F_dim + S_dim
        self.padval= pad_val
        
        self.D_rnn = nn.GRU(
            input_size= self.Zcat,
            hidden_size=nhidden, 
            num_layers=layers, 
            batch_first=True
        )
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden + self.Zcat, nhidden),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(nhidden, 1))
        
    def forward(self, x,T_in, mask= None, s= None):
        seqlen= max(T_in)
        
        xi = x + 0.02 * torch.randn_like(x)
        
        if self.S_dim != 0:
            xi= torch.cat([xi, s], dim=-1)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(
            input=xi, 
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
        
        out= torch.cat([out, xi], dim=-1)
        return input_padded(out, self.out_linear, mask, self.padval)
    
class linearDis(nn.Module):
    def __init__(self, Z_dim,nhidden=500, layers=3):
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
        self.Z_dim= Z_dim
        self.S_dim= S_dim
        self.F_dim= F_dim
        self.padval= pad_val
        self.sampler= rsample
        
        self.G= Generator(Z_dim, S_dim, F_dim, eta_dim, inp_dim, rsample, nhidden, layers, pad_val)
        self.D= Discriminator(F_dim, S_dim, inp_dim, nhidden, layers, pad_val)
        
        if self.fets:
            self.E= Encoder(Z_dim, S_dim, F_dim, eta_dim, inp_dim, rsample, nhidden, layers, pad_val)
            self.LD= linearDis(Z_dim)
            
    def inf(self, T, device, mask=None, s=None):
        
        # with torch.no_grad():
        zr= self.sampler([len(T), self.Z_dim]).to(device)
        return self.G(zr, T, mask=mask,s=s)