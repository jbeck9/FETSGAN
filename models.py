import torch
import torch.nn as nn
from utils import sample_uniform, get_mask

from torch.nn.utils import spectral_norm

def get_device(x):
    device= x.get_device()
    if device < 0:
        device= 'cpu'
    return device

def square_subsequent_mask(sz, device= 'cuda'):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

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
        
        self.hidden= None
        
        
        self.G_rnn = nn.GRU(
            input_size= self.inpdim, 
            hidden_size= nhidden, 
            num_layers=layers, 
            batch_first=True
        )
        
        self.in_linear= nn.Sequential(nn.Linear(eta_dim + self.Zcat, nhidden),
                                        nn.GELU(),
                                        nn.Linear(nhidden, self.inpdim))
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden + self.Zcat + eta_dim, nhidden),
                                        nn.GELU(),
                                        nn.Linear(nhidden, F_dim))
        
        # self.out_bool= nn.Sequential(nn.Linear(nhidden + self.Zcat + eta_dim, nhidden),
        #                                 nn.GELU(),
        #                                 nn.Linear(nhidden, F_dim))
        if Z_dim > 0:
            self.emb_hidden=nn.Sequential(nn.Linear(Z_dim, nhidden),
                                          nn.GELU(),
                                          nn.Linear(nhidden, nhidden*layers))
        
    def forward(self, z, T_in, mask=None, s= None, reset_hidden= True):
        seqlen= max(T_in)
        
        if mask is None and min(T_in) < seqlen:
            mask= get_mask(T_in)
        
        device= get_device(z)
        noise= self.rsample([z.shape[0], seqlen, self.etadim]).to(device)
        
        if self.Z_dim > 0:
            if reset_hidden or self.hidden == None:
                self.hidden= self.emb_hidden(z).reshape(z.shape[0],self.layers, -1).permute((1,0,2))
            zi= z.unsqueeze(1).tile([1, seqlen, 1])
        else:
            zi= self.rsample([z.shape[0], seqlen, self.inpdim]).to(device)
        
        if self.S_dim > 0:
            si= s[:,:seqlen]
            # s_noise= 0.05 * si * torch.randn_like(si)
            # si= si + s_noise
            
            zi= torch.cat([zi, si], dim=-1)
            
        zi= torch.cat([zi, noise], dim=-1)
        
        zi_emb= input_padded(zi, self.in_linear, mask, self.padval, device=device)
        
        z_pack = nn.utils.rnn.pack_padded_sequence(
            input=zi_emb, 
            lengths=T_in, 
            batch_first=True,
            enforce_sorted=False
        )
        
        if self.Z_dim > 0:
            out, self.hidden = self.G_rnn(z_pack, self.hidden)
        else:
            out, self.hidden = self.G_rnn(z_pack)
            
        
        out, T_out = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out, 
            batch_first=True,
            padding_value=self.padval,
            total_length=seqlen
        )
        
        out= torch.cat([out, zi], dim=-1)
        fout= input_padded(out, self.out_linear, mask, self.padval, device=device)
        # bout= input_padded(out, self.out_bool, mask, self.padval, device=device)
        
        return fout, None
    
class TEncoder(nn.Module):
    def __init__(self, Z_dim, S_dim, F_dim, 
                 eta_dim, inp_dim, rsample, nhidden, layers, pad_val, nhead=8):
        super(TEncoder, self).__init__()
        
        self.layers= layers
        self.nhidden= nhidden
        self.S_dim= S_dim
        self.Z_dim= Z_dim
        self.etadim= eta_dim
        self.padval= pad_val
        self.rsample= rsample
        
        self.tanh= nn.Tanh()
        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, batch_first= True)
        
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.in_linear= nn.Sequential(nn.Linear(F_dim, nhidden),
                                        nn.GELU(),
                                        nn.Linear(nhidden, nhidden))
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden, nhidden),
                                        nn.GELU(),
                                        nn.Linear(nhidden, Z_dim))
        
        self.eta_linear= nn.Sequential(nn.Linear(nhidden, nhidden),
                                        nn.GELU(),
                                        nn.Linear(nhidden, Z_dim),
                                        nn.Sigmoid())
        
        
    def forward(self, x, T_in, s=None, return_rweights= False):
        device= get_device(x)
        mask= get_mask(T_in)
        
        noise= self.rsample([x.shape[0], self.Z_dim]).to(device)
        
        # noise= self.rsample([x.shape[0], x.shape[1], self.etadim]).to(device)
        # x= torch.cat([x, noise], dim=-1)
        
        if self.S_dim > 0:
            x= torch.cat([x, s], dim=-1)
        
        x= input_padded(x, self.in_linear, mask, self.padval)
        
        enc_out= self.model(x, src_key_padding_mask= ~mask.to(device))
        
        # enc_out_end= enc_out[:, -1]
        enc_out_end= enc_out.max(dim=1)[0]
        
        out= self.out_linear(enc_out_end)
        eta_out= self.eta_linear(enc_out_end)
        
        # out= out + eta_out * (noise - out)
        out += (eta_out * noise)
        
        out= self.tanh(out)
        # out= torch.clamp(out, -1, 1)
        
        if return_rweights:
            return out, float(eta_out.mean())
        else:
            return out
        
class TDiscriminator(nn.Module):
    def __init__(self, F_dim,S_dim, 
                 inp_dim=50, nhidden=32, layers=2, pad_val= -2, nhead= 4, drop_rate= 0.2):
        super(TDiscriminator, self).__init__()
        
        self.layers= layers
        self.nhidden= nhidden
        self.S_dim= S_dim
        self.padval= pad_val
        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, batch_first= True, dropout= drop_rate)
        
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.in_linear= nn.Sequential(nn.Linear(F_dim, nhidden),
                                        nn.GELU())
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden, 1))
        
        
    def forward(self, x, T_in, s=None, return_rweights= False):
        device= get_device(x)
        mask= get_mask(T_in)
        
        ssm= square_subsequent_mask(x.shape[1])
        
        if self.S_dim > 0:
            x= torch.cat([x, s], dim=-1)
        
        x= input_padded(x, self.in_linear, mask, self.padval)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            enc_out= self.model(x, src_key_padding_mask= (~mask).to(device).float(), mask= ssm)
        
        out= input_padded(enc_out, self.out_linear, mask, self.padval)
        
        return out
    
class Encoder(nn.Module):
    def __init__(self, Z_dim, S_dim, F_dim, 
                 eta_dim, inp_dim, rsample, nhidden, layers, pad_val):
        super(Encoder, self).__init__()
        
        self.S_dim= S_dim
        self.etadim= eta_dim
        self.Zcat= F_dim
        self.padval= pad_val
        self.rsample= rsample
        self.Z_dim= Z_dim

        
        self.E_rnn = nn.GRU(
            input_size= self.Zcat, 
            hidden_size=nhidden,
            num_layers=layers, 
            batch_first=True,
        )
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden + eta_dim, nhidden),
                                        nn.GELU(),
                                        nn.Linear(nhidden, Z_dim),
                                        nn.Tanh())
        
        # self.out_noise_weights= nn.Sequential(nn.Linear(nhidden, nhidden),
        #                                 nn.GELU(),
        #                                 nn.Linear(nhidden, Z_dim),
        #                                 nn.Sigmoid())
        
        
    def forward(self, x,T_in, return_rweights= False):
        device= get_device(x)
        
        # noise= self.rsample([x.shape[0], self.Z_dim]).to(device)
        noise= self.rsample([x.shape[0], self.etadim]).to(device)
        # x= torch.cat([x,noise], dim=-1)
            
        # x= self.in_linear(x)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=T_in, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        _, hidden = self.E_rnn(x_pack)
        
        out= self.out_linear(torch.cat([hidden[-1], noise], dim=-1))
        # nweights= self.out_noise_weights(hidden[-1])
        
        # nweights= torch.clamp(nweights, 0.2, 1)
        
        # out= out + nweights * (noise - out)
        
        # out= torch.clamp(out, -1, 1)
        if return_rweights:
            return out, 1
        else:
            return out

def normalize_tensor(tensor, mask):
   masked_tensor = tensor * mask.unsqueeze(-1)
   sums = masked_tensor.sum(dim=1, keepdim=True)
   counts = mask.sum(dim=1, keepdim=True).unsqueeze(-1)
   means = sums / counts
   centered = masked_tensor - means * mask.unsqueeze(-1)
   squared_diffs = centered ** 2
   variances = squared_diffs.sum(dim=1, keepdim=True) / counts
   stds = torch.sqrt(variances + 1e-8)
   normalized = centered / stds
   return normalized * mask.unsqueeze(-1)    
    
class Discriminator(nn.Module):
    def __init__(self, F_dim,S_dim, 
                 inp_dim=32, nhidden=32, layers=2, pad_val= -2):
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
        
        self.in_linear= nn.Sequential(nn.Linear(self.Zcat, nhidden),
                                        nn.LeakyReLU(0.2),
                                        nn.Dropout1d(0.4),
                                        nn.Linear(nhidden, inp_dim))
        
        self.out_linear= nn.Sequential(nn.Linear(nhidden + self.Zcat, nhidden // 2),
                                        nn.LeakyReLU(0.2),
                                        nn.Dropout1d(0.4),
                                        nn.Linear(nhidden // 2, 1))
        
    def forward(self, x,T_in, mask= None, s= None):
        seqlen= max(T_in)
        
        xi= normalize_tensor(x, mask)
        # xi= torch.clone(x)
        
        if self.S_dim != 0:
            xi= torch.cat([xi, s], dim=-1)
        
        xi_emb= input_padded(xi, self.in_linear, mask, self.padval)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(
            input=xi_emb, 
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
    def __init__(self, Z_dim,nhidden=256, layers=2):
        super(linearDis, self).__init__()
        
        net= [nn.Linear(Z_dim, nhidden),nn.Dropout(0.1),nn.GELU()]
        for _ in range(layers):
            net.extend([nn.Linear(nhidden, nhidden),nn.Dropout(0.1),nn.GELU()])
        net.extend([nn.Linear(nhidden, 1)])
        
        self.net= nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)
    
class LDis(nn.Module):
   def __init__(self, input_dim, rsample):
       super().__init__()
       self.rsample= rsample
       hidden_dim = max(128, input_dim * 2)
       
       self.network = nn.Sequential(
           spectral_norm(nn.Linear(input_dim, hidden_dim)),
           nn.LeakyReLU(0.2),
           
           spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
           nn.LeakyReLU(0.2),
           
           spectral_norm(nn.Linear(hidden_dim // 2, 1))
       )
   
   def forward(self, x):
       noise= 0.02*self.rsample(x.shape).to(x.device)
       return self.network(x + noise)
    
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
        self.D= Discriminator(F_dim, S_dim, nhidden // 4, nhidden // 4, pad_val=pad_val)
        
        if self.fets:
            self.E= Encoder(Z_dim, S_dim, F_dim, eta_dim, inp_dim, rsample, nhidden, layers, pad_val)
            self.LD= LDis(Z_dim, rsample)
            
    def inf(self, T, device, mask=None, s=None, reset_hidden= True, zr= None):
        
        # with torch.no_grad():
        if zr == None:
            zr= self.sampler([len(T), self.Z_dim]).to(device)
        
        return self.G(zr, T, mask=mask,s=s, reset_hidden= reset_hidden)
