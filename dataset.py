import torch
import json
import numpy as np

def norm(x, min, max, a=-1, b=1):
    return (((b-a) * (x - min)) / (max - min)) + a

def load_data(name):
    if name == 'sines':
        return gen_sines()
    
    else:
        f = open(f"data/{name}.json", "r")
        js = f.read()
        f.close()
        return np.array(json.loads(js))

def gen_sines(nvar=1, T=100, samples=500, amp_range= [0.2, 1], f_range= [1, 10], xo_range= [0, 2*np.pi]):
    output= np.empty([samples, T, nvar])
    for n in range(samples):
        
        rand_shape= [1, nvar]
        
        t= np.tile(np.arange(0,1,1/T), (nvar,1)).T
        
        amp= np.random.uniform(amp_range[0], amp_range[1], rand_shape)
        f= np.random.uniform(f_range[0], f_range[1], rand_shape)
        xo= np.random.uniform(xo_range[0], xo_range[1], rand_shape)
        
        y= amp*np.sin(2*np.pi*f*t + xo)
        
        output[n] = y
        
    return output

class ETSDataset(torch.utils.data.Dataset):
    def __init__(self, data, S_dim,  padval=-2):
        
        self.padval= padval
        
        time= np.count_nonzero((data[:,:,-1] != padval), 1).tolist()
        
        self.T = torch.LongTensor(time)
        self.X = torch.FloatTensor(data)[:,:,S_dim:]
        self.S = torch.FloatTensor(data)[:,:,:S_dim]
        
        self.max_length= max(self.T)
        self.shape= data.shape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x= self.X[idx]
        s= self.S[idx]
        t= self.T[idx]
    
        return (x,s,t)