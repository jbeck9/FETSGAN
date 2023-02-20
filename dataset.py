import torch
import json
import numpy as np

def norm(x, min, max, a=-1, b=1):
    return (((b-a) * (x - min)) / (max - min)) + a

def toy_loader(name):
    if name in ['stock', 'metro', 'energy']:
        f = open(f"data/{name}.json", "r")
        js = f.read()
        f.close()
        return np.array(json.loads(js))
    
    elif name == 'sines':
        return gen_sines()

def gen_sines(nvar=1, T=100, samples=1000, amp_range= [0.2, 1], f_range= [1, 10], xo_range= [0, 2*np.pi]):
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
    def __init__(self, data, padval=-2):
        
        self.padval= padval
        
        time= np.count_nonzero((data[:,:,-1] != padval), 1).tolist()
        
        self.T1 = torch.LongTensor(time)
        self.X1 = torch.FloatTensor(data)
        
        self.max_length= max(self.T1)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        x= self.X1[idx]
        t= self.T1[idx]
    
        return x,t