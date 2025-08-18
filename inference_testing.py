import yaml
import models
import data_logging
import utils
import dataset
import time

import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    with open('params.yaml') as file:
        p = yaml.full_load(file)
    
    data= dataset.ETSDataset(dataset.load_data(p['dataset']), p['s_dim'], padval= p['pad_val'])
    model= models.fetsGan(p['z_dim'], p['s_dim'], data.X.shape[-1], 
                          eta_dim= p['eta_dim'], rsample=  getattr(utils, p['rsample']), 
                          nhidden= p['nhidden'], layers= p['layers'], pad_val= p['pad_val'])
    
    data_logging.load_model(model)
    model.eval()
    
    with torch.no_grad():
        out= utils.inference(model, data, rsample=  getattr(utils, p['rsample']))
    
    fig,ax= plt.subplots(1,1)
    for i in range(len(out)):
        ax.plot(data.X[i, :, 0])
        ax.plot(out[i, :, 0])
        
        plt.show()
        plt.pause(2)
        ax.clear()