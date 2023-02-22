import yaml
import models
import data_logging
import dataset
import utils

import torch
from datetime import datetime

def main(p):
    data= utils.get_dataset(p['dataset'])
    
    model= models.fetsGan(p['z_dim'], p['s_dim'], data.shape[-1], 
                          eta_dim= p['eta_dim'], rsample=  getattr(utils, p['rsample']), 
                          nhidden= p['nhidden'], layers= p['layers'], pad_val= p['pad_val'])
    
    if p['use_logger']:
        logger= data_logging.logger('output/log1')
        logger.launch()
    else:
        logger= None
    
    print('Training...')
    utils.train(model, data, p['learning_rate'],
                thres= p['fat_thres'], batch_size= p['batch_size'], epochs= p['epochs'], dis_coef= p['dis_coef'],
                lsgan= p['lsgan'], lam= p['lambda'], logger=logger)
    
    print("\nRunning Inference...")
    output= utils.inference(model, data, rsample=  getattr(utils, p['rsample']), logger= logger)
    
    print("Saving Model...")
    current_datetime = datetime.now()
    save_name = current_datetime.strftime("%H:%M:%S_%m-%d-%Y")
    torch.save(model.state_dict(), f"output/models/{save_name}.pth")

if __name__ == '__main__':
    with open('params.yaml') as file:
        params = yaml.full_load(file)
    main(params)