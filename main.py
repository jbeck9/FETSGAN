import yaml
import models
import data_logging
import utils
import dataset
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(p):
    data= dataset.ETSDataset(dataset.load_data(p['dataset']), p['s_dim'])
    
    model= models.fetsGan(p['z_dim'], p['s_dim'], data.X.shape[-1], 
                          eta_dim= p['eta_dim'], rsample=  getattr(utils, p['rsample']), 
                          nhidden= p['nhidden'], layers= p['layers'], pad_val= p['pad_val'])
    
    # print(count_parameters(model.D))
    # return
    
    if p['load_model']:
        data_logging.load_model(model)
    
    if p['use_logger']:
        logger= data_logging.logger('output/log1')
        logger.launch()
    else:
        logger= None
    
    if p['run_training']:
        print('Training...')
        utils.train(model, data, p['learning_rate'],
                    thres= p['fat_thres'], batch_size= p['batch_size'], epochs= p['epochs'], dis_coef= p['dis_coef'],
                    lsgan= p['lsgan'], lam= p['lambda'], logger=logger)
    
    if p['run_inf']:
        print("\nRunning Inference...")
        a= time.time()
        output= utils.inference(model, data, rsample=  getattr(utils, p['rsample']), logger= logger)
        print(time.time() - a)
    
    if p['save_output']:
        print("Saving Model...")
        data_logging.save_output(model, output)
    

if __name__ == '__main__':
    with open('params.yaml') as file:
        params = yaml.full_load(file)
    out= main(params)