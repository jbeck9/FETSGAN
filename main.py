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
        output= utils.inference(model, data, rsample=  getattr(utils, p['rsample']))
        print(time.time() - a)
    
    if p['save_output']:
        print("Saving Model...")
        data_logging.save_output(model, output)
    

if __name__ == '__main__':
    with open('params.yaml') as file:
        params = yaml.full_load(file)
    # out= main(params)
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import numpy as np
    
    p= params
    data= dataset.ETSDataset(dataset.load_data(p['dataset']), p['s_dim'])
    model= models.fetsGan(p['z_dim'], p['s_dim'], data.X.shape[-1], 
                          eta_dim= p['eta_dim'], rsample=  getattr(utils, p['rsample']), 
                          nhidden= p['nhidden'], layers= p['layers'], pad_val= p['pad_val'])
    data_logging.load_model(model)
    
    torch.save(model.G.state_dict(), "pem.pth")
    
    # raise
    
    
    
    # data.S[:,:,0] = 0.52* torch.ones_like(data.S[:,:,0])
    
    nruns= 5
    infs=[]
    dr= []
    with torch.no_grad():
        for _ in range(nruns):
            out= utils.inference(model, data, rsample=  getattr(utils, p['rsample']))
            infs.append(out)
            dr.append(torch.count_nonzero(out == 0) / torch.count_nonzero(out > 0))
    
    S= data.S
    # dev= models.get_device(S)
    # rsample=  getattr(utils, p['rsample'])
    # zr= rsample([S.shape[0], model.Z_dim]).to(dev)
    # out= torch.zeros([S.shape[0], S.shape[1], model.F_dim])
    # for n in range(S.shape[1]):
    #     out[:,n]= utils.one_step_inf(model, S[:,n].unsqueeze(1), zr=zr)[:,0]
        
    # out= out.detach().cpu()
            
    # sns.kdeplot(np.array(dr))
            
    
    fig,ax= plt.subplots(1,1)
    for i in range(data.shape[0]):
        i = i*5
        # plt.plot(data.S[i, :data.T[i], 3], label= "vlead (radar) (smoothed)", c= 'green')
        plt.plot(data.S[i, :data.T[i], 3] - data.X[i, :data.T[i], 1], label= "model", c= 'blue')
        plt.plot(data.S[i, :data.T[i], 3], label= "vLead", c= 'red')
        # for n in range(nruns):
            # plt.plot(infs[n][i, :data.T[i], 1], label= "$\hat{x}$", c= 'red', alpha= 0.4)
            
        # plt.plot(out[i, :data.T[i]], label= "$single_x$", c= 'purple')
            
        # plt.plot(data.S[i, :data.T[i], -1], label= "gt (radar)", c= 'green')
        
        data_logging.legend_without_duplicate_labels(ax)
        # ax.set_title( "PEM (Speed of Lead Vehicle)",
        #              fontdict={'fontsize' :15})
        ax.set_xlabel("t(20 ms)", fontdict={'fontsize' :12})
        ax.set_ylabel("Distance (m/s)", fontdict={'fontsize' :12})
        
        plt.show()
        plt.pause(0.8)
        input(i)
        plt.clf()