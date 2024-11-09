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
        data_logging.save_output(model)
    

if __name__ == '__main__':
    with open('params.yaml') as file:
        params = yaml.full_load(file)
        
    # out= main(params)
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import numpy as np
    
    import random
    import pickle
    random.seed(1)
    
    p= params
    data= dataset.ETSDataset(dataset.load_data(p['dataset']), p['s_dim'])
    model= models.fetsGan(p['z_dim'], p['s_dim'], data.X.shape[-1], 
                          eta_dim= p['eta_dim'], rsample=  getattr(utils, p['rsample']), 
                          nhidden= p['nhidden'], layers= p['layers'], pad_val= p['pad_val'])
    data_logging.load_model(model)
    
    model.eval()
    torch.save(model.G.state_dict(), "pem.pth")
    
    # raise
    
    
    
    # data.S[:,:,0] = 0.52* torch.ones_like(data.S[:,:,0])
    
    nruns= 2
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
    
    # pickle.dump(data.X, open('X.p', 'wb'))
    # pickle.dump(S, open('S.p', 'wb'))
    # pickle.dump(infs, open('X_hat_list.p', 'wb'))
    # raise
    
    fig,ax= plt.subplots(1,1)
    for _ in range(data.shape[0]):
        i = np.random.randint(0, data.shape[0])
        # plt.plot(data.S[i, :data.T[i], 3], label= "vlead (radar) (smoothed)", c= 'green')
        realmask= data.X[i, :data.T[i], 0] == 0
        real_traj= data.S[i, :data.T[i], 3] - data.X[i, :data.T[i], 1]
        real_traj[realmask] = 0
        # plt.plot(real_traj, label= "model", c= 'blue')
        plt.plot(data.S[i, :data.T[i], 3], label= "$Y$", c= 'green')
        plt.plot(real_traj, label= "$Y - x$", c= 'blue')
        for n in range(nruns):
            genmask= infs[n][i, :data.T[i], 0] == 0
            gen_traj= data.S[i, :data.T[i], 3] - infs[n][i, :data.T[i], 1]
            gen_traj[genmask] = 0
            
            plt.plot(gen_traj, label= "$Y - \hat{x}$", c= 'red', alpha= 0.4)
            
        # plt.plot(out[i, :data.T[i]], label= "$single_x$", c= 'purple')
            
        # plt.plot(data.S[i, :data.T[i], -1], label= "gt (radar)", c= 'green')
        
        data_logging.legend_without_duplicate_labels(ax)
        ax.set_title( "Perception Outputs (Lead Vehicle Distance)",
                      fontdict={'fontsize' :13})
        ax.set_xlabel("t(20 ms)", fontdict={'fontsize' :13})
        ax.set_ylabel("Distance (m)", fontdict={'fontsize' :13})
        # plt.legend()
        
        fig.set_size_inches(8, 5)
        fig.tight_layout()
            
        plt.show()
        plt.pause(0.8)
        
        # if (real_traj < 0).any() or (gen_traj < 0).any():
        inp= input(i)
        if inp == 'y':
            fig.savefig(f'/home/cosmos/Pictures/figs/{i}', bbox_inches='tight')
        ax.clear()