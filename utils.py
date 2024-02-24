import torch
import torch.optim as optim
import torch.nn as nn

import dataset
import models
import data_logging

from tqdm import trange

def sample_uniform(shape, hlim=1, llim= -1):
    x= torch.empty(shape)
    return x.uniform_(llim, hlim)

def sample_stdgaussian(shape):
    return torch.randn(shape)

def get_mask(T):
    out= torch.zeros([len(T), int(max(T))]).long()
    
    for n in range(len(T)):
        ind= int(T[n])
        out[n, :ind] = True
        
    return out.bool()

def fat(x,l,margin=0.1, usesum=False):
    mse= torch.square(x - l).mean(dim=-1)
    
    mask= mse.argmax(dim=1)
    
    altmask= (mse > margin).long()
    maskmask= altmask.any(dim=1)
    
    mask[maskmask] = torch.argmax(altmask, dim=1)[maskmask]
    
    loss=torch.empty([x.shape[0]]).cuda()
    for n,x in enumerate(mse):
        if usesum == True:
            ind=mask[n] + 1
            loss[n]= x[:ind].mean()
        else:
            loss[n]= x[mask[n]]
    
    return loss.mean(), mask.float().mean()


def train(net, data, lr, thres= 0.1, batch_size= 256, epochs=2000, dis_coef=2, lsgan=True, lam= 10, logger= None):
    
    if torch.cuda.is_available():
        dev= 'cuda'
    else:
        dev= 'cpu'
    
    dataloader = torch.utils.data.DataLoader(
        dataset=data, 
        batch_size=batch_size,
        shuffle=True,
        drop_last= True
    )
    
    net= net.to(dev)
    
    Go= optim.Adam(net.G.parameters(), lr=lr)
    Do= optim.Adam(net.D.parameters(), lr=dis_coef*lr)
    
    Gs= optim.lr_scheduler.ExponentialLR(Go, 0.1)
    Ds= optim.lr_scheduler.ExponentialLR(Do, 0.1)
    
    
    if net.fets:
        LDo= optim.Adam(net.LD.parameters(), lr=dis_coef*lr)
        Eo= optim.Adam(net.E.parameters(), lr=lr)
        
        LDs= optim.lr_scheduler.ExponentialLR(LDo, 0.1)
        Es= optim.lr_scheduler.ExponentialLR(Eo, 0.1)
        
    
    if lsgan:
        gan_loss= nn.MSELoss()
    else:
        gan_loss= nn.BCEWithLogitsLoss()
    
    epoch_bar = trange(epochs, desc=f"Epoch: 0, Recon: 0, ADVF: 0, ADVE: 0", position=0, leave=True)
    tuneset= False
    # fat_index=1
    for epoch in epoch_bar:
        for i,(X,S,T) in enumerate(dataloader):
            
            # T= [int(fat_index)]*batch_size
            # t= int(torch.randint(1,99,[1]))
            # T= [t]*batch_size

            x= X.to(dev)
            s= S.to(dev)
            mask= get_mask(T).to(dev)
            
            #Encoding-Generator Training
            #------------------
            if net.fets:
                Go.zero_grad()
                Eo.zero_grad()
                
                zx= net.E(x,T, s=s)
                G_x= net.G(zx, T, mask=mask, s=s)
                
                G_rand= net.inf(T, dev, mask=mask, s=s)
                
                recon, fat_index= fat(G_x, x, thres)
                # fat_index= int(fat_index) + 5
                # print(fat_index)
                
                # recon= gan_loss(G_x, x)
                
                pred_ld= net.LD(zx)
                adv_ld= gan_loss(pred_ld, torch.ones_like(pred_ld))
                
                pred_d= net.D(G_rand, T, mask, s)[mask]
                adv_d= gan_loss(pred_d, torch.ones_like(pred_d))
                
                ge_objective= lam*recon + adv_d + adv_ld
                # ge_objective= adv_ld
                ge_objective.backward()
                Go.step()
                Eo.step()
                
            else:
                Go.zero_grad()
                
                G_x= net.G(x, T, s=s)
                
                pred_d= net.D(G_x, T, mask, s)[mask]
                adv_d= gan_loss(pred_d, torch.ones_like(pred_d))
                
                ge_objective= adv_d
                ge_objective.backward()
                Go.step()
            #------------------
            #Feature Discriminator Training
            #------------------
            Do.zero_grad()
            
            pred_fake_x= net.D(G_rand.detach(), T, mask, s)[mask]
            pred_real_x= net.D(x,T, mask, s)[mask]
            
            l_fake_x= gan_loss(pred_fake_x, torch.zeros_like(pred_fake_x))
            l_real_x= gan_loss(pred_real_x, torch.ones_like(pred_real_x))
            
            d_x_loss= 0.5*(l_fake_x + l_real_x)
            d_x_loss.backward()
            
            Do.step()
            #------------------
            #Encoding Discriminator Training
            #------------------
            if net.fets:
                LDo.zero_grad()
                
                pred_fake= net.LD(zx.detach())
                pred_real= net.LD(net.sampler(zx.shape).to(dev))
                
                l_fake_z= gan_loss(pred_fake, torch.zeros_like(pred_fake))
                l_real_z= gan_loss(pred_real, torch.ones_like(pred_real))
                
                d_z_loss= 0.5*(l_fake_z + l_real_z)
                d_z_loss.backward()
                
                LDo.step()
                
            #------------------
        if (epoch / epochs) > 0.8 and not tuneset:
            Gs.step()
            Ds.step()
            if net.fets:
                Es.step()
                LDs.step()
            tuneset= True
        
        if net.fets:
            epoch_bar.set_description(f"Epoch: {epoch}, Recon: {float(recon):.4f}, ADVF: {float(adv_d):.4f}, ADVE: {float(adv_ld):.4f}")
        else:
            epoch_bar.set_description(f"Epoch: {epoch}, ADVF: {float(adv_d):.4f}")
        
        if logger is not None:
            if net.fets:
                logger.add_scalar("Scalars/Recon", float(recon))
                logger.add_scalar("Scalars/Feature Adv", float(d_x_loss))
                logger.add_scalar("Scalars/Embedding Adv", float(d_z_loss))
                logger.add_scalar("Scalars/G,E Objective", float(ge_objective))
                
                logger.proj(zx.detach().cpu(), 'Zx')
                # logger.proj(net.sampler(zx.shape), 'rand')
                logger.plot(x[0].detach(), G_x[0].detach(),T[0], net, dev, s=s[0].unsqueeze(0).detach())
            else:
                logger.add_scalar("Scalars/Feature Adv", float(adv_d))
            logger.step()
        
        
def inference(net, data, rsample= sample_uniform, logger= None):
    net = net.cpu()
    x= data.X
    s= data.S
    mask= get_mask(data.T)
    
    # if logger is not None and net.fets:
    #     zx= net.E(x, data.T, s=s)
    #     logger.embed(zx)
    return net.inf(data.T, 'cpu', mask, s)
    
    
        
    