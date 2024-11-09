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

def fat(mse_nr,margin=0.1):
    mse= mse_nr.mean(dim=-1)
    
    mask= mse.argmax(dim=1)
    
    altmask= (mse > margin).long()
    maskmask= altmask.any(dim=1)
    
    mask[maskmask] = torch.argmax(altmask, dim=1)[maskmask]
    
    loss=torch.empty([mse.shape[0]]).cuda()
    for n,x in enumerate(mse):
        # if maskmask[n]:
        #     loss[n]= x[mask[n]]
        # else:
        #     loss[n] = mse[n].mean()
            
        loss[n]= x[mask[n]]
    
    return loss.mean(), mask.float()


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
        LDo= optim.Adam(net.LD.parameters(), lr=0.8*lr)
        Eo= optim.Adam(net.E.parameters(), lr=lr)
        
        LDs= optim.lr_scheduler.ExponentialLR(LDo, 0.1)
        Es= optim.lr_scheduler.ExponentialLR(Eo, 0.1)
        
    
    if lsgan:
        gan_loss= nn.MSELoss()
    else:
        gan_loss= nn.BCEWithLogitsLoss()
        
    minval=0
    mse_loss= nn.MSELoss(reduction= 'none')
    bce_loss= nn.BCEWithLogitsLoss(reduction= 'none')
    
    epoch_bar = trange(epochs, desc=f"Epoch: 0, Recon: 0, ADVF: 0, ADVE: 0", position=0, leave=True)
    tuneset= False
    # fat_index=1
    for epoch in epoch_bar:
        for i,(X,S,T) in enumerate(dataloader):
            
            # T= [int(fat_index)]*batch_size
            # t= int(torch.randint(1,99,[1]))
            # T= [t]*batch_size
            
            # T= [20] * batch_size
            x= X.to(dev)
            s= S.to(dev)
            mask= get_mask(T).to(dev)
            
            #Encoding-Generator Training
            #------------------
            if net.fets:
                Go.zero_grad()
                Eo.zero_grad()
                
                b_mask= x != minval
                b= b_mask.double()
                
                weights= torch.ones_like(b)
                weights[~b_mask] = 15
                
                zx, r_mean= net.E(x,T, return_rweights= True)
                G_x, G_x_bool= net.G(zx, T, mask=mask, s=s)
                G_x[~b_mask] = minval
                
                G_rand, G_rand_bool= net.inf(T, dev, mask=mask, s=s)
                tmask= mask * (torch.sigmoid(G_rand_bool) < 0.15)[:,:,0]
                G_rand[tmask] = minval
                
                mse_x= mse_loss(G_x, x)
                mse_x[:,:,-1] = 5 * mse_x[:,:,-1]
                
                # mse_x[mask]= mse_x[mask] / (torch.abs(x[mask]) + 1)
                
                bce_x= (weights[mask] * bce_loss(G_x_bool[mask], b[mask])).mean()
                recon, fat_index= fat(mse_x, thres)
                # fat_index= int(fat_index) + 5
                # print(fat_index)
                
                # recon= gan_loss(G_x, x)
                
                pred_ld= net.LD(zx)
                adv_ld= gan_loss(pred_ld, torch.ones_like(pred_ld))
                
                pred_d= net.D(G_x, T, mask, s)
                pred_d_inf= net.D(G_rand, T, mask, s)
                
                label_real= torch.ones_like(pred_d)
                label_real[~mask] = net.padval
                
                label_fake= torch.zeros_like(pred_d)
                label_fake[~mask] = net.padval
                
                # adv_d_1, d_fat_1= fat(mse_loss(pred_d, label_real), 0.95)
                # adv_d_2, d_fat_2= fat(mse_loss(pred_d_inf, label_real), 0.95)
                
                adv_d_1= mse_loss(pred_d, label_real)[mask].mean()
                adv_d_2= mse_loss(pred_d_inf, label_real)[mask].mean()
                
                adv_d= adv_d_1 + adv_d_2
                
                l_dist= mse_loss(G_rand[:,:,0][mask].mean(), x[:,:,0][mask].mean()) + mse_loss(G_rand[:,:,1][mask].mean(), x[:,:,1][mask].mean()) \
                    # + mse_loss(G_rand[:,:,0][mask].std(), x[:,:,0][mask].std()) + mse_loss(G_rand[:,:,1][mask].std(), x[:,:,1][mask].std())
                    
                l_sp= (s[:,:,3] - G_rand[:,:,1])[mask]
                l_sp[l_sp >= -2]= 0
                l_sp = 100 * torch.square(l_sp).mean()
                
                ge_objective= lam*recon + adv_d + adv_ld + 0.1*bce_x + l_sp# + 2*l_dist
                # ge_objective= lam*recon + adv_ld
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
            
            pred_fake_x= net.D(G_x.detach(), T, mask, s)
            pred_fake_x_inf= net.D(G_rand.detach(), T, mask, s)
            pred_real_x= net.D(x,T, mask, s)
            
            
            l_fake_x= gan_loss(pred_fake_x[mask], torch.zeros_like(pred_fake_x[mask]))
            l_fake_x_inf= gan_loss(pred_fake_x_inf[mask], torch.zeros_like(pred_fake_x[mask]))
            l_real_x= gan_loss(pred_real_x[mask], torch.ones_like(pred_real_x[mask]))
            
            # l_fake_x, _= fat(mse_loss(pred_fake_x, label_fake), 0.8)
            # l_fake_x_inf, _= fat(mse_loss(pred_fake_x_inf, label_fake), 0.8)
            # l_real_x, _= fat(mse_loss(pred_real_x, label_real), 0.8)
            
            
            d_x_loss= 0.25*l_fake_x + 0.25*l_fake_x_inf + 0.5*l_real_x
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
                logger.add_scalar("Scalars/FAT Index", float(fat_index.mean()))
                # logger.add_scalar("Scalars/D FAT Index", float(d_fat_2.mean()))
                logger.add_scalar("Scalars/Dropout", float(bce_x))
                logger.add_scalar("Scalars/Feature Adv Gen", float(adv_d))
                logger.add_scalar("Scalars/Feature Adv", float(d_x_loss))
                logger.add_scalar("Scalars/Embedding Adv", float(d_z_loss))
                logger.add_scalar("Scalars/G,E Objective", float(ge_objective))
                logger.add_scalar("Scalars/Random Weight", float(r_mean))
                logger.add_scalar("Scalars/Dist Loss", float(l_sp))
                
                logger.proj(zx.detach().cpu(), 'Zx')
                # logger.proj(net.sampler(zx.shape), 'rand')
                logger.plot(x[0].detach(), G_x[0].detach(), G_rand[0].detach(),T[0], net, dev, s=s[0].unsqueeze(0).detach())
            else:
                logger.add_scalar("Scalars/Feature Adv", float(adv_d))
            logger.step()
        
        
def inference(net, data, rsample= sample_uniform):
    net = net.cpu()
    x= data.X.cpu()
    s= data.S.cpu()
    mask= get_mask(data.T)
    
    # if logger is not None and net.fets:
    #     zx= net.E(x, data.T, s=s)
    #     logger.embed(zx)
    out, b= net.inf(data.T, 'cpu', mask, s)
    tmask= mask * (torch.sigmoid(b) < 0.15)[:,:,0]
    out[tmask] = 0
    
    return out

def one_step_inf(net, s, zr):
    T= [1] * s.shape[0]
    mask= get_mask(T)
    
    out, b= net.inf(T, models.get_device(s), mask, s, reset_hidden= False, zr= zr)
    tmask= mask * (torch.sigmoid(b) < 0.5)[:,:,0]
    out[tmask] = 0
    
    return out
    
    
        
    
