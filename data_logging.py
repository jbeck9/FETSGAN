from torch.utils.tensorboard import SummaryWriter
import threading
import os
import webbrowser

import numpy as np
import matplotlib.pyplot as plt

import shutil
import time

class logger():
    def __init__(self, logdir= 'output/logs'):
        self.index= 0
        self.logdir= logdir
        self.writer= SummaryWriter(logdir)
        
        plt.ioff()
        self.default_plot_settings= [plt.rcParams["figure.dpi"], plt.rcParams["figure.figsize"]]
        
        
    def step(self):
        self.index += 1
        
    def launch(self, block= False, remove_dir= True):
        if remove_dir:
            shutil.rmtree(self.logdir, ignore_errors=True)
            os.mkdir(self.logdir)
        
        t1= threading.Thread(target=self._launchtb)
        t2= threading.Thread(target=self._launchwb)
        
        t1.start()
        t2.start()
        
        if block:
            t1.join()
            t2.join()
        
        
    def _launchtb(self):
        os.system('tensorboard --logdir=' + self.logdir + ">/dev/null 2>&1")
        
    def _launchwb(self, url= "http://localhost:6006/"):
        webbrowser.open(url, new=2)
        
    def add_scalar(self, name, value):
        self.writer.add_scalar(name, value, self.index)
        
    def embed(self, data, plot_dim=0):
        self.writer.add_embedding(data, global_step=self.index)
        
    def proj(self, data, sample_rate= 100):
        if self.index % sample_rate == 0:
            
            plt.rcParams["figure.dpi"] = 200
            plt.rcParams["figure.figsize"] = [5,2]
            
            fig, ax= plt.subplots(1,1)
            ax.scatter(data[:,0], data[:,1])
            
            self.writer.add_figure("2D Projection", fig, self.index)
            # self.writer.add_image("2D Projection", fig2img(fig, ax))
            plt.close(fig)
            
            plt.rcParams["figure.dpi"] = self.default_plot_settings[0]
            plt.rcParams["figure.figsize"] = self.default_plot_settings[1]
    
    def plot(self, x,T, net, device, sample_rate= 100, plot_index= 0):
        if self.index % sample_rate == 0:
            x= x.cpu()
            rx= net.inf([T], device).cpu()[0]
            
            plt.rcParams["figure.dpi"] = 200
            plt.rcParams["figure.figsize"] = [5,2]
            
            fig, ax= plt.subplots(1,1)
            ax.plot(x[:,plot_index], label="x")
            ax.plot(rx[:,plot_index], label= "$\hat{x}$")
            plt.legend()
            
            self.writer.add_figure("Sample Plot", fig, self.index)
            plt.close(fig)
            
            plt.rcParams["figure.dpi"] = self.default_plot_settings[0]
            plt.rcParams["figure.figsize"] = self.default_plot_settings[1]
            
            
    def ts2img(self, ts, index=0):
        plt.rcParams["figure.dpi"] = 200
        plt.rcParams["figure.figsize"] = [5,2]
        
        fig, ax= plt.subplots(1,1)
        ax.plot(ts[:,index])
        
        ax.axis('off')
        fig.tight_layout(pad=0.1)
        ax.margins(0)
        fig.canvas.draw()
        
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img= image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,)).astype(float)
        
        plt.close(fig)
        
        plt.rcParams["figure.dpi"] = self.default_plot_settings[0]
        plt.rcParams["figure.figsize"] = self.default_plot_settings[1]
        
        return np.moveaxis(img, 2,0) / 255
    