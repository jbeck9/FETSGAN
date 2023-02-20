from torch.utils.tensorboard import SummaryWriter
import threading
import os
import webbrowser

import numpy as np
import matplotlib.pyplot as plt

def fig2img(fig, ax):
    ax.axis('off')
    fig.tight_layout(pad=0.1)
    ax.margins(0)
    fig.canvas.draw()
    
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

class logger():
    def __init__(self, logdir= 'output/logs'):
        self.index= 0
        self.logdir= logdir
        self.writer= SummaryWriter(logdir)
        
        plt.ioff()
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["figure.figsize"] = [5,2]
        
        
    def step(self):
        self.index += 1
        
    def launch(self, block= False):
        t1= threading.Thread(target=self._launchtb)
        t2= threading.Thread(target=self._launchwb)
        
        t1.start()
        t2.start()
        
        if block:
            t1.join()
            t2.join()
        
        
    def _launchtb(self):
        os.system('tensorboard --logdir=' + self.logdir)
        
    def _launchwb(self, url= "http://localhost:6006/"):
        webbrowser.open(url, new=2)
        
    def add_scalar(self, name, value):
        self.writer.add_scalar(name, value, self.index)
        
    def embed(self, data):
        self.writer.add_embedding(data, global_step=self.index)
        
    def proj(self, data):
        fig, ax= plt.subplots(1,1)
        ax.scatter(data[:,0], data[:,1])
        
        self.writer.add_figure("2D Projection", fig, self.index)
        
        # self.writer.add_image("2D Projection", fig2img(fig, ax))
        
        plt.close(fig)
    
    
    
if __name__ == '__main__':
    l= logger()
    l.launch()