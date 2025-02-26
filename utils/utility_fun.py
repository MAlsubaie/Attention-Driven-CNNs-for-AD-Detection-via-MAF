import os
import matplotlib.pyplot as plt

def save_plot(history, metric, filename):
    dir_name = os.path.dirname(filename)
    if dir_name != "/" and dir_name != "./" and dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    
    plt.plot(history[metric])
    plt.plot(history['val_'+metric])
    plt.title('Model '+metric)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(filename)
    plt.close()