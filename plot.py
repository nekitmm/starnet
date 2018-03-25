import numpy as np
import matplotlib.pyplot as plt  

LOGS_DIR = './logs/'                   # Output directory.

# list of nice colours
# thanks to this tutorial:
# http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
tableau20 = np.array([[ 31, 119, 180],
                      [174, 199, 232],
                      [255, 127,  14],
                      [255, 187, 120],
                      [ 44, 160,  44],
                      [152, 223, 138],
                      [214,  39,  40],
                      [255, 152, 150],
                      [148, 103, 189],
                      [197, 176, 213],
                      [140,  86,  75],
                      [196, 156, 148],
                      [227, 119, 194],
                      [247, 182, 210],
                      [127, 127, 127],
                      [199, 199, 199],
                      [188, 189,  34],
                      [219, 219, 141],
                      [ 23, 190, 207],
                      [158, 218, 229]], dtype = np.float)

tableau20 /=  255

def plot():
    createPlot(file_name = "perceptual_losses",
               plot_name = "Perceptual losses",
               data_labels = ['P 1', 'P 2', 'P 3', 'P 4', 'P 5', 'P 6', 'P 7', 'P 8'],
               X_label = "Epoch",
               data = np.loadtxt(LOGS_DIR + '/perceptual_losses.txt', skiprows = 1, delimiter = '	'))
    createPlot(file_name = "L1_loss",
               plot_name = "L1 loss",
               data_labels = [''],
               X_label = "Epoch",
               data = np.loadtxt(LOGS_DIR + '/L1_loss.txt', skiprows = 1, delimiter = '	'))
    createPlot(file_name = "accuracy",
               plot_name = "Accuracy, %",
               data_labels = [''],
               X_label = "Epoch",
               data = np.loadtxt(LOGS_DIR + '/accuracy.txt', skiprows = 1, delimiter = '	'))
    createPlot(file_name = "total_loss",
               plot_name = "Total loss",
               data_labels = [''],
               X_label = "Epoch",
               data = np.loadtxt(LOGS_DIR + '/total_loss.txt', skiprows = 1, delimiter = '	'))
    createPlot(file_name = "adversarial_loss",
               plot_name = "Adversarial loss",
               data_labels = ['GAN', 'Discriminative'],
               X_label = "Epoch",
               data = np.loadtxt(LOGS_DIR + '/adversarial_losses.txt', skiprows = 1, delimiter = '	'))
    
def createPlot(file_name, plot_name, data_labels, X_label, data):
    # plot size with 16:9 ratio
    plt.figure(figsize = (16, 9))
    
    # start plot
    ax = plt.subplot(111)
    
    # remove top and right box boundaries
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # define boundaries for plot
    MAX_X = int(data[-1, 0]) + 2
    MAX_Y = int(np.max(data[:, 1:]) * 12) / 10
    plt.ylim(0, MAX_Y)    
    plt.xlim(0, MAX_X)
    
    # ticks and lines
    yaxis_range = np.arange(0, MAX_Y, MAX_Y / 10)
    xaxis_range = np.arange(0, MAX_Y, MAX_Y / 10)
    plt.yticks(yaxis_range, [str(x) for x in yaxis_range], fontsize = 20)
    plt.xticks(fontsize = 20)
    for y in yaxis_range:
        plt.plot(range(0, MAX_X + 1), [y] * len(range(0, MAX_X + 1)), "--", lw = 0.5, color = "black", alpha = 0.3)
        
    plt.tick_params(axis = "both", which = "both", bottom = "on", top = "off",
                    labelbottom = "on", left = "on", right = "off", labelleft = "on")
    
    # list of positions of data labels
    ly_pos = np.arange(0.25, 1, 0.06) * MAX_Y
    
    # selected list of colours
    c_ind = [0, 4, 16, 6, 3, 10, 8, 14]
    
    # X axis label
    plt.text(MAX_X + 0.25, 0, X_label, fontsize = 20, color = "black")
    
    #plot the data
    for i in range(1, data.shape[1]):
        # select colour
        color = c_ind[i - 1]
        # plot
        plt.plot(data[:, 0], data[:, i], lw = 1.5, color = tableau20[color])
        # add label for this data
        plt.text(MAX_X + 0.5, ly_pos[i - 1], data_labels[i - 1], fontsize = 20, color = tableau20[color])
    
    # add caption
    plt.text(MAX_X / 2, MAX_Y * 1.05, plot_name, fontsize = 25, ha = "center")
    plt.savefig(LOGS_DIR + file_name + ".png", bbox_inches = "tight")
    plt.close()