# -*- coding: utf-8 -*-
"""
Spyder Editor

This is the code for Yusuf's MDPI Entropy paper draft script file.
Only change some parameter values at the beginning to generate all related figures automatically.


REvised to calculate the Mutual Information using binned input. The MI code is
copied from Harvard paper code at: 
    https://github.com/artemyk/ibsgd/blob/iclr2018/simplebinmi.py
    
13 Dec 2021

"""

#%%
import glob
import time
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import pandas as pd
import imageio as im
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import load_model

#%% Set the testing environmental variables

# CNN kernel size. 3 means 3x3, 5 means 5x5 and 7 means 7x7
kernel_size = 3
cnn_number  = 1


# Hypo-parameters for model configuration
batch_size = 1024
epochs = 100

# Training starting and ending image index for mutual information calculation
image_start = 1000
image_end = 1120
dense_unit  = 512
activationFunc = 'tanh'   # tanh, relu

# path of saved epoch models, figures
path_epoch_models = "./Epoch_models/"

#%%
# Information Theoretic Quantities Calculation Functions
def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en


#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.concatenate((Y,X), axis=0)
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)


#Information Gain (or the Mutual Information)
def gain(Y, X):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return entropy(Y) - cEntropy(Y,X)

# Probability calculation of array x
# From https://github.com/artemyk/ibsgd/blob/iclr2018/simplebinmi.py
def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x) #.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

# Mutual Information calculation between inputdata X and layerdata T using binned method
# From https://github.com/artemyk/ibsgd/blob/iclr2018/simplebinmi.py
# inputdata is X, layerdata is Y. return I(Y,X) = sum[p(x) * H(y|x)]
def calc_MI(inputdata, layerdata, num_of_bins):
    p_xs, unique_inverse_x = get_unique_probs(inputdata)
    
    bins = np.linspace(0, 1, num_of_bins, dtype='float32') 
    digitized = bins[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins) - 1].reshape(len(layerdata), -1)
    p_ts, _ = get_unique_probs( digitized )
    
    H_LAYER = -np.sum(p_ts * np.log(p_ts))
    H_LAYER_GIVEN_INPUT = 0.
    for xval in unique_inverse_x:
        p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
        H_LAYER_GIVEN_INPUT += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))
    return np.abs(H_LAYER - H_LAYER_GIVEN_INPUT) 

#reshape a image from 28x28x1 to 10x28x1
def image_newShape(x):
    new_x = []
    for i in range(4):
        y=0.0
        for j in range(3):
            y += x[i*3+j]
        new_x.append(y / 3.0)
        
    for i in range(2):
        y=0.0
        for j in range(2):
            y += x[12+i*2+j]
        new_x.append(y / 2.0)
        
    for i in range(4):
        y=0.0
        for j in range(3):
            y += x[16+i*3+j]
        new_x.append(y / 3.0)
        
    return new_x

# reshape an array of n image from n x 28 x 28 x 1 to n*10 x 28 x 1
def reshape_images(input_image):
    output_image = []
    num_of_images = len(input_image)
    for i in range(num_of_images):
        new_image = image_newShape(input_image[i])
        output_image.append(new_image)
    
    output_image = np.asarray(output_image).reshape((num_of_images*10, 28))
    return np.asarray(output_image).reshape((num_of_images*10, 28))

# reshape an array of n image from n x 28 x 28 x 1 to n x 10 x1
def reshape_images2(input_image):
    output_image = []
    num_of_images = len(input_image)
    for i in range(num_of_images):
        new_image = np.sum(image_newShape(input_image[i]), axis=1)/28.0
        output_image.append(new_image)
    
    output_image = np.asarray(output_image).reshape((num_of_images, 10))
    return output_image #np.asarray(output_image).reshape((num_of_images*10, 28))

#%%
max_cnn = cnn_number
start_time_all = time.time()

for cnn_number in range(1, max_cnn+1):
    total_layers = cnn_number + 4 # We have 4 additional layers: MaxPool, Faltten, Dense1, Dense2 for Output
    path_figures = "./Figures/kernel_" + str(kernel_size) + "x" + str(kernel_size) + "/" + str(cnn_number) + "-layer/"

    # figuure out the suffix letter for different CNN layers: 1->figure_x_a, 2->figure_x_b, ...
    figSuffix = chr(ord('a') + cnn_number - 1);

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), padding='same', input_shape = (28, 28, 1), activation = activationFunc))
    
    # Steps - Add more hidden CNN layers, by Xiyu
    layer_count = cnn_number - 1
    while layer_count > 0:
        layer_count = layer_count - 1      
        classifier.add(Conv2D(32, (kernel_size, kernel_size), padding='same', activation=activationFunc))
    
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(units = dense_unit, activation = activationFunc)) #or 128 or 512
    classifier.add(Dense(units = 10, activation = 'softmax'))  # Output layer
    
    classifier.summary()
    
    # Compiling the CNN
    classifier.compile(optimizer = 'rmsprop',
                        loss = 'categorical_crossentropy', 
                        metrics = ['accuracy'])
    
    #%%
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    
    l_train = y_train
    l_test = y_test
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    
    #%%
    # Plotting histogram of randomly selected images (Figures 2a and 2b)
    a=x_train[image_start + 150]
    b=x_train[image_start + 160]
    unique, count = np.unique(a.flatten(), return_counts=True, axis=0)
    x,y= np.unique(b.flatten(), return_counts=True, axis=0)
    #prob = count/len(a.flatten())
    
    fig2a=plt.figure(dpi=600)
    plt.ylim(0, 700)
    plt.bar(unique,count,width=0.01)
    plt.xlabel("Normalized Pixel Values")
    plt.ylabel("Count")
    #plt.title(f"Histogram of Image 1")
    plt.tight_layout();
    fig2a.savefig(path_figures + "Figure_2a.png", dpi=600)
    
    fig2b = plt.figure(dpi=600)
    plt.ylim(0, 700)
    plt.bar(x, y, width=0.01)
    plt.xlabel("Normalized Pixel Values")
    plt.ylabel("Count")
    #plt.title(f"Histogram of Image 2")
    plt.tight_layout();
    plt.show()
    fig2b.savefig(path_figures + "Figure_2b.png", dpi=600)
    
    
    # Printing Information Thoeretic Calculations of Images ( Table 5.5)
    
    print("\nEntropy of Image 1, H(X): ",entropy(a.flatten()))
    print("Entropy of Image 2, H(Y): ",entropy(b.flatten()))
    print("Mutual Information of Images,  I(X;Y): ",gain(a.flatten(),b.flatten()))
    print("Conditional Entropy of Images, H(X|Y) : ",cEntropy(a.flatten(),b.flatten()))
    print("Conditional Entropy of Images, H(Y|X) : ",cEntropy(b.flatten(),a.flatten()))
    print("Joint Entropy of Images, H(X,Y): ",jEntropy(a.flatten(),b.flatten()))
    
    #%%
    # Training  and Saving Designed Model
    checkpoint = keras.callbacks.ModelCheckpoint(path_epoch_models + 'Model_{epoch}.h5', save_freq='epoch') 
    
    start_time = time.time()
    history=classifier.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=batch_size, epochs=epochs,callbacks=[checkpoint])
    print('Training took {} seconds'.format(time.time()-start_time))
    
    # Save training history as a history dictionary file so we can reload late to draw the accuracy and loss curves
    with open(path_figures + 'histroyDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Reload the history as follows
    # history = pickle.load(open(path_figures + 'historyDict', "rb"))
    
    # Save training history as a CSV file
    # Convert the history.history dict to a pandas DataFrame:    
    hist_df = pd.DataFrame(history.history) 
    hist_df.to_csv(path_figures + 'history.csv')
    
    #%%
    # Plotting of Training and Validation Curves (Figure 5.7)
    '''The output of model.fit is a model.History object which is a record of metrics at each epoch. This can be used to graph the training and validation accuracy
    to see where they plateaued off and if overfitting can subsequently be avoided'''
    
    # plot loss
    # pyplot.subplot(211)
    fig = plt.figure(dpi=600)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], color='blue',label="Train")
    plt.plot(history.history['val_loss'], color='red', label='Test')
    #pyplot.title("MNIST Data Trining")
    plt.legend()
    plt.tight_layout();
    fig.savefig(path_figures + "loss.png", dpi=600)
    
    # plot accuracy
    # pyplot.subplot(212)
    fig = plt.figure(dpi=600)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='Train')
    plt.plot(history.history['val_accuracy'], color='red', label='Test')
    plt.legend()
    plt.tight_layout();
    fig.savefig(path_figures + "accuracy.png", dpi=600)
    plt.show()
    
    #%%
    from tensorflow.keras.models import load_model
    ## Loading the saved weights of each epoch to the new model variable Model 
    
    all_epoch_model_list = []
    for i in range(1, epochs+1):
        epoch_model_name = 'Model_' + str(i) + '.h5'
        Model = load_model(path_epoch_models + epoch_model_name)    
        all_epoch_model_list.append(Model)
        
    #%%
    # Calculating Mutual Information Between Input and Output I(X;T), output and labels I(Y;T) (Figure 3, Figure 4) for each epoch model
    images=x_train[image_start:image_end]
    images_averaged = reshape_images2(images)
    images_index = np.arange(image_start, image_end, 1).reshape((image_end - image_start), 1)
    images_lable = y_train[image_start:image_end]
    #images_averaged = np.sum(images, axis=(1, 2))/(image_end - image_start)
    
    foutputs=[]
    start_time= time.time();
    for i, model in enumerate(all_epoch_model_list):
        layer_outputs=[layer.output for layer in model.layers[:]]
        activation_model=models.Model(inputs=model.input,outputs=layer_outputs)
        foutputs.append(activation_model.predict(images))   # foutputs[i][j] has the output of every layer j for each predicated image i.
    print('Testing {} images for every epoch took {} seconds'.format(image_end-image_start, time.time()-start_time))
    
    # Save output to a .csv file
    pd_foutputs = pd.DataFrame(foutputs)
    pd_foutputs.to_csv(path_figures + 'foutputs.csv')
    
    # after saving the training output of every layers, we can have a separated application to do the data analysis.
    with open('foutputs.pkl','wb') as file_out: 
        pickle.dump(foutputs, file_out)
    file_out.close()
    
    # with open('foutputs.pkl','rb') as file_in:
    #     foutputs = pickle.load(file_in)
    # file_in.close()
        
    #Mutual Information Between Input and Output , Output and Label 
    i_YT=[] #I(T;T) betweeen output T and training lable Y
    i_XT=[] #I(X;T) between input X and output T
    for i in range(len(foutputs)):
        # the following two lines are using the old wrong MI calculation mathod. All figures in the paper are based on this.
        # i_YT.append(gain(foutputs[i][total_layers-1].flatten(), y_train[image_start:image_end].flatten()))  # number 6 is the last layer (layer7). if we have 4 layers, it should be 3
        # i_XT.append(gain(images.flatten(), foutputs[i][total_layers-1].flatten()))
        
        i_YT.append(calc_MI(foutputs[i][total_layers-1].flatten(), images_lable.flatten(), 11))
        
        # Use different methods to calculate the MI between input and output lable.
        #i_XT.append(calc_MI(images, foutputs[i][total_layers-1], 30))        
        #i_XT.append(calc_MI(foutputs[i][total_layers-1], images_averaged, 31))
        #i_XT.append(calc_MI(images_index, foutputs[i][total_layers-1], 31))
        #i_XT.append(calc_MI(l_train[image_start:image_end], foutputs[i][total_layers-1], 31))
        #i_XT.append(calc_MI(images_averaged, foutputs[i][total_layers-1], 31))
        i_XT.append(calc_MI(images_averaged.flatten(), foutputs[i][total_layers-1].flatten(), 31))
        
    #%%    
    # fig, ax = plt.subplots()      # Yusuf's original code
    # # make a plot
    # ax.plot(range(len(all_epoch_model_list)), i_YT, color="red", marker=".")
    # # set x-axis label
    # # set y-axis label
    # ax.set_ylabel("I(Y;T)") #,color="red",fontsize=12)
    # ax.set_xlabel("Epochs")
    # plt.savefig("Figures/Figure4c")
    # plt.title("Mutual Information")
    # plt.show()
    
    # Save mutual information to a CSV file
    mInfo_dict = {'I_XT' : i_XT, 'I_YT' : i_YT}
    mInfo_df = pd.DataFrame(mInfo_dict)
    mInfo_df.to_csv(path_figures + 'mutualInfo_epoch.csv')
    
    # Figure 3: Plot mutual information I(X;T) between input X and the final trained output T 
    # here we only plot the first 60 epochs' output - train label mutual info
    fig3 = plt.figure(dpi=600)
    #plt.ylim(5, 14);
    plt.plot(range(epochs), i_XT[0:epochs], color="blue", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Mutual Information I(X;T)")
    plt.tight_layout();
    plt.show()
    fig3.savefig(path_figures + "Figure_3" + figSuffix + "_IXT.png", dpi=600)
    
    # Figure 4: Plot mutual information I(Y;T) between training label Y and final trained output T 
    # here we only plot the first 60 epochs' output - train label mutual info
    fig4 = plt.figure(dpi=600)
    #plt.ylim(3, 6.5);
    plt.plot(range(epochs), i_YT[0:epochs], color="blue", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Mutual Information I(Y;T)")
    plt.tight_layout();
    plt.show()
    fig4.savefig(path_figures + "Figure_4" + figSuffix + "_IYT.png", dpi=600)
    
    #%%
    # Plotting Mutual Information Along the Layers (Figure 6)
    """ 
    These quantities calculated by assumption of I(X;T)=H(T) where T is a deterministic function of X.
    """
    images=x_train[image_start:image_end]
    
    epoch_list = [1, 2, 5, 10, 20, 30, 40, 50, 60, 100]
    epoch_output=[]
    start_time = time.time();
    for i, epoch_index in enumerate(epoch_list): # i:0-9
        epoch_model = all_epoch_model_list[epoch_index - 1]
        layer_outputs=[layer.output for layer in epoch_model.layers[:]]
        activation_model=models.Model(inputs=epoch_model.input, outputs=layer_outputs)
        epoch_output.append(activation_model.predict(images))
    print('Testing {} images for every epoch 1, 2, 5, 10, 20, 30, 40, 50, 60, 100 took {} seconds'.format(image_end-image_start, time.time()-start_time))
    
    epoch_entropy=[]
    start_time = time.time();
    for i, model in enumerate(epoch_output):# i:0-9
        layer_entropy=[]
        for j in range(len(epoch_output[0])): #j:0-(total_layer-1), if total layer =6, j=0-5
            #layerentropi.append(entropy(outputs[i][j].flatten()))
            layer_entropy.append(entropy(epoch_output[i][j].flatten()))
        # append every layers' entropy to the epoch entropy, so that each epoch_entropy[i] has all layers' entropy[i][j], j=0 to total_layers - 1
        epoch_entropy.append(layer_entropy)  
    print('Calculating mutual information between layers of an epoch took {} seconds'.format(time.time()-start_time))
    
    # Save the mutual information of each layer for 10 epochs to a CSV file
    layerInfo_df = pd.DataFrame(epoch_entropy)
    layerInfo_df.to_csv(path_figures + 'entropy_' + str(cnn_number) + 'layer_10Epochs.csv')
    
    # Define the x-coordinator for the layer-mutual-info in y-axis. If we have 6 layers, the max. is 6
    x_coordinator = []
    for i in range(total_layers):
        x_coordinator.append(i+1) # x-axis coordinator, 1, 2, ..., total_layers
        
    fig6 = plt.figure(dpi=600)
    #plt.ylim(0, 14)
    for i in range(0, 10):
        plt.plot(x_coordinator, epoch_entropy[i], marker='.', markersize=10)
        #plt.scatter(x_coordinator, epoch_entropy[i], label=str(epoch_list[i]))
    
    xticks = []    
    xtick_number = total_layers + 2
    for i in range(xtick_number):
        xticks.append(i+1)
    
    #plt.title("Mutual Information- I(X;T)")
    plt.xlabel("Layers")
    plt.ylabel("Mutual Information: I(X:T)")
    ax = plt.gca()
    #ax.set_xticks([1,2,3,4,5,6,7,8,9,10]) # Anyway, we have 10 xticks
    ax.set_xticks(xticks)
    
    # fill the xtick labels according to the number of CNN layers
    xtick_label_list = []
    #for i in range(10):
    for i in range(xtick_number):
        xtick_label_list.append("")
    for i in range(cnn_number):
        xtick_label_list[i] = "Conv" + str(i+1)
        print(i, xtick_label_list[i])
    xtick_label_list[i+1] = "MaxPool"; xtick_label_list[i+2] = "Flatten"; 
    xtick_label_list[i+3] = "Dense";   xtick_label_list[i+4] = "Output"
    
    #ax.set_xticklabels(["Conv1","Conv2","Conv3","Conv4","MaxPool","Flatten","Dense","Output", ""],rotation = 90)
    ax.set_xticklabels(xtick_label_list, rotation=90)
    plt.legend(epoch_list, title='Epoch', loc='lower right')
    plt.tight_layout();
    plt.show()
    fig6.savefig(path_figures + "Figure_6" + figSuffix + "_layerMutualInfo.png")

print('\n\nTotal running time: {} seconds'.format(time.time() - start_time_all))
