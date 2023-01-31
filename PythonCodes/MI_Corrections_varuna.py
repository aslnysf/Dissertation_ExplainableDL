# Use this one to calculate MI between two images of similar size


# Mutual Information calculation between inputdata X and layerdata T using binned method
# From https://github.com/artemyk/ibsgd/blob/iclr2018/simplebinmi.py
# inputdata is X, layerdata is Y. return I(Y,X) = sum[p(x) * H(y|x)]
def calc_MI(inputdata, layerdata, num_of_bins_input, num_of_bins_output):
    bins_input = np.linspace(0, 1, num_of_bins_input, dtype='float32') 
    
    digitized_input = bins_input[np.digitize(np.squeeze(inputdata.reshape(1, -1)), bins_input) - 1].reshape(len(inputdata), -1)
    p_xs, unique_inverse_x = get_unique_probs(digitized_input)
    
    bins_output = np.linspace(0, 1, num_of_bins_output, dtype='float32') 
    
    digitized = bins_output[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bins_output) - 1].reshape(len(layerdata), -1)
    p_ts, _ = get_unique_probs( digitized )
    #print('x, p_t|x', unique_inverse_x, p_xs)
    H_LAYER = -np.sum(p_ts * np.log(p_ts))
    H_LAYER_GIVEN_INPUT = 0.
    for xval in np.unique(unique_inverse_x):
        p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
        #print('x, p_t|x, p_xs[xval]', xval, p_t_given_x, p_xs[xval])
        H_LAYER_GIVEN_INPUT += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))
    return H_LAYER - H_LAYER_GIVEN_INPUT, H_LAYER,p_xs, p_ts


# Use this one to calculate MI between images and the labels

# Mutual Information calculation between inputdata X and expected labels Y (true labels) using binned method
# From https://github.com/artemyk/ibsgd/blob/iclr2018/simplebinmi.py
# inputdata is X, labels are Y. return I(Y,X) = H(X) - H(X|Y) = H(X) - \Sigma_y p(y)*H(X|Y=y)
#Expecting reshaped inputs and labels
def calc_MI_withOneHotY(inputdata, labels, num_of_bins_input, num_of_bins_output):
    bins_input = np.linspace(0, 1, num_of_bins_input, dtype='float32') 
    digitized_input = bins_input[np.digitize(np.squeeze(inputdata.reshape(1, -1)), bins_input) - 1].reshape(len(inputdata), -1)
    
    unique_y, count = np.unique(labels, return_counts=True, axis=0)
    prob_y = count/len(labels)

    #print('x, p_t|x', unique_inverse_x, p_xs)
    H_X = entropy(digitized_input)
    
    H_X_GIVEN_Y = 0.
    
    x_reshaped = inputdata.reshape(len(digitized_input),-1)
    
    for yval in unique_y:
    #extract X for yval
        mask = []
        for y_ind in range(len(labels)):
            m = (labels[y_ind,]==yval).all()
            mask.append(m)
            #print(mask)
        mask = np.array(mask)
    
        x_extract = x_reshaped[mask,:]
        entr_xr = entropy(x_extract)
    
        H_X_GIVEN_Y += len(x_extract)/len(x_reshaped)*entr_xr
    
        print(len(x_extract)/len(x_reshaped), entr_xr, len(x_extract)/len(x_reshaped)*entr_xr)
    
    
    return H_X - H_X_GIVEN_Y, H_X, H_X_GIVEN_Y




# Mutual Information calculation between inputdata X and expected labels Y (true labels) using binned method
# From https://github.com/artemyk/ibsgd/blob/iclr2018/simplebinmi.py

def entropy_OneHotY(labels):
    unique_y, count = np.unique(labels, return_counts=True, axis=0)
    prob_y = count/len(labels)

    #print(prob_y)
    en = np.sum((-1)*prob_y*np.log2(prob_y))
    return en

# inputdata is X, labels are Y. return I(Y,X) = H(X) - H(X|Y) = H(X) - \Sigma_y p(y)*H(X|Y=y)
#Expecting reshaped inputs and labels
def calc_MI_betweenOneHot(labels_true, labels_predict):
    H_True = entropy_OneHotY(labels_true)
    
    unique_p, count = np.unique(labels_predict, return_counts=True, axis=0)
    prob_p = count/len(labels_predict)

    #H(True|predict)
    H_T_GIVEN_P = 0.
    
    for yval in unique_p:
    #extract X for yval
        mask = []
        for y_ind in range(len(labels_predict)):
            m = (labels_predict[y_ind,]==yval).all()
            mask.append(m)
            #print(mask)
        mask = np.array(mask)
    
        l_extract = labels_true[mask,:]
        entr_lr = entropy_OneHotY(l_extract)
    
        H_T_GIVEN_P += len(l_extract)/len(labels_true)*entr_lr
    
        #print(len(l_extract)/len(labels_true), entr_lr, len(l_extract)/len(labels_true)*entr_lr)
        print(len(l_extract)/len(labels_true), entr_lr)
    
    return H_True - H_T_GIVEN_P


#Run the calc_MI_betweenOneHot
test_y = y_train[1:101,:]
test_y2 = y_train[101:201,:]
test_y2.shape
calc_MI_betweenOneHot(test_y2, test_y)
