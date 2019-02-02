# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:24:59 2018

@author: newstar1993
"""
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import pandas as pd
import logging
import time
from sklearn.metrics import accuracy_score
from sklearn import preprocessing




def fold_i_of_k(dataset, i, k):
    n = len(dataset)
    return dataset[n*(i-1)//k:n*i//k]
# %% train_test_split
def train_test(data,y,Id, test_size = 0.1):
    import numpy as np
    data = np.array(data)
    y = np.array(y)
    train_number = round((1-test_size )*91)
    n_col = data.shape[1]
    indices = np.random.choice(91, train_number, replace = False)
    total_index = np.array(list(range(91)))
    test_ind = np.delete(total_index,indices)
    train_x = np.zeros([1,n_col])
    train_y = np.ndarray([])
    test_x = np.zeros([1,n_col])
    test_y = np.ndarray([])
    
    for ind in indices:
        for j in range(101):
            u = data[ind+j*91,:]
            u = u.reshape(1,n_col) 
            train_x = np.append(train_x,u,axis = 0)
            train_y = np.append(train_y,y[ind+j*91])
            
    for ind in test_ind:
        for j in range(101):
            u = data[ind+j*91,:]
            u = u.reshape(1,n_col)
            test_x = np.append(test_x,u,axis = 0)
            test_y = np.append(test_y,y[ind+j*91])
   # return indices, test_ind
    train_x = np.delete(train_x, (0), axis=0)
    train_y = np.delete(train_y, (0), axis=0)
    test_x = np.delete(test_x, (0), axis=0)
    test_y = np.delete(test_y, (0), axis=0)
    
    return train_x,train_y,test_x,test_y
#%%
def cross_val_10(data,y,i,total_index,N):
    import numpy as np
    data = np.array(data)
    y = np.array(y)
    n_col = data.shape[1]
   
    
    test_ind = fold_i_of_k(total_index, i, 10)
    test_index = total_index[test_ind]
    train_ind = np.delete(total_index, test_ind)

    train_x = np.zeros([1,n_col])
    train_y = np.ndarray([])
    test_x = np.zeros([1,n_col])
    test_y = np.ndarray([])
    for ind in train_ind:
        for j in range(N):
            u = data[ind+j*91,:]
            u = u.reshape(1,n_col) 
            train_x = np.append(train_x,u,axis = 0)
            train_y = np.append(train_y,y[ind+j*91])
    for ind in test_index:
        for j in range(N):
            u = data[ind+j*91,:]
            u = u.reshape(1,n_col)
            test_x = np.append(test_x,u,axis = 0)
            test_y = np.append(test_y,y[ind+j*91])
   # return indices, test_ind
    train_x = np.delete(train_x, (0), axis=0)
    train_y = np.delete(train_y, (0), axis=0)
    test_x = np.delete(test_x, (0), axis=0)
    test_y = np.delete(test_y, (0), axis=0)
    
    return train_x,train_y,test_x,test_y,
#%%
def cross_val_10_50(data,y,i,total_index,N):
    import numpy as np
    data = np.array(data)
    y = np.array(y)
    n_col = data.shape[1]
   
    
    test_ind = fold_i_of_k(total_index, i, 10)
    test_index = total_index[test_ind]
    train_ind = np.delete(total_index, test_ind)

    train_x = np.zeros([1,n_col])
    train_y = np.ndarray([])
    test_x = np.zeros([1,n_col])
    test_y = np.ndarray([])
    for ind in train_ind:
        for j in range(N):
            u = data[j+ind*N,:]
            u = u.reshape(1,n_col) 
            train_x = np.append(train_x,u,axis = 0)
            train_y = np.append(train_y,y[j+ind*N])
    for ind in test_index:
        for j in range(N):
            u = data[j+ind*N,:]
            u = u.reshape(1,n_col)
            test_x = np.append(test_x,u,axis = 0)
            test_y = np.append(test_y,y[j+ind*N])
   # return indices, test_ind
    train_x = np.delete(train_x, (0), axis=0)
    train_y = np.delete(train_y, (0), axis=0)
    test_x = np.delete(test_x, (0), axis=0)
    test_y = np.delete(test_y, (0), axis=0)
    return train_x,train_y,test_x,test_y
#%% Transfer data from numpy to tensor in pytorch
def XnumpyToTensor(x_data_np):
    x_data_np = np.array(x_data_np, dtype=np.float32)        
    print(x_data_np.shape)
    print(type(x_data_np))

    if use_cuda:
        lgr.info ("Using the GPU")    
        X_tensor = Variable(torch.from_numpy(x_data_np).cuda())    
    else:
        lgr.info ("Using the CPU")
        X_tensor = Variable(torch.from_numpy(x_data_np)) 
    
    print(type(X_tensor.data))
    print(x_data_np.shape)
    print(type(x_data_np))    
    return X_tensor


# Convert the np arrays into the correct dimention and type

def YnumpyToTensor(y_data_np):    
    y_data_np=y_data_np.reshape((y_data_np.shape[0],1)) # Must be reshaped for PyTorch!
    print(y_data_np.shape)
    print(type(y_data_np))

    if use_cuda:
        lgr.info ("Using the GPU")            
   
        Y_tensor = Variable(torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()         
    else:
        lgr.info ("Using the CPU")        
            
        Y_tensor = Variable(torch.from_numpy(y_data_np)).type(torch.FloatTensor)         

    print(type(Y_tensor.data)) 
    print(y_data_np.shape)
    print(type(y_data_np))  
    
    return Y_tensor




# build the model in NN.
def ten_cv_nn(data_all, y, n_nodes = 128, n_aug = 1):
# NN params
    DROPOUT_PROB = 0.90
    LR = 0.005
    MOMENTUM= 0.9
    dropout = torch.nn.Dropout(p=1 - (DROPOUT_PROB))
    lgr.info(dropout)

    hiddenLayer1Size = n_nodes

    N_FEATURES=data_all.shape[1]
    linear1=torch.nn.Linear(N_FEATURES, hiddenLayer1Size, bias=True) 
    torch.nn.init.xavier_uniform(linear1.weight)

    linear6=torch.nn.Linear(hiddenLayer1Size, 1)
    torch.nn.init.xavier_uniform(linear6.weight)

    sigmoid = torch.nn.Sigmoid()
    tanh=torch.nn.Tanh()
    relu=torch.nn.LeakyReLU()
    net = torch.nn.Sequential(linear1,nn.BatchNorm1d(hiddenLayer1Size),relu,
                              linear6,sigmoid
                              )
    lgr.info(net)  
    optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=5e-3) 
    loss_func=torch.nn.BCELoss() 

    if use_cuda:
        lgr.info ("Using the GPU")    
        net.cuda()
        loss_func.cuda()

    lgr.info (optimizer)
    lgr.info (loss_func)

    start_time = time.time()    
    epochs=1500 

    n_runs = 10
    acc_train = np.zeros([n_runs,1])
    acc_test = np.zeros([n_runs,1])
    total_index = np.random.permutation(91)
    for i in range(n_runs):

        #X_train,y_train,X_test, y_test = train_test(data_all, y, Id, test_size=0.1)

        X_train,y_train,X_test, y_test = cross_val_10(data_all, y,  i+1,total_index, n_aug)
        print (X_train.shape)
        print (y_train.shape)
        print (X_test.shape)
        print (y_test.shape)


        all_losses = []

        X_tensor_train= XnumpyToTensor(X_train)
        Y_tensor_train= YnumpyToTensor(y_train)

        print(type(X_tensor_train.data), type(Y_tensor_train.data)) 

        for step in range(epochs):    
            out = net(X_tensor_train)                 # input x and predict based on x
            cost = loss_func(out, Y_tensor_train)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

            optimizer.zero_grad()   # clear gradients for next train
            cost.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
   

        if step % 5 == 0:        
            loss = cost.data
            all_losses.append(loss)

            print(step, cost.data.cpu().numpy())
            
        prediction = (net(X_tensor_train).data).float() # probabilities         
#         prediction = (net(X_tensor).data > 0.5).float() # zero or one
#         print ("Pred:" + str (prediction)) # Pred:Variable containing: 0 or 1
#         pred_y = prediction.data.numpy().squeeze()            
        pred_y = prediction.cpu().numpy().squeeze()
        pred_y=(pred_y > 0.5)
        target_y = Y_tensor_train.cpu().data.numpy()

        acc_train[i] = (accuracy_score(target_y, pred_y))
        print ('training acc= %3.5f'%acc_train[i]  )     

        end_time = time.time()
        print ('{} {:6.3f} seconds'.format('GPU:', end_time-start_time))

#%matplotlib inline
#   import matplotlib.pyplot as plt


#%% Test on test set
        net.eval()

    # Validation data
        print (X_test.shape)
        print (y_test.shape)

        X_tensor_val= XnumpyToTensor(X_test)
        Y_tensor_val= YnumpyToTensor(y_test)


        print(type(X_tensor_val.data), type(Y_tensor_val.data)) # should be 'torch.cuda.FloatTensor'

        predicted_val = (net(X_tensor_val).data).float() # probabilities 
        # predicted_val = (net(X_tensor_val).data > 0.5).float() # zero or one
        pred_y = predicted_val.cpu().numpy()
        pred_y=(pred_y > 0.5)
        target_y = Y_tensor_val.cpu().data.numpy()                

        print (type(pred_y))
        print (type(target_y))

        acc_test[i] = ((pred_y == target_y).mean())
        print ('\n')
        print ('test acc= %2.5f'%acc_test[i] ) 
        
    return np.mean(acc_test)
#%%  xgboost training
def ten_cv_xgboost(data_all, y, n_est=1000, Gamma = 0 , Learning_rate = 0.05, Max_depth = 1, Min_child_weight =5, Subsample = 0.8, Colsample_bytree=0.8, Num_round = 2000, n_aug = 1):
# NN params

    from xgboost.sklearn import XGBClassifier 
    from sklearn.metrics import accuracy_score
    import numpy as np 
    n_runs = 10
    acc_train = np.zeros([n_runs,1])
    acc_test = np.zeros([n_runs,1])
    total_index = np.random.permutation(91)
    test_acc = np.zeros(n_runs)
    for i in range(n_runs):

        #X_train,y_train,X_test, y_test = train_test(data_all, y, Id, test_size=0.1)

        X_train,y_train,X_test, y_test = cross_val_10(data_all, y,  i+1,total_index,n_aug)
        print (X_train.shape)
        print (y_train.shape)
        print (X_test.shape)
        print (y_test.shape)


        from sklearn.metrics import mean_squared_error

        rf_reg = XGBClassifier(n_estimators=n_est, gamma =Gamma, learning_rate = Learning_rate,
                                   nthread=4,  max_depth =  Max_depth,
                                  min_child_weight =Min_child_weight, subsample = Subsample,colsample_bytree=Colsample_bytree, 
                                   num_round = Num_round)
        rf_reg.fit(X_train,y_train)
        preds_train = rf_reg.predict(X_train)
       # train_acc[i] = accuracy_score(y_train, preds_train)
        preds_rf= rf_reg.predict(X_test)
        acc_train[i] = (accuracy_score(y_train, preds_train))
        print ('training acc= %3.5f'%acc_train[i]  )     


        test_acc[i] = accuracy_score(y_test, preds_rf)
    return np.mean(test_acc)


#%% train different models on different datasets
def train_model(aug = False, flt = True, feature_step_5 = True):
     
    if feature_step_5 == False:
        
        data=  pd.read_csv('data_preprocessing/training_data_aug_50.csv', header=None, prefix = 'V')
        data_original = data
        data.head()
        Id =data_original.pop('V0')
        y = data_original.pop('V54')
        
        
        y=(y > 1.5).astype(int)
        y = np.array(y)
        data = np.array(data_original)

        # take the augmentation, if augmented then ignore this part
        
        #ind_1 = range(99, 151, 5)
        #data2 = data[:,list(ind_1)]
        #data1 = data[:,1434:1436]
        #data1 = data[:,99:349]
        data_all = data
        #data_all = data_original
        test_acc_nn = np.zeros(40)
        
        test_acc_xg = np.zeros(40)
        
        for i in range(40):
            test_acc_nn[i]  = ten_cv_nn(data_all, y)
            
            test_acc_xg[i]  = ten_cv_xgboost(data_all,y,n_est=1000, Gamma = 20, Learning_rate = 0.05,
                                          Max_depth = 2,
                                         Min_child_weight =1, Subsample = 0.8
                                         ,Colsample_bytree=0.8, Num_round = 100
                                         , n_aug = 1)
        
            
        print('The average acc for NN over 40 exp: %3.4f' %(np.mean(test_acc_nn)))
        print('The average acc for Xgboost over 40 exp: %3.4f' %(np.mean(test_acc_xg)))
        
        
    if aug == False and flt == False and feature_step_5 == True:
        data = pd.read_csv('data_preprocessing/original_data.csv', header=None, prefix = 'V')
        data_original = data
        Id = data_original.pop('V1437')
        y = data_original.pop('V0')
        y=(y > 1.5).astype(int)
        y = np.array(y)
        data = np.array(data_original)

        # take the augmentation, if augmented then ignore this part
        
        ind_1 = range(99, 151, 5)
        data2 = data[:,list(ind_1)]
        data1 = data[:,1434:1436]
 #       data1 = data[:,99:349]
        data_all = np.hstack([data1,data2])
        #data_all = data_original
        test_acc_nn = np.zeros(40)
        test_acc_xg = np.zeros(40)
        
        for i in range(40):
            test_acc_nn[i]  = ten_cv_nn(data_all, y )
            
            test_acc_xg[i]  = ten_cv_xgboost(data_all,y,n_est=1000, Gamma = 20, Learning_rate = 0.05,
                                          Max_depth = 2,
                                         Min_child_weight =1, Subsample = 0.8
                                         ,Colsample_bytree=0.8, Num_round = 100
                                         , n_aug = 1)
        
            
        print('The average acc for NN over 40 exp: %3.4f' %(np.mean(test_acc_nn)))
        print('The average acc for Xgboost over 40 exp: %3.4f' %(np.mean(test_acc_xg)))
        
        
    if aug == False and flt == True and feature_step_5 == False:
        
        data = pd.read_csv('data_preprocessing/filtered_data_boxplot.csv', header=None, prefix = 'V')
        data_original = data
        Id = data_original.pop('V1436')
        y = data_original.pop('V1437')
        y=(y > 1.5).astype(int)
        y = np.array(y)
        data = np.array(data_original)

        # take the augmentation, if augmented then ignore this part
        
        ind_1 = range(99, 151, 5)
        
        data2 = data[:,list(ind_1)]
        data1 = data[:,1434:1436]
        
        data_all = np.hstack([data1,data2])
        #data_all = data_original
        test_acc_nn = np.zeros(40)
        test_acc_xg = np.zeros(40)

        for i in range(40):
            
            test_acc_nn[i]  = ten_cv_nn(data_all,y)
            
            test_acc_xg[i]  = ten_cv_xgboost(data_all,y,n_est=1000, Gamma = 20, Learning_rate = 0.05,
                                          Max_depth = 2,
                                         Min_child_weight =1, Subsample = 0.8
                                         ,Colsample_bytree=0.8, Num_round = 100
                                         , n_aug = 1)
        
            
        print('The average acc for NN over 40 exp: %3.4f' %(np.mean(test_acc_nn)))
        print('The average acc for Xgboost over 40 exp: %3.4f' %(np.mean(test_acc_xg)))
        
    if aug == True and flt == False and feature_step_5 == False:
        
        data = pd.read_csv('data_preprocessing/training_data_mean_fcz_Gaussian.csv', header=None, prefix = 'V')
        
        data_original = data
        Id = data_original.pop('V0')
        y = data_original.pop('V1437')
        y=(y > 1.5).astype(int)
        y = np.array(y)
        data = np.array(data_original)

        # take the augmentation, if augmented then ignore this part
        
        ind_1 = range(99, 151, 5)
        
        data2 = data[:,list(ind_1)]
        data1 = data[:,1434:1436]
        
        data_all = np.hstack([data1,data2])
        #data_all = data_original
        test_acc_nn = np.zeros(40)
        test_acc_xg = np.zeros(40)

        for i in range(40):
            test_acc_nn[i] = ten_cv_nn(data_all, y, n_aug = 100)
            
            test_acc_xg[i]  = ten_cv_xgboost(data_all,y,n_est=1000, Gamma = 20, Learning_rate = 0.05,
                                          Max_depth = 2,
                                         Min_child_weight =1, Subsample = 0.8
                                         ,Colsample_bytree=0.8, Num_round = 100
                                         , n_aug = 100)
        
            
        print('The average acc for NN over 40 exp: %3.4f' %(np.mean(test_acc_nn)))
        print('The average acc for Xgboost over 40 exp: %3.4f' %(np.mean(test_acc_xg)))
        
    if aug == True and flt == True and feature_step_5 == False:
        
        data = pd.read_csv('data_preprocessing/filtered_data_boxplot_fcz_Gaussian.csv', header=None, prefix = 'V')
        
        data_original = data
        
        Id = data_original.pop('V0')
        y = data_original.pop('V1437')
        y=(y > 1.5).astype(int)
        y = np.array(y)
        data = np.array(data_original)

        # take the augmentation, if augmented then ignore this part
        
        ind_1 = range(99, 151, 5)
        
        data2 = data[:,list(ind_1)]
        data1 = data[:,1434:1436]
        
        data_all = np.hstack([data1,data2])
        #data_all = data_original
        test_acc_nn = np.zeros(40)
        test_acc_xg = np.zeros(40)

        for i in range(40):
            test_acc_nn[i] = ten_cv_nn(data_all, y,  n_aug = 100)
            
            test_acc_xg[i]  = ten_cv_xgboost(data_all,y,n_est=1000, Gamma = 20, Learning_rate = 0.05,
                                          Max_depth = 2,
                                         Min_child_weight =1, Subsample = 0.8
                                         ,Colsample_bytree=0.8, Num_round = 100
                                         , n_aug = 100)
        
            
        print('The average acc for NN over 40 exp: %3.4f' %(np.mean(test_acc_nn)))
        print('The average acc for Xgboost over 40 exp: %3.4f' %(np.mean(test_acc_xg)))
    
    
    return test_acc_nn, test_acc_xg

#% major body of the data test


    # change the parameters to train on different datasets
handler=logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
#%% use GPU
use_cuda = torch.cuda.is_available()
# use_cuda = False

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

lgr.info("USE CUDA=" + str (use_cuda))
   
train_model(aug = False, flt = False, feature_step_5 = True)