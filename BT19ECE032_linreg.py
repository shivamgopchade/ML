import pandas as pd
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot

def BT19ECE032_dataset_div_shuffle(df,train_ratio,test_ratio):
    #load csv file
    if '.csv' in df:
        df=pd.read_csv(df)
    #load xlsv file
    elif '.xls' in df:
        df=pd.read_xml(df)
    #load mat file
    else:
        file=loadmat(df)
        del file['__header__']
        del file['__version__']
        del file['__globals__']
        for i in file:
            key=i
        file=file[key]
        dic={}
        #print(file['hwydata'][0][0].shape)
        #print(file['hwyheaders'][0][0][0])
        for i in range(file['hwyheaders'][0][0][0].shape[0]):
            #print(file['hwyheaders'][0][0][0][i][0])

            dic[file['hwyheaders'][0][0][0][i][0]]=file['hwydata'][0][0][:,i]
        df=pd.DataFrame(dic)
        #print(df)
    #shuffle the data
    shuffled=df.sample(frac=1)
    len1=round(shuffled.shape[0]*train_ratio)
    train_data=shuffled.iloc[1:len1]
    test_data=shuffled[len1:]
    #print(train_data)
    #print(test_data)
    return [train_data,test_data]

def accuracy(y_actual,y_pred):
    return 1-np.sum(np.square(y_actual-y_pred))/np.sum(np.square(y_actual-np.mean(y_actual)))

def mse(y_actual,y_pred):
    return np.sum(np.square(y_actual-y_pred))/(2*y_pred.shape[0])

def load_data(df):
    [train, test] = BT19ECE032_dataset_div_shuffle(df, 0.5, 0.5)

    #splitting training/testing features and outcomes
    X_train = np.array([train['Licensed drivers (thousands)'],
                        train['Fatalities per 100K licensed drivers']]).reshape(-1, 2)
    Y_train = np.array(train['Fatalities involving high blood alcohol']).reshape(-1, 1)
    X_test = np.array(
        [test['Licensed drivers (thousands)'],test['Fatalities per 100K licensed drivers']]).reshape(-1, 2)
    Y_test = np.array(test['Fatalities involving high blood alcohol']).reshape(-1, 1)
    # Fatalities per 100M vehicle-miles traveled
    # adding bias in trainging set
    ones = np.ones(shape=(X_train.shape[0], 1))
    X_train = np.append(X_train, ones, axis=1)

    # adding bias in testing set
    ones = np.ones(shape=(X_test.shape[0], 1))
    X_test = np.append(X_test, ones, axis=1)

    return [X_train,Y_train,X_test,Y_test]

def linreg_pseudo_inv(df):
    #getting shuffled data
    [X_train,Y_train,X_test,Y_test]=load_data(df)

    #performing linear regression
    a=np.dot(X_train.T,X_train)
    b=np.linalg.inv(a)
    #getting weights
    w=np.dot(np.dot(b,X_train.T),Y_train)
    #predicting on test set
    y_pred=np.dot(X_test,w)

    #testing accuracy of tained model with testing data
    print("MSE:",mse(Y_test,y_pred))
    #print("Accuracy:",accuracy(Y_test,y_pred))
    return y_pred

def gradient_descent(df):

    [X_train,Y_train,X_test,Y_test]=load_data(df)

    #print(Y_train)
    m = X_train.shape[0]
    theta = np.random.rand(X_train.shape[1]).reshape((X_train.shape[1],1))
    #print(X_train,theta)
    # applying gradient descent
    a = 0.000000003
    cost_list = []
    for i in range(50):
        y_pred=np.dot(X_train, theta)
        y_hat=y_pred-Y_train
        #print(y_pred[0])
        cost_val = (1 /2*m) * (np.sum(np.dot(y_hat.T,y_hat)))
        cost_list.append(cost_val)

        #print(theta)
        dw=np.dot(X_train.T,y_hat)/(m)
        theta = theta - a*dw

        #print(theta)
    # Predicting our Hypothesis
    b = theta
    Y_pred = X_test.dot(b)
    print("MSE:",mse(Y_test,Y_pred))
    pyplot.plot(cost_list)
    pyplot.xlabel("no of iterations")
    pyplot.ylabel("cost value")
    pyplot.show()

linreg_pseudo_inv('./dataset/Matlab_accidents.mat')