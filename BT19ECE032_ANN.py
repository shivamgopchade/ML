import pandas as pd
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt

def BT19ECE032_dataset_div_shuffle(df,tr=0.8,vr=0.1,ter=0.1):
        file=loadmat(df)
        del file['__header__']
        del file['__version__']
        del file['__globals__']

        label = file['t'][0]
        labels=np.reshape(label,(1,-1))
        d=np.append(file['x'],labels,axis=0)
        dic={}
        for i in range(d.shape[0]):
            if(i==d.shape[0]-1):
                dic['y']=d[i,:]
            else:
                dic['label'+str(i)]=d[i,:]

        df=pd.DataFrame(dic)
        shuffled=df.sample(frac=1)
        len_tr=round(shuffled.shape[0]*tr)
        len_val=round(shuffled.shape[0]*vr)
        len_ter=round(shuffled.shape[0]*ter)
        #print(shuffled.shape[1])
        X_train=shuffled.iloc[0:len_tr,:shuffled.shape[1]-1]
        #print(shuffled.shape[1])

        Y_train=shuffled.iloc[0:len_tr,shuffled.shape[1]-1]

        X_test = shuffled.iloc[len_tr:len_tr+len_val, :shuffled.shape[1]-1]

        Y_test = shuffled.iloc[len_tr:len_tr+len_val, shuffled.shape[1]-1]

        X_val = shuffled.iloc[len_tr+len_val:, :shuffled.shape[1]-1]
        Y_val = shuffled.iloc[len_tr+len_val:, shuffled.shape[1]-1]
        return [np.array(X_train).T,np.array(Y_train).T,np.array(X_test).T,np.array(Y_test).T,np.array(X_val).T,np.array(Y_val).T]

def initialize_params(dims):
    params={}
    np.random.seed(3)
    L=len(dims)
    for i in range(L-1):
        params['w'+str(i)]=np.random.randn(dims[i+1],dims[i])*0.01
        params['b'+str(i)]=np.zeros(shape=(dims[i+1],1))
    return params

def step_up(x,w,b,activation):
    z=np.dot(w,x)+b
    a=0
    error=1e-8
    if(activation=='sigmoid'):
            a=1/(1+np.exp(-z))+error
    else:
        a=np.maximum(z,float(0))
        #print("a:",a.shape)
    return a,[x,w,b,z]

def step_down(dZ,c,activation,_lambda=0.01):
    x,w,b,z=c
    m=dZ.shape[1]
    dW=np.dot(dZ,x.T)/m+(_lambda*w)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m

    if(activation=='sigmoid'):
        dZ*=(z*(1-z)).astype(float)
    else:
        dZ*=(z>float(0)).astype(float)

    dZ_prev=np.dot(w.T,dZ)

    return dZ_prev,dW,db

def forward_prop(X_train,params):
    layers=int(np.ceil(len(params)/2))
    x=X_train
    cach=[]
    for i in range(layers-1):
        #print(i)
        w=params['w'+str(i)]
        b=params['b'+str(i)]
        x,c=step_up(x,w,b,'relu')
        cach.append(c)
    w=params['w'+str(layers-1)]
    b=params['b'+str(layers-1)]
    x,c=step_up(x,w,b,'sigmoid')

    cach.append(c)
    return x,cach

def backward_prop(y_pred,Y_train,payload):
    gradients={}
    L=len(payload)
    error=1e-8
    #print("y:",y_pred)
    dZ=-np.divide(Y_train,y_pred+error)+np.divide((1-Y_train),(1-y_pred+error))
    c=payload[L-1]
    gradients['dZ'+str(L-1)],gradients['dw'+str(L)],gradients['db'+str(L)]=step_down(dZ,c,'sigmoid')

    for i in reversed(range(L-1)):
        #print(i)
        c=payload[i]
        dZ=gradients['dZ'+str(i+1)]
        gradients['dZ' + str(i)], gradients['dw' + str(i+1)], gradients['db' + str(i+1)] = step_down(dZ, c, 'relu')

    return gradients

def compute_cost(y_pred,Y_train,params,_lambda=0.01):
    m=y_pred.shape[1]
    error=1e-8
    j=np.sum(-1*np.multiply(Y_train,np.log(y_pred+error))+np.multiply(1-Y_train,np.log(1-y_pred+error)))
    sum=0
    L = int(np.ceil(len(params) / 2))
    for i in range(L):
        w=params['w'+str(i)]
        sum+=np.sum(np.square(w))
    j+=sum
    j/=m
    j=np.squeeze(j)
    return j

def accuracy(y_pred,y_actual,th):
    y_hat=(y_pred>th).astype('float64')
    #print(y_hat.shape,y_actual.shape)
    a=np.sum(y_hat[0]==y_actual)
    #print(y_pred.shape,y_actual.shape)
    return 100*a/y_pred.shape[1]

def optimize_params(params,gradients,a):
    L=int(np.ceil(len(params)/2))
    #print("L:",L)
    #print("params",params['w0'])
    for i in range(L):
        params['w'+str(i)]=params['w'+str(i)]-a*gradients['dw'+str(i+1)]
        params['b' + str(i)] = params['b' + str(i)] - a * gradients['db' + str(i+1)]
    #print("params size:",len(params))
    return params

def load_Data(path):
    df=pd.read_csv(path)
    shuffled=df.sample(frac=1)
    n=shuffled.shape[0]
    m=shuffled.shape[1]
    X_train=np.array(shuffled.iloc[:int(0.8*n),:m-1]).T
    Y_train=np.array(shuffled.iloc[:int(0.8*n),m-1]).T
    X_test = np.array(shuffled.iloc[int(0.8 * n):,:m-1]).T
    Y_test = np.array(shuffled.iloc[int(0.8 * n):,m-1]).T
    return [X_train,Y_train,X_test,Y_test]

def NN(dims,iter=100,a=0.00001):
    [X_train,Y_train,X_test,Y_test] = load_Data('diagnosis.csv')
    params=initialize_params(dims)
    #print("X_train:",X_train.shape)
    cost_train=[]
    for i in range(iter):

        # print("i",i)
        y_pred,payload=forward_prop(X_train,params)
        c1=compute_cost(y_pred,Y_train,params)
        #print("cost:",c)
        cost_train.append(c1)
        gradients=backward_prop(y_pred,Y_train,payload)

        params=optimize_params(params,gradients,a)

    return [params,X_test,Y_test]

def find_metric(y_pred,y_actual,th):
    y=(y_pred>th).astype(int)
    tp=np.sum(np.logical_and(y==y_actual,y==1))
    tn=np.sum(np.logical_and(y==y_actual, y==0))
    fp=np.sum(np.logical_and(y==1, y!=y_actual))
    fn=np.sum(np.logical_and(y==0 ,y!=y_actual))
    confusion_mat=np.array([[tp,fp],[fn,tn]])
    #print(confusion_mat)
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    return [sensitivity,specificity,confusion_mat]


def BT19ECE032_linreg(arch):
    [params, X_test, Y_test]=NN(arch,40,0.0001)

    y1,t1=forward_prop(X_test,params)
    print("test accuracy",accuracy(y1,Y_test,0.5))
    x=[]
    y=[]
    for i in [0.5,0.6,0.7,0.8,0.9,1]:
        s1,s2,mat=find_metric(y1,Y_test,i)
        x.append(s1)
        y.append(1-s2)
    s1,s2,mat=find_metric(y1,Y_test,0.5)
    print(s1)
    plt.plot(x,y)
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    print("Confusion matrix:",mat)
    return
BT19ECE032_linreg([31,20,1])