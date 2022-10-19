import numpy.random
from scipy.io import loadmat
import numpy as np
from sklearn.svm import SVC,SVR
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import precision_score,recall_score,roc_curve,auc,accuracy_score,confusion_matrix
from matplotlib import pyplot as plt
def load_mat(path):
    file=loadmat(path)
    x = file['meas']
    y = file['species']
    Y=[]
    for i in y:
        if (i[0][0]=='versicolor'):
            Y.append(0)
        elif i[0][0]=='virginica':
            Y.append(1)
        else:
            Y.append(2)
    Y=np.array(Y).reshape((-1,1))
    d=np.append(x,Y,axis=1)
    numpy.random.shuffle(d)
    return d

def data_split(path):
    d=load_mat(path)
    n=d.shape[0]
    m=d.shape[1]
    X_train=d[:round(0.8*n),:m-1]
    Y_train=d[:round(0.8*n),m-1]
    X_test=d[round(0.8*n):round(0.9*n),:m-1]
    Y_test=d[round(0.8*n):round(0.9*n),m-1]
    X_val=d[round(0.9*n):,:m-1]
    Y_val=d[round(0.9*n):,m-1]
    return [X_train,Y_train,X_test,Y_test,X_val,Y_val]

def grid_search(_kernel,k,l,C,E,G):
    svc=svm.SVR()
    #C=[0.5,2.5,5.5,100,250]
    #E=[0.00001,0.00005,0.0001,0.0003,0.0009,0.001,0.005,0.02,1,3,7]
    #G=[0.4,0.7,0.01,1.2,3,7]
    grid=GridSearchCV(estimator=SVR(kernel=_kernel),param_grid={'C':C,'epsilon':E,'gamma':G},
                      cv=5,scoring='neg_mean_squared_error',verbose=1,n_jobs=-1)
    grid.fit(k,l)
    return [grid.best_score_,grid.best_params_]

#svm function for sigmoid,poly and rbf kernels
def BT19ECE032_svm(path,_gamma,_kernel='rbf'):
    ##Splitting dataset
    [X_train, Y_train, X_test, Y_test, X_val, Y_val]=data_split(path)

    ##Fitting dataset over svm model
    if(_kernel!='linear'):
        model=SVC(kernel=_kernel,gamma=_gamma)
    else:
        model=SVC(kernel=_kernel)
    model.fit(X_train,Y_train)

    #obtaining predicted output over test dataset
    y_pred=model.predict(X_test)

    #grid search to get optimal params for min error
    [bs,params]=grid_search(_kernel,X_train,Y_train,[0.5,2.5,5.5,100,250],[0.00001,0.00005,0.0001,0.0003,0.0009,0.001,0.005,0.02,1,3,7],[0.4,0.7,0.01,1.2,3,7])
    print(params)
    sens=precision_score(Y_test,y_pred,average=None)
    spec=recall_score(Y_test,y_pred,average=None)
    print("Accuracy:",accuracy_score(Y_test,y_pred))
    print("Sensitivity:",sens)
    print("Specificity:",spec)
    print("Confusion_matrix",confusion_matrix(Y_test,y_pred))
    plt.plot(1-spec,sens)
    plt.xlabel('1-specificity')
    plt.ylabel('sensitivity')
    plt.title('ROC curve')
    plt.show()
    return y_pred

BT19ECE032_svm('./dataset/fisheriris_matlab.mat',0.5)