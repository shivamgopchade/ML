from sklearn.decomposition import PCA
from scipy.io import loadmat
import numpy as np
import pandas as pd

def load_data(path):
    if '.csv' in path:
        df=pd.read_csv(path)
    #load xlsv file
    elif '.xls' in path:
        df=pd.read_xml(path)
    else:
        file = loadmat(path)
        del file['__header__']
        del file['__version__']
        del file['__globals__']
        for i in file:
            key = i
        file = file[key]
        #print(file)
        dic = {}
        for i in range(file.shape[0]):
            dic['label' + str(i + 1)] = file[i]
        df = pd.DataFrame(dic, index=[i for i in range(file.shape[1])])
    #print(df)
    shuffled = df.sample(frac=1)
    return np.array(shuffled)

# Principal Component Analysis
def PCA_(data_path):
    data=load_data(data_path)
    print("Original data shape:",data.shape)

    #standarize data
    data_meaned=data-np.mean(data,axis=0)

    #cov matrix
    cov=np.cov(data_meaned,rowvar=False)

    #calculate eign value and vector matrix
    eign_value,eign_vector=np.linalg.eig(cov)

    #find max eign value and corresponding eign vector
    sorted_i=np.argsort(eign_value)
    sorted_eignVal=eign_value[sorted_i]
    sorted_eignVec=eign_vector[sorted_i,:]

    #consider number of componenets from 100 to 20
    n_componenets=20
    eign_vector_subset=sorted_eignVec[:n_componenets,:]

    #transform data along principle components
    data_transformed=np.dot(data_meaned,eign_vector_subset.T)

    print("Transformed data shape:",data_transformed.shape)
    return [data_transformed,sorted_eignVal[:n_componenets],eign_vector_subset,data_meaned]

def PCA_reverse(componenets,eigen_vector_subset):
    reconstructed_data=np.dot(componenets,eigen_vector_subset)
    print("reconstructed data shape:",reconstructed_data.shape)
    return reconstructed_data

def MSE(original_d,reconstructed_d):
    return np.sum(np.sum(np.square(original_d-reconstructed_d)))/original_d.size

[componenets,eigen_vals,eigen_vector_subset,original_data]=PCA_('./dataset/Matlab_cancer (1).mat')
reconstructed_data=PCA_reverse(componenets,eigen_vector_subset)
error=MSE(original_data,reconstructed_data)
print("Error:",error)