import pandas as pd
from scipy.io import loadmat

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
        for i in range(file.shape[0]):
            dic['label'+str(i+1)]=file[i]
        df=pd.DataFrame(dic,index=[i for i in range(file.shape[1])])

    #shuffle the data
    shuffled=df.sample(frac=1)
    len1=round(shuffled.shape[0]*train_ratio)
    train_data=shuffled.iloc[1:len1]
    test_data=shuffled[len1:]
    #print(train_data)
    #print(test_data)
    return [train_data,test_data]


#BT19ECE032_dataset_div_shuffle("./dataset/cars_test_annos.mat",0.7,0.3)