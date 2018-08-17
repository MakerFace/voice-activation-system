import numpy as np
from feature_extractor import feature_extract
from file_operation import get_all_filenames

#prepare data
def preparing_data(wav_path,tg_path):
    files = get_all_filenames(wav_path)
    
    train_datas = []
    train_labels = []
    
    for file in files:
        file = file[:-4]
        data,label = feature_extract(file,wav_path,tg_path)
        train_datas.append(data)
        train_labels.append(label)
        
    #list(array) => narray
    x_train = np.concatenate((train_datas[0],train_datas[1]),0)
    y_train = np.concatenate((train_labels[0],train_labels[1]),0)
    for i in range(2,len(train_datas)):
        x_train = np.concatenate((x_train,train_datas[i]),0)
        y_train = np.concatenate((y_train,train_labels[i]),0)
    return x_train,y_train