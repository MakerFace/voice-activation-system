import os
import numpy as np
from feature_extractor import feature_extract
from file_operation import get_all_filenames

train_wav_npy_filename = 'train_wav.npy'
train_tg_npy_filename = 'train_label.npy'
test_wav_npy_filename = 'test_wav.npy'
test_tg_npy_filename = 'test_label.npy'
val_wav_npy_filename = 'val_wav.npy'
val_tg_npy_filename = 'val_label.npy'

#prepare data
def serialize_data(wav_path,tg_path,npy_path):
    files = get_all_filenames(wav_path[0])

    wav_datas = None
    wav_labels = None

    for file in files:
        print(file)
        file = file[:-4]
        data,label = feature_extract(file,wav_path[0],tg_path[0])
        if wav_datas is None:
            wav_datas = data
            wav_labels = label
        else:
            wav_datas = np.concatenate((wav_datas,data),0)
            wav_labels = np.concatenate((wav_labels,label),0)
    
    files = get_all_filenames(wav_path[1])
    for file in files:
        print(file)
        file = file[:-4]
        data,label = feature_extract(file,wav_path[1],tg_path[1])
        wav_datas = np.concatenate((wav_datas,data),0)
        wav_labels = np.concatenate((wav_labels,label),0)
    print('read files over')
    datas = np.hstack([wav_datas,wav_labels])
    np.random.shuffle(datas)
    x_train = datas[:int(len(datas)*0.6),:-5]
    y_train = datas[:int(len(datas)*0.6),-5:]

    x_test = datas[int(len(datas)*0.6):int(len(datas)*0.8),:-5]
    y_test = datas[int(len(datas)*0.6):int(len(datas)*0.8),-5:]

    x_val = datas[int(len(datas)*0.8):,:-5]
    y_val = datas[int(len(datas)*0.8):,-5:]
    
    print('write train npy')
    np.save(os.path.join(npy_path,train_wav_npy_filename),x_train)
    np.save(os.path.join(npy_path,train_tg_npy_filename),y_train)

    print('write test npy')
    np.save(os.path.join(npy_path,test_wav_npy_filename),x_test)
    np.save(os.path.join(npy_path,test_tg_npy_filename),y_test)

    print('write val npy')
    np.save(os.path.join(npy_path,val_wav_npy_filename),x_val)
    np.save(os.path.join(npy_path,val_tg_npy_filename),y_val)

    print('write over')

def load_data_from_npy(npypath):
    return np.load(npypath)

if __name__ == '__main__':
    train_wav_path = '../Audio Data/database_train_wav'
    train_tg_path = '../Audio Data/database_train_label'
    test_wav_path = '../Audio Data/database_test_wav'
    test_tg_path = '../Audio Data/database_test_label'
    npy_path = '../Audio Data/npy data'

    wav_path = [train_wav_path,test_wav_path]
    tg_path = [train_tg_path,test_tg_path]
    
    serialize_data(wav_path,tg_path,npy_path)