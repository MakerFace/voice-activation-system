import numpy as np

def get_past(source,delta=0):
    result = np.zeros(source.shape)
    j=0
    for i in range(delta,len(source)):
        result[i] = source[j]
        j = j + 1
    return result

def get_future(source,delta=0):
    result = np.zeros(source.shape)
    j = delta
    for i in range(len(source) - delta):
        result[i] = source[j]
        j = j + 1
    return result

def get_martix(source,npast=7,nfutrue=5):

    _source = np.array(source)

    for i in range(npast):
        past = get_past(source,i+1)
        _source = np.concatenate((past,_source),1)

    for i in range(nfutrue):
        future = get_future(source,i+1)
        _source = np.concatenate((_source,future),1)
    
    #_source = _source.reshape((source.shape[0],source.shape[1],npast+nfutrue+1))

    return _source