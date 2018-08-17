import numpy as np
import textgrid as tg

def str2int(_str):
    if(_str == ''):
        _str = 0
    else:
        _str = int(_str)
    return _str

def read_textgrid(filename,length):
    wav_textgrid = tg.TextGrid()
    wav_textgrid.read(filename)
    wav_tier = wav_textgrid.getFirst('neidatongxue')
    results = []

    j = 0
    for i in range(length):
        if i * 0.01 >= wav_tier[j].minTime and i*0.01 <=wav_tier[j].maxTime:
            results.append(str2int(wav_tier[j].mark))
        else:
            if(j < len(wav_tier) - 1):
                j = j + 1
                results.append(str2int(wav_tier[j].mark))
            else:
                results.append(0)
    return np.array(results)

