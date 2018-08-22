from feature_extractor import feature_extract
import numpy as np
from keras.models import load_model
from keras.models import Sequential
import matplotlib.pyplot as plt

wsmooth = 30
testindex = '1'

test,label = feature_extract('test' + testindex,'./test/','')

model=load_model('./models/model-test.h5')
predictions=model.predict_classes(test)

print(predictions.shape)
'''
pospredictions=np.zeros(predictions.shape)
for i in range(predictions.shape[1]):
    for j in range(predictions.shape[0]):
        hsmooth=max(0,j-wsmooth)
        for k in range(hsmooth,j):
            pospredictions[j][i]+=predictions[k][i]
        pospredictions[j][i]=pospredictions[j][i]/(j-hsmooth+1)

pre_x = range(0,pospredictions.shape[0])
posterior = []
for i in pre_x:
    list = pospredictions[i].tolist()
    max_index = list.index(max(list))
    posterior.append(max_index)

with open('./test/test' + testindex + '.txt','w') as log:
    log.write(str(posterior))

plt.plot(pre_x,posterior,'b')
plt.xlabel('frame')
plt.ylabel('category')
plt.savefig('./test/test' + str(testindex) + '.png')
plt.show()
'''

with open('./test/test' + testindex + '.txt','w') as log:
    log.write(str(predictions))
    print(predictions)

pre_x = range(predictions.shape[0])
plt.plot(pre_x,predictions,'b')
plt.xlabel('frame')
plt.ylabel('category')
plt.savefig('./test/test' + str(testindex) + '.png')
plt.show()