# Naive LSTM to learn three-char window to one-char mapping
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import tushare as ts
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sbn
from tqdm import tqdm

cache_path='stock_cache/'

def getData():
    stock_code='399300'
    if not os.path.exists(cache_path+'k_data.pkl'):        
        k_data=ts.get_k_data(code=stock_code,start='2012-01-01',end='2018-02-15',retry_count=5)
        k_data=k_data.sort_index()
        #print(k_data)
        #k_data['date']=k_data.index
        k_rNo=pd.DataFrame(list(range(0,k_data.shape[0])),columns=['RowNo'])
        k_data_l=pd.concat([k_data.ix[:,['date','close','code']],k_rNo],axis=1)
        k_data_n=k_data.ix[:,['close','code']].copy()
        k_data_n.columns=['close_n','code_n']
        k_rNo_n=pd.DataFrame(list(range(-1,k_data.shape[0]-1)),columns=['RowNo'])
        k_data_n=pd.concat([k_data_n,k_rNo_n],axis=1)
        k_data=pd.merge(k_data_l,k_data_n,on='RowNo')
        k_data['change'] = k_data.apply(lambda row: getPriceCode(row['close'], row['close_n']), axis=1)
        k_data=k_data.ix[:,['date','close','code','close_n','change']]        
        with open(cache_path+'k_data.pkl', 'wb') as data_k:
            pickle.dump((k_data), data_k) 
    else:
        with open(cache_path+'k_data.pkl', 'rb') as data_k:
            k_data=pickle.load(data_k) 
    sbn.countplot(x=u'change',data=k_data)
    plt.show()    
    return k_data

def getPriceCode(price_last,price_this):
    if price_last==0:
        return 0
    p=price_this/price_last-1
    p_code=0
    if p<=-0.03:
        p_code=-2
    elif p<=-0.01:
        p_code=-1
    #elif p<=0:
        #p_code=-1        
    elif p<=0.01:
        p_code=0
    elif p<=0.03:
        p_code=1
    else:
        p_code=2
    return p_code

def procData():
    #myfont=matplotlib.font_manager.FontProperties(fname="/System/Library/Fonts/msyh.ttf")
    #sbn.set(font=myfont.get_name())
    k_data=getData()
    seq_len=30
    dataX = []
    dataY = []    
    for i in tqdm(range(0,k_data.shape[0]-seq_len-1)):
        one_seq=[]
        for j in range(0,seq_len):
            one_seq.append(k_data.ix[i+j,'change'])
        dataX.append(one_seq)
        dataY.append(k_data.ix[i+seq_len,'change'])
    return dataX,dataY

    
def train():
    # fix random seed for reproducibility
    seq_len=30
    np.random.seed(7)
    dataX,dataY=procData()
    X = np.reshape(dataX, (len(dataX), seq_len, 1))
    # normalize
    X = X / float(3)
    # one hot encode the output variable
    X_Test=X[len(dataX)-74:len(dataX)-1]
    print(X_Test)
    y = np_utils.to_categorical(dataY)
    # create and fit the model
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    hist=model.fit(X, y, epochs=5, batch_size=1, verbose=2,validation_split=0.05)
    print(hist.history['val_acc'])
    # summarize performance of the model
    Y_test=model.predict(X_Test)
    scores = model.evaluate(X, y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1]*100))
    print(Y_test)
    print(y[-73])
    
    
if __name__ == "__main__":
    train()

