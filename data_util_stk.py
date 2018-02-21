# Naive LSTM to learn three-char window to one-char mapping
import numpy as np
import tushare as ts
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime

cache_path='stock_cache/'

def getData():
    stock_code='399300'
    if not os.path.exists(cache_path+'k_data.pkl'):        
        k_data=ts.get_k_data(code=stock_code,start='2012-01-01',end='2018-02-15',retry_count=5)
        k_data=k_data.sort_index()
        #print(k_data)
        #k_data['date']=k_data.index
        k_rNo=pd.DataFrame(list(range(0,k_data.shape[0])),columns=['RowNo'])
        k_data_l=pd.concat([k_data.ix[:,['date','close','volume','code']],k_rNo],axis=1)
        k_data_n=k_data.ix[:,['close','volume','code']].copy()
        k_data_n.columns=['close_n','volume_n','code_n']
        k_rNo_n=pd.DataFrame(list(range(-1,k_data.shape[0]-1)),columns=['RowNo'])
        k_data_n=pd.concat([k_data_n,k_rNo_n],axis=1)
        k_data=pd.merge(k_data_l,k_data_n,on='RowNo')
        k_data['change'] = k_data.apply(lambda row: getPriceCode(row['close'], row['close_n']), axis=1)
        #axis : {0 or ‘index’, 1 or ‘columns’}, default 0        
            #0 or ‘index’: apply function to each column
            #1 or ‘columns’: apply function to each row       
        
        k_data['v_change'] = k_data.apply(lambda row: getVolunmnCode(row['volume'], row['volume_n']), axis=1)
        k_data['day'] = k_data.apply(lambda row: getDateDay(row['date']), axis=1)
        k_data['weekday'] = k_data.apply(lambda row: getWeekDay(row['date']), axis=1)
        k_data=k_data.ix[:,['code','date','close','close_n','volume','volume_n','change','v_change','day','weekday']]   
        print(k_data)
        # 对上表的prglngth列做一个直方图        
        #sns.distplot(k_data['v_change'])
        #sns.jointplot("change", "volume_n", k_data)  
        #sns.pairplot(k_data,vars=['change','v_change','volume_n','volume'])
        fig=plt.figure()
        fig.add_subplot(121)
        sns.boxplot(x='day',y='change',data=k_data)
        fig.add_subplot(122)
        sns.boxplot(x='weekday',y='change',data=k_data)        
        plt.show()          
 
        return
        
        with open(cache_path+'k_data.pkl', 'wb') as data_k:
            pickle.dump((k_data), data_k) 
    else:
        with open(cache_path+'k_data.pkl', 'rb') as data_k:
            k_data=pickle.load(data_k) 
    sns.countplot(x=u'change',data=k_data)
    plt.show()    
    return k_data

def getPriceCode(price_last,price_this):
    if price_last==0:
        return 0
    p=price_this/price_last-1
    return p
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
def getVolunmnCode(v_last,v_this):
    if v_last==0:
        return 0
    p=v_this/v_last-1    
    return p
def getDateDay(date):
    x_date=datetime.datetime.strptime(date, '%Y-%m-%d')
    return x_date.day

def getWeekDay(date):
    x_date=datetime.datetime.strptime(date, '%Y-%m-%d')
    return x_date.weekday()

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

 
    
if __name__ == "__main__":
    getData()
    #train()

