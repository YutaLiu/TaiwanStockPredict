# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import requests
import pandas as pd
import numpy as np
from io import StringIO
import datetime
from requests.exceptions import ConnectionError
import time





def remove_td(column):
    remove_one = column.split('<')
    remove_two = remove_one[0].split('>')
    return remove_two[1].replace(",", "")

def translate_dataFrame(response):
     # 拆解內容
    table_array = response.split('<table')
    tr_array = table_array[1].split('<tr')
    
    # 拆解td
    data = []
    column = []
    for i in range(len(tr_array)):
        td_array = tr_array[i].split('<td')
        if(len(td_array)>1):
            code = remove_td(td_array[1])
            name = remove_td(td_array[2])
            revenue  = remove_td(td_array[3])
            profitRatio = remove_td(td_array[4])
            profitMargin = remove_td(td_array[5])
            preTaxIncomeMargin = remove_td(td_array[6])
            afterTaxIncomeMargin = remove_td(td_array[7])
            if( i == 1 ):
                
                column.append(code)
                column.append(name)
                column.append(revenue)
                column.append(profitRatio)
                column.append(profitMargin)
                column.append(preTaxIncomeMargin)
                column.append(afterTaxIncomeMargin)            
            try :
                if(type(int(code)) == int):
                    #print (code)
                    data.append([code,name, revenue, profitRatio, profitMargin, preTaxIncomeMargin, afterTaxIncomeMargin])
            except ValueError:
                continue

    return pd.DataFrame(data=data, columns=column)


def financial_statement(year, season, _stype):
    time.sleep(10)
    if year >= 1000:
        year -= 1911
    print ("年 =  ",year , "季 =  " , season , "type = " , _stype)
    if _stype ==  0 :
        #'綜合損益彙總表'
        url = 'https://mops.twse.com.tw/mops/web/ajax_t163sb04'
    elif _stype == 1:
        #'資產負債彙總表'
        url = 'https://mops.twse.com.tw/mops/web/ajax_t163sb05'
    elif _stype == 2:
        #'營益分析彙總表'
        url = 'https://mops.twse.com.tw/mops/web/ajax_t163sb06'
    else:
        print('type does not match')

    r = requests.post(url, {
        'encodeURIComponent':1,
        'step':1,
        'firstin':1,
        'off':1,
        'TYPEK':'sii',
        'year':str(year),
        'season':str(season),
    })

    r.encoding = 'utf8'
    if _stype == 2 :
        dfs = translate_dataFrame(r.text)
    else :
        dfs = pd.read_html(r.text, header=0)
        dfs = pd.concat(dfs[1:], axis=0, sort=False)

    return dfs





def Caculate_n_Season(year,month,n):
    year_list = []
    season_list = []
    
    
    for sea in range(0,n):
        if month >= 10:
            season_list.append(3)
            year_list.append(year)
            month = month-3
            
        elif month >= 7:
            season_list.append(2)
            year_list.append(year)
            month = month-3            

        elif month >= 4:
            season_list.append(1)
            year_list.append(year)
            month = month-3            

        else :
            season_list.append(4)
            year = year-1
            year_list.append(year)
            month = 10
    
    return year_list,season_list

# step2. 進入目標網站,爬取盤後資訊
def get_stock(date):
    return requests.post('http://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=' + date + '&type=ALL')

# step3. 篩選出個股盤後資訊
    

def stock_to_list(date):
    str_list = []
    
    
    try : 
        r = get_stock(date)
        if r.text.split('\n') == [''] :
            return 0,0
            #date = datetime.datetime.strptime(date, '%Y%m%d') + datetime.timedelta(days=1) 
            #date = date.strftime('%Y%m%d')
            #r = get_stock(date)
        for i in r.text.split('\n'):
            if len(i.split('",')) == 17 and i[0] != '=':       
                i = i.strip(",\r\n")
                str_list.append(i)      
        return date,str_list
    except ConnectionError as e :
        print (date , "ConnectionError")
        return 0,0

def Get_History_Financial_Dataframe(get_year_list , get_season_list):
    financial_dataframe = pd.DataFrame()
    for get in range(0,len(get_year_list)):
        
        SB04_df = financial_statement(get_year_list[get], get_season_list[get], 0)
        SB04_df = SB04_df[['公司代號','公司名稱','營業費用','營業收入','營業成本','營業毛利（毛損）','營業費用','營業利益（損失）','營業外收入及支出','稅前淨利（淨損）','所得稅費用（利益）','繼續營業單位本期淨利（淨損）','本期淨利（淨損）','其他綜合損益（淨額）','本期綜合損益總額','淨利（淨損）歸屬於母公司業主','淨利（淨損）歸屬於非控制權益','綜合損益總額歸屬於母公司業主','綜合損益總額歸屬於非控制權益','基本每股盈餘（元）']]
        SB04_df = SB04_df.fillna(0)
        SB04_df = SB04_df.replace("--",0)
        SB04_df = SB04_df.rename(columns={c: c+"_prev"+str(get) for c in SB04_df.columns[2:]})
        if get > 0 :
            SB04_df = SB04_df.drop(columns = ['公司名稱'])
        #SB04_df['公司代號'] = SB04_df['公司代號'].replace(" ","").astype('category')
        #SB04_df['公司名稱'] = SB04_df['公司名稱'].replace(" ","").astype('category')
        
        
        SB05_df = financial_statement(get_year_list[get], get_season_list[get], 1)

        SB05_df = SB05_df[['公司代號','公司名稱','流動資產','非流動資產','資產總額','流動負債','非流動負債','負債總額','股本','資本公積','保留盈餘','其他權益','歸屬於母公司業主之權益合計','非控制權益','權益總額','母公司暨子公司所持有之母公司庫藏股股數（單位：股）','每股參考淨值']]
        SB05_df = SB05_df.fillna(0)
        SB05_df = SB05_df.replace("--",0)

        
        SB05_df = SB05_df.rename(columns={c: c+"_prev"+str(get) for c in SB05_df.columns[2:]})
        SB05_df = SB05_df.drop(columns = ['公司名稱'])
        #SB05_df['公司代號'] = SB05_df['公司代號'].replace(" ","").astype('category')
        #SB05_df['公司名稱'] = SB05_df['公司名稱'].replace(" ","").astype('category')
        
        SB06_df = financial_statement(get_year_list[get], get_season_list[get], 2)
        SB06_df = SB06_df.fillna(0)
        SB06_df = SB06_df.replace("--",0)        
        SB06_df = SB06_df.rename(columns={c: c+"_prev"+str(get) for c in SB06_df.columns[2:]})
        SB06_df = SB06_df.drop(columns=['營業收入'+"_prev"+str(get),'公司名稱'])
        SB06_df['公司代號'] = SB06_df['公司代號'].astype('int64')
        #SB06_df['公司名稱'] = SB06_df['公司名稱'].replace(" ","").astype('category')
        
        if get == 0 :
            financial_dataframe = pd.merge(SB04_df, SB05_df,on = ['公司代號'])
            financial_dataframe = pd.merge(financial_dataframe, SB06_df,on = ['公司代號'])
        else :
            financial_dataframe = pd.merge(financial_dataframe, SB04_df,on = ['公司代號'])
            financial_dataframe = pd.merge(financial_dataframe, SB05_df,on = ['公司代號'])
            financial_dataframe = pd.merge(financial_dataframe, SB06_df,on = ['公司代號'])

    return financial_dataframe


def Drop_percent_invalid_Data(dataframe , percent):
    j=0
    for i in dataframe.eq(0).sum(axis=1).value_counts(dropna=False , normalize = True).sort_index(ascending = False).cumsum():
        if i > percent:
            Drop_Value = dataframe.eq(0).sum(axis=1).value_counts(dropna=False , normalize = True).sort_index(ascending = False).cumsum().index[j]
            print (i," Drop Value >=",dataframe.eq(0).sum(axis=1).value_counts(dropna=False , normalize = True).sort_index(ascending = False).cumsum().index[j])
            break
        j+=1
    print (j)
    dataframe = dataframe.loc[dataframe.eq(0).sum(axis=1) < Drop_Value]
    return dataframe

Traing_Day_Source = '20200101'
"""Traing Stock Get"""
ini = True
if ini:
    Stock_dataframe_Now = pd.DataFrame()
Traing_Day = Traing_Day_Source
for i in range(0,3):
    time.sleep(20)
    for colname in Stock_dataframe_Now.columns:
        if "_Prev"+Traing_Day in colname:
            continue
    real_date,Stock_dataframe = stock_to_list(Traing_Day)
    Traing_Day = datetime.datetime.strptime(Traing_Day, '%Y%m%d') - datetime.timedelta(days=1) 
    Traing_Day = Traing_Day.strftime('%Y%m%d')  
    if real_date == 0 :
        continue
    print ("Real Get Traing Stock Day :",real_date)
    Stock_dataframe_prev = pd.read_csv(StringIO("\n".join(Stock_dataframe ))).rename(columns ={'證券代號':'公司代號'})  
    Stock_dataframe_prev = Stock_dataframe_prev.drop(['證券名稱'] , axis = 1)
    Stock_dataframe_prev = Stock_dataframe_prev.drop(['漲跌(+/-)'] , axis = 1)
    Stock_dataframe_prev = Stock_dataframe_prev.replace("--",0)
    Stock_dataframe_prev.drop( Stock_dataframe_prev[Stock_dataframe_prev['收盤價'] == '--' ].index , inplace=True)
    Stock_dataframe_prev = Stock_dataframe_prev.rename(columns={c: c+"_PrevDay"+Traing_Day for c in Stock_dataframe_prev.columns[2:]})
    Stock_dataframe_prev['公司代號'] = Stock_dataframe_prev['公司代號'].astype('category')
    

    Stock_dataframe_prev = Stock_dataframe_prev.replace("--",0)
    
    if Stock_dataframe_Now.empty:
        Traing_Day_Source = Traing_Day
        Stock_dataframe_Now = Stock_dataframe_prev
    else :
        Stock_dataframe_Now = pd.merge(Stock_dataframe_prev, Stock_dataframe_Now ,how = 'left' , on=['公司代號'])


Test_Day = datetime.datetime.strptime(Traing_Day_Source, '%Y%m%d') + datetime.timedelta(days=30) 
Test_Day = Test_Day.strftime('%Y%m%d')
real_date,Stock_dataframe = stock_to_list(Test_Day)

Stock_dataframe_Now['單月漲跌'] =  Stock_dataframe_Now['收盤價_PrevDay'+Traing_Day_Source].str.replace(",","").astype('float')  - Stock_dataframe_Now['收盤價_PrevDay'+Traing_Day].str.replace(",","").astype('float') 
Stock_dataframe_Now = Stock_dataframe_Now.drop(['收盤價_PrevDay'+Traing_Day_Source] , axis = 1)
Test_Day = datetime.datetime.strptime(Traing_Day_Source, '%Y%m%d') + datetime.timedelta(days=30) 
Test_Day = Test_Day.strftime('%Y%m%d')

real_date,Stock_dataframe = stock_to_list(Test_Day)
print ("Real Get Test Stock Day :",real_date)

Stock_dataframe_Now = pd.read_csv(StringIO("\n".join(Stock_dataframe ))).rename(columns ={'證券代號':'公司代號'}) 

#Stock_dataframe_Now = Stock_dataframe_Now.rename(columns={c: c+"_Now" for c in Stock_dataframe_Now.columns[2:]})
Stock_dataframe_Now['公司代號'] = Stock_dataframe_Now['公司代號'].astype('category')

#Stock_dataframe_Now = Stock_dataframe_Now[['公司代號','證券名稱','收盤價_Now']]
#Stock_dataframe_Now =  pd.merge(Stock_dataframe_Now, Stock_dataframe_prev,on = ['公司代號','證券名稱'])




Stock_dataframe_Now = Stock_dataframe_Now.fillna(0)
Stock_dataframe_Now[["公司代號"]] = Stock_dataframe_Now[["公司代號"]].astype('str')
str_cols = Stock_dataframe_Now.select_dtypes(['object']).columns
Stock_dataframe_Now[str_cols] = Stock_dataframe_Now[str_cols].stack().str.replace(",","").unstack()
Month = datetime.datetime.strptime(real_date, '%Y%m%d').month
Year = datetime.datetime.strptime(real_date, '%Y%m%d').year
#Step4. Get pre 3 season 財報

year_list , season_list = Caculate_n_Season(Year,Month,3)


financial_dataframe = Get_History_Financial_Dataframe(year_list , season_list )
financial_dataframe = financial_dataframe.rename(columns={"公司名稱": "證券名稱"})
financial_dataframe = financial_dataframe.drop(['證券名稱'] , axis = 1)
financial_dataframe[["公司代號"]] = financial_dataframe[["公司代號"]].astype('str')



Traing_Frame = pd.merge(Stock_dataframe_Now, financial_dataframe, on=['公司代號'], how='left')
Traing_Frame = Drop_percent_invalid_Data(Traing_Frame,0.02)




Traing_Data = Traing_Frame.drop(['單月漲跌'] , axis = 1)
Test_Data = Traing_Frame['單月漲跌']


Traing_Data = pd.get_dummies(data=Traing_Data, columns=['公司代號'])
Traing_Data = Traing_Data.fillna(0)
from sklearn.model_selection import train_test_split




from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


estimators = np.arange(200, 2000, 10)
scores = []
for n in estimators:
    X_train, X_test, y_train, y_test = train_test_split( Traing_Data, Test_Data, test_size=0.25, random_state=42)
    model = RandomForestRegressor()
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    score = model.score(X_test, y_test)
    scores.append(score)
    print ("Forest estimators :",n ,"分數 :",score)
    plt.title("Predict vs Actually")
    plt.xlabel("Predict 漲幅")
    plt.ylabel("Actually 漲幅")
    plt.scatter(ypred, y_test)    
    plt.show()
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
