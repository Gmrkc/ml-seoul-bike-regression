"""
Muhammet Kara 171805036
Tuğçe Çördük 171805006
Furkan Gümrükçü 171805057
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def corr_filter(dataset):
    
    corr_new_dataset = dataset.copy()                               # Return edilecek yeni dataset
    corr_all = dataset.corr()                                       # Bütün veriler arasındaki corr hesabı  
    corr_output = corr_all.iloc[:-1, -1].values                     # Sadece output ile diğer veriler arası corr
    corr_output = np.absolute(corr_output)                          # Output corr verisinin mutlak değeri
    corr_output_median = np.median(corr_output)                           # Output corr verisinin ortalaması
    selected_value_arr = np.array([])                               # Seçilen değerler için boş array
    index_arr = np.array([])                                        # Seçilen değerlerin indexleri için boş array

    for value in corr_output:                                       # Ortalamayla kıyaslayıp değerleri arraye atan döngü
        if(value >= corr_output_median):
            selected_value_arr = np.append(selected_value_arr, value)
    
    for value in corr_output:                                       # Seçilmesi gereken satırların indexini tutan arrayı oluşturma
        for x in selected_value_arr:
            if value == x:
                index_arr = np.append(index_arr, np.where(corr_output == value))

    index_arr = np.append(index_arr, len(dataset.columns) - 1)      # Seçilen sütunlara output sütununun eklenmesi
    corr_new_dataset = corr_new_dataset.iloc[:, index_arr]          # Seçilen sütunlarla yeni datasetin oluşturulması
    
    return corr_new_dataset


def mse_linear_reg(x, y):                                           # Linear regression

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    lr = LinearRegression()                                 
    lr.fit(x_train,y_train)
    predicts = lr.predict(x_test)
    mse = np.mean((predicts - y_test)**2)
    
    return mse


def mse_filter(dataset):
    
    mse_new_dataset = dataset.copy()                                # Return edilecek yeni dataset
    mse_arr = np.array([])                                          # Seçilen değerlerin mseleri için boş array
    selected_value_arr = np.array([])                               # Seçilen değerler için boş 
    index_arr = np.array([])                                        # Seçilen değerlerin indexleri için boş array

    for i in range(1, len(dataset.columns), 1):
        
        x = dataset.iloc[:, i].values.reshape(-1, 1)                # input
        y = dataset.iloc[:, -1].values.reshape(-1, 1)               # output
        mse_arr = np.append(mse_arr, mse_linear_reg(x, y))
        
    mse_arr = np.absolute(mse_arr)                                  # mse verisinin mutlak değeri
    mse_median = np.median(mse_arr)                                       # mse verisinin ortalaması

    for value in mse_arr:                                           # Ortalamayla kıyaslayıp değerleri arraye atan döngü
        if(value <= mse_median):
            selected_value_arr = np.append(selected_value_arr, value)

    for value in mse_arr:                                           # Seçilmesi gereken satırların indexini tutan arrayı oluşturma
        for x in selected_value_arr:
            if value == x:
                index_arr = np.append(index_arr, np.where(mse_arr == value))
                    
    index_arr = np.append(index_arr, len(dataset.columns) - 1)      # Seçilen sütunlara output sütununun eklenmesi
    mse_new_dataset = mse_new_dataset.iloc[:, index_arr]            # Seçilen sütunlarla yeni datasetin oluşturulması
    
    return mse_new_dataset