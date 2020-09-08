#!/usr/bin/env python
# coding: utf-8

# # Read in data RADIATOR

# In[20]:


import pandas as pd
import numpy as np


# In[2]:


def clean_sas(data):
  for i in data.columns:
    if type(data[i][0]) == bytes:
      data[i] = data[i].str.decode('utf-8')
  return data


# In[21]:


train = clean_sas(pd.read_sas("/rapids/notebooks/my_data/train_lc_val.sas7bdat"))


# In[22]:


test = clean_sas(pd.read_sas("/rapids/notebooks/my_data/predict_lc_val.sas7bdat"))


# # Best model CATBOOST

# In[23]:


# use termial bash conda install catboost
import catboost
catboost.__version__


# In[24]:


from catboost import CatBoostRegressor, Pool, cv


# In[25]:


import sklearn


# In[26]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[27]:


import sys
import os
import time
from datetime import datetime


# In[28]:


train.columns


# In[29]:


train.dtypes


# In[30]:


traincat=train.select_dtypes(include=['object'])


# In[31]:


traincat


# In[32]:


trainint=train.select_dtypes(include=['int'])


# In[33]:


trainint


# In[34]:


nan_values = train.isna()
nan_columns = nan_values.any()

columns_with_nan = train.columns[nan_columns].tolist()
print(columns_with_nan)


# In[35]:


nan_values = test.isna()
nan_columns = nan_values.any()

columns_with_nan = test.columns[nan_columns].tolist()
print(columns_with_nan)


# In[36]:


test['mega_store_key']


# In[37]:


trainbackup=train.copy()


# In[38]:


testbackup=test.copy()


# In[ ]:


# train=train.fillna(0)


# In[ ]:


# test=train.fillna(0)


# In[39]:


test.columns


# In[40]:


otrain = train.reindex(sorted(train.columns), axis=1)
otest = test.reindex(sorted(test.columns), axis=1)


# In[41]:


otrain["URBAN_RURAL_SEGMENT"]=otrain["URBAN_RURAL_SEGMENT"].astype(int)


# In[42]:


otest["URBAN_RURAL_SEGMENT"]=otest["URBAN_RURAL_SEGMENT"].astype(int)


# In[ ]:


# for i in ['CLIMATE_CODE','hub_store_flag','major','major_intersection','minor','max_per_car','part_type','fogid','PLCS','stk_period','urban_rural_segment']:
# for i in ['CLIMATE_CODE','MAX_PER_CAR','fogid','PLCS','stkpds','URBAN_RURAL_SEGMENT']:


# In[ ]:


# drop list
# "item", "store","nxt_sales_qty","stkpds","source","agemax","min_model_yr","APP_LKUPS","HUB_STORE_KEY","mega_store_key","DATE_OPENED","weeks","rtl","open_coeff","open_coeff2","fogid","MAX_PER_CAR","PLCS","lks_comm_ratio","lks_shift_c2r","URBAN_RURAL"


# In[43]:


def modelrun(iterations_v, learning_rate_v, depth_v):  
    
        tr_drop = ["item", "store","nxt_sales_qty","stkpds","source","agemax","min_model_yr","APP_LKUPS","HUB_STORE_KEY","mega_store_key","DATE_OPENED","weeks","rtl","open_coeff","open_coeff2","fogid","MAX_PER_CAR","PLCS","lks_comm_ratio","lks_shift_c2r","URBAN_RURAL"]
        te_drop = tr_drop.copy()
   

        y_train = otrain["nxt_sales_qty"]
        x_train = otrain.drop(tr_drop,1)
 

        x_test = otest.drop(te_drop,1)

        x_train = x_train[sorted(list(x_train.columns))]
        x_test = x_test[sorted(list(x_test.columns))]

        categorical_features_indices=[]
        for i in ['CLIMATE_CODE', 'COMM_SALES_FLAG','URBAN_RURAL_SEGMENT']:
            if x_train[i].dtype=="float64" or x_test[i].dtype=="float64":
                x_train[i]=x_train[i].astype(str)
                x_test[i]=x_test[i].astype(str)
            categorical_features_indices.append(x_train.columns.get_loc(i))

        model = CatBoostRegressor(
          iterations=iterations_v,
          depth=depth_v,
          learning_rate=learning_rate_v,
          loss_function='RMSE',
          eval_metric='RMSE',
          random_seed=42,
          logging_level='Silent'
        )

        X_trainN, X_cv, y_trainN, y_cv = train_test_split(x_train, y_train, train_size=0.8, random_state=1234)
        
        train_pool = Pool(data=X_trainN, 
                  label=y_trainN, 
                  cat_features=categorical_features_indices)

        validation_pool = Pool(data=X_cv, 
                          label=y_cv, 
                       cat_features=categorical_features_indices)
        model.fit(
            train_pool,
            eval_set=validation_pool,
            use_best_model=True,
            verbose=True
        )
        
    
        preds = model.predict(x_test)

        pred_n13 = pd.DataFrame(otest[["item","store","nxt_sales_qty"]])
        pred_n13["pred"] = preds


        comp = pred_n13.copy()
        comp = comp.groupby("item").sum()
        comp = comp[comp.nxt_sales_qty!=0]
        comp["ml_diff"] = np.round(np.abs((comp["nxt_sales_qty"]-comp["pred"])/comp["nxt_sales_qty"])*100,2)

        in10 = comp[np.abs(comp["ml_diff"])<10]["pred"].count()
        in20 = comp[np.abs(comp["ml_diff"])<20]["pred"].count()
        
        in10pct = in10/pred_n13["item"].nunique()
        in20pct = in20/pred_n13["item"].nunique()
       
        
        d = {"learning_rate": learning_rate_v,
            "iterations": iterations_v,
            "depth": depth_v,
            "in10": in10,
            "in20": in20,
            "in10percent": in10pct,
            "in20percent": in20pct}
        return d


# In[65]:


iterations_v1=[1000]
learning_rate_v1=[0.01]
depth_v1=[7]
exe_count=len(iterations_v1)*len(learning_rate_v1)*len(depth_v1)


# In[66]:


c=0
results1=[]
for i in  iterations_v1:
    for j in learning_rate_v1: 
        for k in depth_v1: 
            results1.append(modelrun(iterations_v = i, learning_rate_v = j, depth_v=k))
            c+=1
            print(np.round(c/exe_count,2)*100)


# In[64]:


seq = [x['in20'] for x in results1]
print(min(seq))

print(max(seq))
seq.index(max(seq))
print(results1[seq.index(max(seq))])
print(results1[seq.index(min(seq))])


# In[60]:


print("hello world")


# 33
# 41
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 5, 'in10': 20, 'in20': 41, 'in10percent': 0.06042296072507553, 'in20percent': 0.12386706948640483}
# {'learning_rate': 0.01, 'iterations': 500, 'depth': 5, 'in10': 20, 'in20': 33, 'in10percent': 0.06042296072507553, 'in20percent': 0.09969788519637462}

# 37
# 37
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 8, 'in10': 21, 'in20': 37, 'in10percent': 0.0634441087613293, 'in20percent': 0.11178247734138973}
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 8, 'in10': 21, 'in20': 37, 'in10percent': 0.0634441087613293, 'in20percent': 0.11178247734138973}

# 107
# 107
# {'learning_rate': 0.01, 'iterations': 900, 'depth': 6, 'in10': 50, 'in20': 107, 'in10percent': 0.1510574018126888, 'in20percent': 0.32326283987915405}
# {'learning_rate': 0.01, 'iterations': 900, 'depth': 6, 'in10': 50, 'in20': 107, 'in10percent': 0.1510574018126888, 'in20percent': 0.32326283987915405}

# In[ ]:





# In[ ]:




