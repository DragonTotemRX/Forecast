#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
def clean_sas(data):
  for i in data.columns:
    if type(data[i][0]) == bytes:
      data[i] = data[i].str.decode('utf-8')
  return data
train = clean_sas(pd.read_sas("/rapids/notebooks/team_data/personal/cjackso/ngf_check/train_2020.sas7bdat"))
test = clean_sas(pd.read_sas("/rapids/notebooks/team_data/personal/cjackso/ngf_check/predict_2020.sas7bdat"))
from catboost import CatBoostRegressor, Pool, cv
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sys
import os
import time
from datetime import datetime
otrain = train.reindex(sorted(train.columns), axis=1)
otest = test.reindex(sorted(test.columns), axis=1)
otrain["URBAN_RURAL_SEGMENT"]=otrain["URBAN_RURAL_SEGMENT"].astype(int)
otest["URBAN_RURAL_SEGMENT"]=otest["URBAN_RURAL_SEGMENT"].astype(int)
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


# In[24]:


iterations_v1=[500]
learning_rate_v1=[0.05]
depth_v1=[8]
exe_count=len(iterations_v1)*len(learning_rate_v1)*len(depth_v1)
c=0
results1=[]
for i in  iterations_v1:
    for j in learning_rate_v1: 
        for k in depth_v1: 
            results1.append(modelrun(iterations_v = i, learning_rate_v = j, depth_v=k))
            c+=1
            print(np.round(c/exe_count,2)*100)


# In[26]:


seq = [x['in20'] for x in results1]
print(min(seq))

print(max(seq))
seq.index(max(seq))
print(results1[seq.index(max(seq))])
print(results1[seq.index(min(seq))])


# 101
# 146
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 5, 'in10': 70, 'in20': 146, 'in10percent': 0.1446280991735537, 'in20percent': 0.30165289256198347}
# {'learning_rate': 0.05, 'iterations': 500, 'depth': 5, 'in10': 54, 'in20': 101, 'in10percent': 0.1115702479338843, 'in20percent': 0.20867768595041322}

# 148
# 148
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 6, 'in10': 71, 'in20': 148, 'in10percent': 0.14669421487603307, 'in20percent': 0.30578512396694213}
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 6, 'in10': 71, 'in20': 148, 'in10percent': 0.14669421487603307, 'in20percent': 0.30578512396694213}

# 151
# 151
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 7, 'in10': 75, 'in20': 151, 'in10percent': 0.15495867768595042, 'in20percent': 0.3119834710743802}
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 7, 'in10': 75, 'in20': 151, 'in10percent': 0.15495867768595042, 'in20percent': 0.3119834710743802}

# 139
# 139
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 10, 'in10': 73, 'in20': 139, 'in10percent': 0.15082644628099173, 'in20percent': 0.2871900826446281}
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 10, 'in10': 73, 'in20': 139, 'in10percent': 0.15082644628099173, 'in20percent': 0.2871900826446281}

# 138
# 155
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 8, 'in10': 81, 'in20': 155, 'in10percent': 0.16735537190082644, 'in20percent': 0.3202479338842975}
# {'learning_rate': 0.1, 'iterations': 500, 'depth': 9, 'in10': 77, 'in20': 138, 'in10percent': 0.1590909090909091, 'in20percent': 0.28512396694214875}

# 132
# 132
# {'learning_rate': 0.1, 'iterations': 600, 'depth': 8, 'in10': 65, 'in20': 132, 'in10percent': 0.13429752066115702, 'in20percent': 0.2727272727272727}
# {'learning_rate': 0.1, 'iterations': 600, 'depth': 8, 'in10': 65, 'in20': 132, 'in10percent': 0.13429752066115702, 'in20percent': 0.2727272727272727}

# 89
# 89
# {'learning_rate': 0.01, 'iterations': 500, 'depth': 8, 'in10': 47, 'in20': 89, 'in10percent': 0.09710743801652892, 'in20percent': 0.18388429752066116}
# {'learning_rate': 0.01, 'iterations': 500, 'depth': 8, 'in10': 47, 'in20': 89, 'in10percent': 0.09710743801652892, 'in20percent': 0.18388429752066116}

# 103
# 103
# {'learning_rate': 0.05, 'iterations': 200, 'depth': 10, 'in10': 59, 'in20': 103, 'in10percent': 0.12190082644628099, 'in20percent': 0.2128099173553719}
# {'learning_rate': 0.05, 'iterations': 200, 'depth': 10, 'in10': 59, 'in20': 103, 'in10percent': 0.12190082644628099, 'in20percent': 0.2128099173553719}

# 116
# 116
# {'learning_rate': 0.05, 'iterations': 500, 'depth': 8, 'in10': 66, 'in20': 116, 'in10percent': 0.13636363636363635, 'in20percent': 0.2396694214876033}
# {'learning_rate': 0.05, 'iterations': 500, 'depth': 8, 'in10': 66, 'in20': 116, 'in10percent': 0.13636363636363635, 'in20percent': 0.2396694214876033}
