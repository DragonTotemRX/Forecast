'''
CB.py CatBoost for High Volumn items 
'''

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import sys
import os
import psutil
from utils import pr_ot,pr_imp

import time
start = time.time()

##COMMAND LINE 
#input FOGID varX
FOGID = str(sys.argv[1]).upper()

#input FY00P00 varY 
FY_P = str(sys.argv[2]).upper()

#input M1 for stocked M2 for full 
MODEL = str(sys.argv[3]).split("_")[0].upper()

#output option VAL or Predict 
VAL = "Predict" #rename as VAL
if len(str(sys.argv[3]).split("_"))>1:
  VAL = str(sys.argv[3]).split("_")[1].upper()
if VAL == "VAL":
  VAL = True
else:
  VAL = False

#sd source directory 
sd = "/rapids/notebooks/my_data/"+FY_P+"/" + FOGID + "/"
if VAL:
    sd = sd + "Val/"
    
#dd destination directoy 
dd = sd + "objctv/"
if VAL:
  OUTPUT_FILE = "cb_hi_full_val.csv"
else:
  OUTPUT_FILE = "cb_hi_full.csv"

#File names of the train and test 
if len(sys.argv)==6:
  TRAIN_FILE = sd + str(sys.argv[4])
  PRED_FILE = sd + str(sys.argv[5])  
else:
  TRAIN_FILE = sd + "train_hi_v.csv"
  PRED_FILE = sd + "predict_hi_v.csv"
    
#Model parameters drpX for train and keepY for drpY predict 
FEATURES = ['sat_ep', 'lks_age1', 'lks_age2', 'lks_age3', 'lks_age4', 'lks_age5',
       'lks_age6', 'lks_age7', 'lks_age8', 'lks_age9', 'lks_age10',
       'lks_age11', 'lks_age12', 'lks_age13', 'lks_age14', 'lks_age15',
       'lks_age16', 'lks_age17', 'lks_age18', 'lks_age19', 'lks_age20',
       'lks_age21', 'lks_age22', 'lks_age23', 'lks_age24', 'lks_age25',
       'lks_age26', 'lks_age27', 'lks_age28', 'lks_age29', 'lks_age30', 'lks',
       'lks_Y', 'stkpds', 'polk', 'warr', 'CLIMATE_CODE', 'COMM_SALES_FLAG',
       'major_int', 'hubflag', 'URBAN_RURAL_SEGMENT', 'adv', 'ore',
       'store_volume', 'CRIME_INDEX', 'POPULATION', 'PERCENT_BLACK',
       'PERCENT_HISPANIC', 'PERCENT_ASIAN', 'median_age', 'MEDIAN_INCOME',
       'TOTAL_OKV']
TARGET = 'nxt_sales_qty'

#Adjustments based on Model type 
if MODEL in ["M1","MODEL1"]:
  FEATURES = FEATURES + [feat for feat in ['SALES_QTY', 'TRNX']]
  FEATURES.remove('stkpds')

#read data into memory d1 for train d2 for predict 
pr_ot('mem pct='+str(psutil.virtual_memory().percent))
pr_ot("Retrieving data files...")
pr_ot("reading "+TRAIN_FILE)
ds1 = pr_imp(TRAIN_FILE)
pr_ot("reading "+PRED_FILE)
ds2 = pr_imp(PRED_FILE)
if ds1.shape[0] == 0 or ds2.shape[0] == 0:
  pr_ot("Data empty.")
  exit(0)
pr_ot("Data for " + MODEL + " retrieved successfully...")
pr_ot('mem pct='+str(psutil.virtual_memory().percent))

#After the MODEL drops
if MODEL in ["M1","MODEL1"]:
  print('INSIDE if')
  ds2 = ds2[ds2.stkpds==13]
  if VAL:
    OUTPUT_FILE = "cb_hi_stocked_val.csv"
  else:
    OUTPUT_FILE = "cb_hi_stocked.csv"
    
# data process 
items = list(set(ds1['item'].unique()) & set(ds2['item'].unique()))
RESULTS_LIST = []
ps=[i*10.0 for i in range(1,11)]
ds1.set_index("item", inplace = True)
ds2.set_index("item", inplace = True)

CATEGORICAL_FEATS = [feat for feat in ds1[FEATURES].columns if ds1[feat].dtype.name == 'object'] + ['URBAN_RURAL_SEGMENT', 'hubflag', 'major_int']
model = CatBoostRegressor(
iterations=900,
depth=6,
learning_rate=0.01,
loss_function='RMSE',
eval_metric='RMSE',
random_seed=42,
thread_count = 10,
logging_level='Silent',
cat_features=CATEGORICAL_FEATS      
)

# modeling 
pr_ot("Beginning training and testing loop for CB model with items count="+str(len(items)))
for i, item in enumerate(items):
  
  #Percentage Bar
  p= i / len(items) * 100
  if p >= ps[0]:
    percent = ps.pop(0)
    pr_ot(str(percent)+" Percent Completed")

  train =  ds1.loc[[item]]  
  test =  ds2.loc[[item]]

  if train[TARGET].nunique()==1:
    continue

  model.fit(train[sorted(FEATURES)], train[TARGET])
  preds = model.predict(test[sorted(FEATURES)])
  
  if VAL:
    objctv2 = test.reset_index()[["item","store","nxt_sales_qty"]].copy()
  else:
    objctv2 = test.reset_index()[["item","store"]].copy()
  objctv2["pred"] = preds
  RESULTS_LIST.append(objctv2)

objctv = pd.concat(RESULTS_LIST)
    
# export results 
pr_ot("Done.")
if not(os.path.isdir(dd)):
    os.makedirs(dd)
objctv.to_csv(dd+OUTPUT_FILE,index=False)
pr_ot("Results for " + MODEL + " saved successfully to: " + dd + OUTPUT_FILE)

end = time.time()
temp = end-start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('CB_v2.py processing time H:M:S = %d:%d:%d' %(hours,minutes,seconds))