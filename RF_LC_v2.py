'''
RF_LC.py Random Forest for Low Volumn items 
'''
import gc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rf
import os
import sys
import psutil
from utils import pr_ot,pr_imp

import time
start = time.time()

#input FOGID
FOGID = str(sys.argv[1]).upper()

#input FY00P00 
FY_P = str(sys.argv[2]).upper()

#input M1 for stocked M2 for full 
MODEL = str(sys.argv[3]).split("_")[0].upper()

#output option VAL or Predict 
VAL = "Predict"
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
  OUTPUT_FILE = "rf_lc_full_val.csv"
else:
  OUTPUT_FILE = "rf_lc_full.csv"

#TRAIN_FILE train input file name PRED_FILE predict input file name 
if len(sys.argv)==6:
  TRAIN_FILE = sd + str(sys.argv[4])
  PRED_FILE = sd + str(sys.argv[5])  
elif VAL:
  TRAIN_FILE = sd + "train_lc_v.csv"
  PRED_FILE = sd + "predict_lc_v.csv"
else:
  TRAIN_FILE = sd + "train_lc_v.csv"
  PRED_FILE = sd + "predict_lc_v.csv"
    
#Model parameters keepX for train and keepY for predict 
TARGET = 'nxt_sales_qty'
dummies = ["CLIMATE_CODE","adv","ore","hubflag","major_int","MAX_PER_CAR","PLCS","URBAN_RURAL_SEGMENT","fogid","COMM_SALES_FLAG"]
FEATURES = ['sat_ep', 'lks_age1', 'lks_age2', 'lks_age3', 'lks_age4', 'lks_age5',
       'lks_age6', 'lks_age7', 'lks_age8', 'lks_age9', 'lks_age10',
       'lks_age11', 'lks_age12', 'lks_age13', 'lks_age14', 'lks_age15',
       'lks_age16', 'lks_age17', 'lks_age18', 'lks_age19', 'lks_age20',
       'lks_age21', 'lks_age22', 'lks_age23', 'lks_age24', 'lks_age25',
       'lks_age26', 'lks_age27', 'lks_age28', 'lks_age29', 'lks_age30',
       'stkpds', 'polk', 'warr', 'curr_stock', 'PLCS', 'MAX_PER_CAR', 'fogid',
       'CLIMATE_CODE', 'COMM_SALES_FLAG', 'major_int', 'hubflag',
       'URBAN_RURAL_SEGMENT', 'adv', 'ore', 'store_volume', 'CRIME_INDEX',
       'POPULATION', 'PERCENT_BLACK', 'PERCENT_HISPANIC', 'PERCENT_ASIAN',
       'median_age', 'MEDIAN_INCOME', 'TOTAL_OKV', 'itm_stkstrcnt',
       'itm_sales_qty', 'itm_Soldstr0', 'itm_strsls_avg', 'itm_sls_nstd',
       'itm_ttlks', 'itm_strttlks0', 'itm_ttlks_avg', 'itm_ttlks_nstd']

if MODEL in ["M1","MODEL1"]:
    FEATURES += ['SALES_QTY', 'TRNX']
    FEATURES = [feat for feat in FEATURES if feat != 'stkpds']
#read data into memory d1 for train d2 for predict 
pr_ot('mem pct='+str(psutil.virtual_memory().percent))
pr_ot("Retrieving data files...")
pr_ot("reading "+TRAIN_FILE)
train = pr_imp(TRAIN_FILE)
pr_ot("reading "+PRED_FILE)
test = pr_imp(PRED_FILE)
pr_ot("Data for " + MODEL + " retrieved successfully...")
pr_ot('mem pct='+str(psutil.virtual_memory().percent))

#fnZ output file name 
if MODEL in ["M1","MODEL1"]:
  print("m1")
  test = test[test.stkpds==13]
  if VAL:
    OUTPUT_FILE = "rf_lc_stocked_val.csv"
  else:
    OUTPUT_FILE = "rf_lc_stocked.csv"
if train.shape[0] == 0 or test.shape[0] == 0:
  print("Data empty.")
  exit(0)
pr_ot("Data retrieved successfully...")

# data process 
train_dummies = pd.get_dummies(train[dummies],columns=dummies)
test_dummies = pd.get_dummies(test[dummies],columns=dummies)
DUMMY_COLS = list(set(train_dummies.columns) - set(train.columns))

FEATURES += DUMMY_COLS
FEATURES = [feat for feat in FEATURES if feat not in dummies]
train = pd.concat([train,train_dummies],1)
test = pd.concat([test, test_dummies],1)
FEATURES = sorted(FEATURES)

del train_dummies, test_dummies; gc.collect()

# modeling 
pr_ot("Beginning training and testing")
model = rf(n_jobs=10,n_estimators=100,max_depth=10,random_state=10, min_samples_leaf=5) 
model.fit(train[FEATURES], train[TARGET])
preds = model.predict(test[FEATURES])
if VAL:
    objctv2 = pd.DataFrame(test[["item","store","nxt_sales_qty"]])
else:
    objctv2 = pd.DataFrame(test[["item","store"]])
objctv2["pred"] = preds

# export results 
print("Complete.")
if not(os.path.isdir(dd)):
    os.makedirs(dd)
#objctv2.to_csv(dd+OUTPUT_FILE,index=False)
objctv2.to_csv('/rapids/notebooks/my_code/'+'PRED{}'.format(OUTPUT_FILE), index=False)
pr_ot("Results for " + MODEL + " saved successfully to: " + dd + OUTPUT_FILE)

end = time.time()
temp = end-start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('RF_LC_v2.py processing time H:M:S = %d:%d:%d' %(hours,minutes,seconds))