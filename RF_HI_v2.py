'''
RF_HI.py Random Forest for High Volumn items 
'''

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
  print(f'Inside if: {VAL}')
if VAL == "VAL":
  VAL = True
else:
  VAL = False
print(f'VAL: {VAL}')

#sd source directory 
sd = "/rapids/notebooks/my_data/"+FY_P+"/" + FOGID + "/"
if VAL:
    sd = sd + "Val/"
    
#dd destination directoy 
dd = sd + "objctv/"

#TRAIN_FILE is train input file; PRED_FILE predict input file name 
if len(sys.argv)==6:
  TRAIN_FILE = sd + str(sys.argv[4])
  PRED_FILE = sd + str(sys.argv[5])  
else:
  TRAIN_FILE = sd + "train_hi_v.csv"
  PRED_FILE = sd + "predict_hi_v.csv"

#List of all features to be used in model
FEATURES=['polk','warr','stkpds',
'lks_age1', 'lks_age2', 'lks_age3', 'lks_age4', 'lks_age5', 'lks_age6', 'lks_age7', 'lks_age8', 
'lks_age9', 'lks_age10','lks_age11', 'lks_age12', 'lks_age13', 'lks_age14', 'lks_age15', 
'lks_age16', 'lks_age17', 'lks_age18', 'lks_age19','lks_age20', 'lks_age21', 'lks_age22', 
'lks_age23', 'lks_age24', 'lks_age25', 'lks_age26', 'lks_age27', 'lks_age28','lks_age29', 'lks_age30','lks']

TARGET = 'nxt_sales_qty'

if MODEL in ["M1","MODEL1"]:
  FEATURES = FEATURES + ["SALES_QTY", "TRNX"]
  FEATURES.remove("stkpds")

    
#read data into memory d1 for train d2 for predict 
pr_ot('mem pct='+str(psutil.virtual_memory().percent))
pr_ot("Retrieving data files...")
pr_ot("reading "+TRAIN_FILE)
ds1 = pr_imp(TRAIN_FILE)
pr_ot("reading "+PRED_FILE)
ds2 = pr_imp(PRED_FILE)
pr_ot("Data for " + MODEL + " retrieved successfully...")
pr_ot('mem pct='+str(psutil.virtual_memory().percent))

if ds1.shape[0] == 0 or ds2.shape[0] == 0:
  pr_ot("Data empty.")
  exit(0)


#OUTPUT_FILE output file name 
if MODEL in ["M1","MODEL1"]:
  ds2 = ds2[ds2.stkpds==13]
  if VAL:
    OUTPUT_FILE = "rf_hi_stocked_val.csv"
  else:
    OUTPUT_FILE = "rf_hi_stocked.csv"
else:  
  if VAL:
    OUTPUT_FILE = "rf_hi_full_val.csv"
  else:
    OUTPUT_FILE = "rf_hi_full.csv"

print('')
print(f'OUTPUT_FILE: {OUTPUT_FILE}, VAL: {VAL}, MODEL: {MODEL}')
#exit(0)



# data process 
ds1 =pd.get_dummies(ds1, columns=['URBAN_RURAL'])
ds2 = pd.get_dummies(ds2, columns=['URBAN_RURAL'])
FEATURES = FEATURES + ['URBAN_RURAL_U', 'URBAN_RURAL_R']
FEATURES = sorted(FEATURES)# may introduce divergent preds as this is UPPER then lower alphabetization

items = list(set(ds1['item'].unique()) & set(ds2['item'].unique()))
objctv = pd.DataFrame()

ps=[10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0, 100.0]
ds1 = ds1.set_index("item")
ds2= ds2.set_index("item")

# modeling 
pr_ot("Beginning training and testing loop for RF_HI model with items count="+str(len(items)))
RESULTS_LIST = []
for i, item in enumerate(items):
  p= i / len(items) * 100
  if p >= ps[0]:
    percent = ps.pop(0)
    pr_ot(str(percent)+" Percent Completed")
    
  train =  ds1.loc[[item]].reset_index()  #was ds3
  test =  ds2.loc[[item]].reset_index()   #was ds5
  
  model = rf(n_jobs=10,n_estimators=100,max_depth=6,random_state=10)
  model.fit(train[FEATURES], train[TARGET])
  preds = model.predict(test[FEATURES])
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
objctv.to_csv(dd+OUTPUT_FILEindex=False)
#objctv.to_csv('/rapids/notebooks/my_code/PREDS_{}'.format(OUTPUT_FILE), index = False)
pr_ot("Results for " + MODEL + " saved successfully to: " + dd + OUTPUT_FILE)

end = time.time()
temp = end-start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('RF_HI_v2.py processing time H:M:S = %d:%d:%d' %(hours,minutes,seconds))