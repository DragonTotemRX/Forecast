from utils import combine_fcst,combine_full_fcst
import pandas as pd
import sys

# Read in fog as first argument to script from command line and set objctv directory
fogID = str(sys.argv[1]).upper()
prd = str(sys.argv[2]).upper()
val=False
try:
  val = str(sys.argv[3])
  val = True
except:
  pass

if val:
  d = "/rapids/notebooks/my_data/"+prd+"/"+fogID+"/Val/objctv/" 
else:
  d = "/rapids/notebooks/my_data/"+prd+"/"+fogID+"/objctv/" 

# Use function to combine forecasts then save
if val:
  try:
    hi_m1 = combine_fcst(d,val=val)
  except:
    hi_m1 = pd.DataFrame(columns=["item","store","nxt_sales_qty","pred_m1"])
  try:
    lc_m1 = pd.read_csv(d+"rf_lc_stocked_val.csv")
  except:
    lc_m1=pd.DataFrame(columns=["pred"])
  hi_m1['model'] = 1
  lc_m1['model'] = 2
  fcst_m1 = pd.concat([hi_m1[["item","store","nxt_sales_qty","pred_m1","model"]],lc_m1.rename(columns={"pred":"pred_m1"})])
else:  
  try:
    hi_m1 = combine_fcst(d,val=val)
  except:
    hi_m1 = pd.DataFrame(columns=["item","store","pred_m1"])
  try:  
    lc_m1 = pd.read_csv(d+"rf_lc_stocked.csv")
  except:
    lc_m1 = pd.DataFrame(columns=["pred"])
  hi_m1['model'] = 1
  lc_m1['model'] = 2
  fcst_m1 = pd.concat([hi_m1[["item","store","pred_m1","model"]],lc_m1.rename(columns={"pred":"pred_m1"})])

fcst_m1=fcst_m1.sort_values(by=['item', 'store'])
if val:
  fcst_m1.to_csv(d+"pred_stocked_val.csv",index=False)
else:
  fcst_m1.to_csv(d+"pred_stocked.csv",index=False)
print("fcst_m1 exported to "+d)

if val:
  try:
    hi_m2 = combine_fcst(d,full=True,val=val)
  except:
    hi_m2 = pd.DataFrame(columns=["item","store","nxt_sales_qty","pred_m2"])
  try:
    lc_m2 = pd.read_csv(d+"rf_lc_full_val.csv")
  except:
    lc_m2 = pd.DataFrame()
  hi_m2['model'] = 1
  lc_m2['model'] = 2
  fcst_m2 = pd.concat([hi_m2[["item","store","nxt_sales_qty","pred_m2","model"]],lc_m2.rename(columns={"pred":"pred_m2"})])
else:
  try:
    hi_m2 = combine_fcst(d,full=True,val=val)
  except:
    hi_m2 = pd.DataFrame(columns=["item","store","nxt_sales_qty","pred_m2"])
  try:
    lc_m2 = pd.read_csv(d+"rf_lc_full.csv")
  except:
    lc_m2 = pd.DataFrame(columns=["pred"])
  hi_m2['model'] = 1
  lc_m2['model'] = 2
  fcst_m2 = pd.concat([hi_m2[["item","store","pred_m2","model"]],lc_m2.rename(columns={"pred":"pred_m2"})])

fcst_m2=fcst_m2.sort_values(by=['item', 'store'])
if val:
  fcst_m2.to_csv(d+"pred_full_val.csv",index=False)
else:
  fcst_m2.to_csv(d+"pred_full.csv",index=False)
print("fcst_m2 exported to "+d)

fcst_fin = combine_full_fcst(False,fcst_m1,fcst_m2,val=val)
fcst_fin=fcst_fin.sort_values(by=['item', 'store'])  
if val:
  fcst_fin.to_csv(d+"forecast_val.csv",index=False)
else:
  fcst_fin.to_csv(d+"forecast.csv",index=False)
print("fcst_fin exported to "+d)

print("Models combined in "+d)
