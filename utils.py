import cudf
import pandas as pd
import sys
import numpy as np
def get_mx_val(df):
  fd=df.copy()
  cnts=fd.groupby("item").size().reset_index(name="cnt")
  pcntl=fd.groupby("item").pred_fin.quantile([.85,.9,.95,.99]).reset_index().rename(
                                            columns={"level_1":"percentile","pred_fin":"value"})
  fd=pd.merge(pcntl,cnts,on="item",how="inner")
  fd["mx_val"]=np.NaN
  try:
    fd.loc[50>=fd.cnt,"mx_val"]=1*(fd.loc[fd.percentile==.85]["value"])
    fd.loc[(100>=fd.cnt) & (fd.cnt>50),"mx_val"]=1*(fd.loc[fd.percentile==.90]["value"])
    fd.loc[(200>=fd.cnt) & (fd.cnt>100),"mx_val"]=2*(fd.loc[fd.percentile==.95]["value"])
    fd.loc[(500>fd.cnt) & (fd.cnt>200),"mx_val"]=5*(fd.loc[fd.percentile==.95]["value"])
    fd.loc[fd.cnt>=500,"mx_val"]=5*(fd.loc[fd.percentile==.99]["value"])
  except:
    #everything above thresholds 
    fd["mx_val"]=5*(fd.loc[fd.percentile==.99]["value"])   
  fd=fd[pd.notna(fd.mx_val)]
  return fd[["item","mx_val"]]
def get_weight(df):
  fd=df.copy()
  #replace all zero sales with ones and sum as modified total
  fd["mod_sales"]=fd.SALES_QTY.replace(0,1)
  fd = fd.groupby("item").sum().rename(columns={"SALES_QTY":"total",
                                               "mod_sales":"modified_total"})
  fd.loc[fd.total<0,'total'] = 0
  fd["weight"]=np.sqrt(fd.total/fd.modified_total)
  fd.loc[fd.weight<.8,"weight"]=.8
  return fd.reset_index()[["item","weight"]] 

def get_stats(ds):
  stats={}
  stats["count_is_f"] = ds.shape[0]
  stats["count_i_f"] = ds["item"].nunique()
  stats["min_stkpds_f"] = ds["stkpds"].min()
  stats["max_stkpds_f"] = ds["stkpds"].max()
  stats["avg_stkpds_f"] = ds["stkpds"].mean()
  stats["count_is_i"] = ds.groupby(['item']).size().reset_index(name='Count')[["item","Count"]]
  stats["avg_stkpds_i"] = ds.groupby(['item']).mean().reset_index()[["item","stkpds"]].rename(columns={"stkpds":"avg_stkpds"})
  return stats

def pr_imp(fn):
    if fn.split(".")[-1] == "pkl":
      data = pd.read_pickle(fn)
    if fn.split(".")[-1] == "sas7bdat":
      data = pd.read_sas(fn)
    if fn.split(".")[-1] == "csv":
      data = cudf.read_csv(fn).to_pandas()
    if data.shape[0] > 0:
      for i in data.columns:
        if type(data[i][0]) == bytes:
          data[i] = data[i].str.decode('utf-8')
      data[["URBAN_RURAL_SEGMENT","hubflag","major_int"]] = data[["URBAN_RURAL_SEGMENT","hubflag","major_int"]].astype('int8')
    return Reduce_mem_usage(data)

def pr_ot(string):
  sys.stdout.write(string + "\n")
  sys.stdout.flush()
    
def combine_fcst(d,full=False,val=False):
  if val:
    if full:
      cb = pd.read_csv(d+"cb_hi_full_val.csv")
      rf = pd.read_csv(d+"rf_hi_full_val.csv") 
      cn = "pred_m2"
    else:
      cb = pd.read_csv(d+"cb_hi_stocked_val.csv")
      rf = pd.read_csv(d+"rf_hi_stocked_val.csv")
      cn = "pred_m1"
    cb = cb.drop("nxt_sales_qty",1)
  else:
    if full:
      cb = pd.read_csv(d+"cb_hi_full.csv")
      rf = pd.read_csv(d+"rf_hi_full.csv") 
      cn = "pred_m2"
    else:
      cb = pd.read_csv(d+"cb_hi_stocked.csv")
      rf = pd.read_csv(d+"rf_hi_stocked.csv")
      cn = "pred_m1"
  
  cb = cb.rename(columns={"pred":"pred_cb"})
  rf = rf.rename(columns={"pred":"pred_rf"})
  forecast = pd.merge(rf,cb,how="left",on=["item","store"]) 
  forecast = forecast.fillna(-99)
  forecast[cn] = np.where(forecast["pred_cb"] < 0, forecast["pred_rf"], forecast["pred_cb"])
  return forecast

def combine_full_fcst(read=True,f1=None,f2=None,val=False):
  if val:
    if read:
      m1 = pd.read_csv(d+"pred_stocked_val.csv")      
      m2 = pd.read_csv(d+"pred_full_val.csv")
    else:
      m1 = f1
      m2 = f2
    m1 = m1.drop("nxt_sales_qty",1)
  else:
    if read:
      m1 = pd.read_csv(d+"pred_stocked.csv")
      m2 = pd.read_csv(d+"pred_full.csv")
    else:
      m1 = f1
      m2 = f2
  
  m1["model_flag"]=1  
  m1=m1.drop('model',1)
  forecast = pd.merge(m2,m1,how="left",on=["item","store"])
  forecast["pred_m1"] = forecast["pred_m1"].fillna(-99)
  forecast["model_flag"] = forecast["model_flag"].fillna(2)
  forecast["pred_fin"] = np.where(forecast["pred_m1"] < 0, forecast["pred_m2"], forecast["pred_m1"])
  if val:
    forecast = forecast[["item","store","pred_fin","nxt_sales_qty","model_flag","model"]]
  else:
    forecast = forecast[["item","store","pred_fin","model_flag","model"]]
  return forecast


#This from Grandmaster Konstantin Yakovlev 
#modified by Andrew Ott
#from https://www.kaggle.com/kyakovlev/m5-simple-fe
##############################################################################
#Reduced_mem_usage: downcasting numeric features that can be downcasted without
# loss.  i.e. int64 columns with max value of 35 gets downcasted to int8.  This
# FUNCTION IS INPLACE:  It will modify the original df_! 
#INPUTS:
    #df:  dataframe
#OUTPUTS:
    #df: the dataframe with downcasted features
##############################################################################
def Reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        
        #Checking first if it is object or categorical.  If so, just move on.  Further 
        # down the road when I know where this code is going, I will convert objects
        # to categorical.  
        if col_type.name == 'category':
            continue
        if col_type == 'object':
            continue
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
def Test_path():
    print('It worked!')