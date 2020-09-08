#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rf
import sys
import os


# NOtes 5/5/2020
# 
# 
# All_data milage data vs  polk data
# 
# ## potential expected fail polk
# real demand comes from lookups
# 10 cars fail , how many come to AZ?
# 
# # lookup better indicator
# 
# 
# adjustment
# design new adjustment it
# 
# ## 1.are there any individual store has fcst higher than average fcst
# ratio 20 times higher to average cap it to 11 times
# 
# ## 2.trend adjustment: how much trend it can change ? the fcst trend is high or low , put a bound on current one as a coefficent 

# # Results
# 
# 
# 
# ## BTENSION
# 
# #### BTENSION_predictlcval_plc2_act_trend is 0.153
# #### BTENSION_predictlcval_plc2_pred_trend is 0.076
# #### within 20% item_store count is  285
# #### total item_store count is 4402
# #### This model has 20% accuracy as 6.4743%.
# 
# 
# 
# ## HOSES
# 
# #### HOSES_predictlcval_plc2_act_trend is 0.098
# #### HOSES_predictlcval_plc2_pred_trend is 0.049
# #### within 20% item_store count is  426
# #### total item_store count is 9682
# #### This model has 20% accuracy as 4.3999%.
# 
# 
# ## CLUCTHES
# 
# #### CLUTCHES_predictlcval_plc2_act_trend is -0.114
# #### CLUTCHES_predictlcval_plc2_pred_trend is -0.224
# #### within 20% item_store count is  92
# #### total item_store count is 20672
# #### This model has 20% accuracy as 0.445%.

# ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# # BTENSION  LC plc2 stk13 model

# In[53]:


B_train_lc_val  = pd.read_csv("/rapids/notebooks/team_data/personal/cjackso/FY20P07/BTENSION/train_lc_val.csv")
B_predict_lc_val  = pd.read_csv("/rapids/notebooks/team_data/personal/cjackso/FY20P07/BTENSION/predict_lc_val.csv")
B_train_lc_val_plc2 = B_train_lc_val.loc[B_train_lc_val['PLCS'] == 2]
B_predict_lc_val_plc2 = B_predict_lc_val.loc[B_predict_lc_val['PLCS'] == 2]
B_train_lc_val_plc2_13 = B_train_lc_val_plc2.loc[B_train_lc_val_plc2['stkpds'] == 13]
B_predict_lc_val_plc2_13 = B_predict_lc_val_plc2.loc[B_predict_lc_val_plc2['stkpds'] == 13]


# In[54]:


## Stock 13 prediction model

dat1=B_train_lc_val_plc2_13.copy()
cat_features=["CLIMATE_CODE","adv","ore","hubflag","major_int","MAX_PER_CAR","PLCS","URBAN_RURAL_SEGMENT","fogid","COMM_SALES_FLAG"]
dat2=pd.get_dummies(dat1,columns=cat_features)
feature_cols=list(dat2.columns)
x_train_backup=dat2[feature_cols]
y_train=dat2.nxt_sales_qty
x_train=x_train_backup.drop(['source','item', 'store','SALES_QTY','TRNX','agemax',
 'min_model_yr','nxt_sales_qty','lks_comm_ratio','lks_shift_c2r','APP_LKUPS','lks','lks_Y',
 'HUB_STORE_KEY','mega_store_key','URBAN_RURAL', 'DATE_OPENED','weeks', 'rtl', 
 'open_coeff', 'open_coeff2'],axis=1)



dat3 = B_predict_lc_val_plc2_13.copy()
dat4=pd.get_dummies(dat3, columns=cat_features)
cat_features=["CLIMATE_CODE","adv","ore","hubflag","major_int","MAX_PER_CAR","PLCS","URBAN_RURAL_SEGMENT","fogid","COMM_SALES_FLAG"]
feature_cols1=list(dat4.columns)
x_predict_backup=dat4[feature_cols1]
x_predict = x_predict_backup. drop(['source','item', 'store','SALES_QTY','TRNX','agemax',
 'min_model_yr','nxt_sales_qty','lks_comm_ratio','lks_shift_c2r','APP_LKUPS','lks','lks_Y',
 'HUB_STORE_KEY','mega_store_key','URBAN_RURAL', 'DATE_OPENED','weeks', 'rtl', 
 'open_coeff', 'open_coeff2'],axis=1)


# In[55]:


model = rf(n_jobs=10,n_estimators=1500,max_depth=20,random_state=10)
model.fit(x_train, y_train)
preds = model.predict(x_predict)
results = pd.DataFrame(x_predict_backup[["item","store","stkpds","SALES_QTY",'nxt_sales_qty']])

results["pred"] = preds

BTENSION_results = results.copy()

BTENSION_resultsagg = BTENSION_results.copy()

BTENSION_resultsagg['dif'] = (BTENSION_resultsagg['pred'] - BTENSION_resultsagg['nxt_sales_qty']).abs()
BTENSION_resultsagg['t20percentnxtsale'] = BTENSION_resultsagg['nxt_sales_qty']*0.2
b_accurate_predict20p = BTENSION_resultsagg.loc[BTENSION_resultsagg['dif'] < BTENSION_resultsagg['t20percentnxtsale']]
b_p_t20p = round(((b_accurate_predict20p.shape[0]/BTENSION_resultsagg.shape[0])*100),4)
print(f"This model has 20% accuracy as {b_p_t20p}%.")


# In[56]:


print(f"within 20% item_store count is  {b_accurate_predict20p.shape[0]}")
print(f"total item_store count is {BTENSION_resultsagg.shape[0]}")
print(f"This model has 20% accuracy as {b_p_t20p}%.")

                            model = rf(n_jobs=10,n_estimators=100,max_depth=6,random_state=10, min_samples_leaf=5) 
                            This model has 20% accuracy as 3.2258%.

                            model = rf(n_jobs=10,n_estimators=100,max_depth=10,random_state=10, min_samples_leaf=5) 
                            This model has 20% accuracy as 3.7937%.

                            model = rf(n_jobs=10,n_estimators=1000,max_depth=6,random_state=10, min_samples_leaf=5) 
                            This model has 20% accuracy as 3.4303%.

                            model = rf(n_jobs=10,n_estimators=1000,max_depth=10,random_state=10, min_samples_leaf=5)
                            This model has 20% accuracy as 4.1345%.

                            model = rf(n_jobs=10,n_estimators=1000,max_depth=10,random_state=10, min_samples_leaf=10) 
                            This model has 20% accuracy as 3.8392%.


                            model = rf(n_jobs=10,n_estimators=1000,max_depth=10,random_state=10, min_samples_leaf=2) 
                            This model has 20% accuracy as 4.3389%.


                            model = rf(n_jobs=10,n_estimators=1000,max_depth=10,random_state=10) 
                            This model has 20% accuracy as 4.5661%.



                            model = rf(n_jobs=10,n_estimators=1000,max_depth=15,random_state=10) 
                            This model has 20% accuracy as 5.7928%.


                            model = rf(n_jobs=10,n_estimators=1000,max_depth=20,random_state=10)
                            This model has 20% accuracy as 6.2926%.

model = rf(n_jobs=10,n_estimators=1500,max_depth=20,random_state=10) 
This model has 20% accuracy as 6.4743%.


                            model = rf(n_jobs=10,n_estimators=2000,max_depth=20,random_state=10) 
                            This model has 20% accuracy as 6.3835%.

                            model = rf(n_jobs=10,n_estimators=2000,random_state=10) 
                            This model has 20% accuracy as 6.4289%.


                            model = rf(n_jobs=10,n_estimators=3000,random_state=10) 
                            This model has 20% accuracy as 6.2699%.


                            model = rf(n_jobs=10,random_state=10) 
                            This model has 20% accuracy as 5.861%.
                            
                            
                            model = rf(n_jobs=10,n_estimators=1500,max_depth=25,random_state=10) 
                            This model has 20% accuracy as 6.4516%.
                            
                            model = rf(n_jobs=10,n_estimators=1500,max_depth=30,random_state=10)
                            This model has 20% accuracy as 6.2926%.
                            
                            
                            model = rf(n_jobs=10,n_estimators=2000,max_depth=25,random_state=10)
                            This model has 20% accuracy as 6.3153%.
                          
                          
# In[24]:


BTENSION_predictlcval_plc2_act_trend  =round(BTENSION_resultsagg['nxt_sales_qty'].sum()/BTENSION_resultsagg['SALES_QTY'].sum() - 1, 3)
print(f'BTENSION_predictlcval_plc2_act_trend is {BTENSION_predictlcval_plc2_act_trend}')  

BTENSION_predictlcval_plc2_pred_trend  =round(BTENSION_resultsagg['pred'].sum()/BTENSION_resultsagg['SALES_QTY'].sum() - 1, 3)
print(f'BTENSION_predictlcval_plc2_pred_trend is {BTENSION_predictlcval_plc2_pred_trend}')  


# In[25]:


BTENSION_resultsagr = BTENSION_resultsagg.copy()

agr1 = BTENSION_resultsagr.groupby(['item','stkpds'])['pred',"SALES_QTY",'nxt_sales_qty'].sum().reset_index().sort_values(['item', 'stkpds'], ascending=True)
agr2 = BTENSION_resultsagr.groupby(['item','stkpds']).size().to_frame('size').reset_index().sort_values(['item', 'stkpds'], ascending=True)

BTENSION_agr = agr2.merge(agr1, on = ['item','stkpds'])
BTENSION_agr = BTENSION_agr.rename({'size':'count'}, axis = 1)
BTENSION_agr['act_trend'] = round(BTENSION_agr['nxt_sales_qty']/BTENSION_agr['SALES_QTY'] - 1, 3)
BTENSION_agr['pred_trend'] = round(BTENSION_agr['pred']/BTENSION_agr['SALES_QTY'] - 1, 3)
BTENSION_agr.to_csv("/rapids/notebooks/my_code/LV_CAT_PLC2/BTENSION_plc2_trend__stk13agr.csv")


# In[26]:


BTENSION_agr


# In[57]:


BTENSION_agr_itemagr = BTENSION_agr.copy()


# In[65]:


BTENSION_agr_itemagr['dif'] = (BTENSION_agr_itemagr['pred'] - BTENSION_agr_itemagr['nxt_sales_qty']).abs()
BTENSION_agr_itemagr['t20percentnxtsale_item'] = BTENSION_agr_itemagr['nxt_sales_qty']*0.2
b_accurate_predict20p_item = BTENSION_agr_itemagr.loc[BTENSION_agr_itemagr['dif'] <= BTENSION_agr_itemagr['t20percentnxtsale_item']]
b_p_t20p_item = round(((b_accurate_predict20p_item.shape[0]/BTENSION_agr_itemagr.shape[0])*100),4)
print(f"within 20% item count is  {b_accurate_predict20p_item.shape[0]}")
print(f"total item count is {BTENSION_agr_itemagr.shape[0]}")
print(f"This model has 20% accuracy as {b_p_t20p_item}%.")


# In[ ]:





# # HOSES  LC plc2 stk13 model

# In[50]:


H_train_lc_val  = pd.read_csv("/rapids/notebooks/team_data/personal/cjackso/FY20P07/HOSES/train_lc_val.csv")
H_predict_lc_val  = pd.read_csv("/rapids/notebooks/team_data/personal/cjackso/FY20P07/HOSES/predict_lc_val.csv")
H_train_lc_val_plc2 = H_train_lc_val.loc[H_train_lc_val['PLCS'] == 2]
H_predict_lc_val_plc2 = H_predict_lc_val.loc[H_predict_lc_val['PLCS'] == 2]
H_train_lc_val_plc2_13 = H_train_lc_val_plc2.loc[H_train_lc_val_plc2['stkpds'] == 13]
H_predict_lc_val_plc2_13 = H_predict_lc_val_plc2.loc[H_predict_lc_val_plc2['stkpds'] == 13]

## Stock 13 prediction model

dat1=H_train_lc_val_plc2_13.copy()
cat_features=["CLIMATE_CODE","adv","ore","hubflag","major_int","MAX_PER_CAR","PLCS","URBAN_RURAL_SEGMENT","fogid","COMM_SALES_FLAG"]
dat2=pd.get_dummies(dat1,columns=cat_features)
feature_cols=list(dat2.columns)
x_train_backup=dat2[feature_cols]
y_train=dat2.nxt_sales_qty
x_train=x_train_backup.drop(['source','item', 'store','SALES_QTY','TRNX','agemax',
 'min_model_yr','nxt_sales_qty','lks_comm_ratio','lks_shift_c2r','APP_LKUPS','lks','lks_Y',
 'HUB_STORE_KEY','mega_store_key','URBAN_RURAL', 'DATE_OPENED','weeks', 'rtl', 
 'open_coeff', 'open_coeff2'],axis=1)



dat3 = H_predict_lc_val_plc2_13.copy()
dat4=pd.get_dummies(dat3, columns=cat_features)
cat_features=["CLIMATE_CODE","adv","ore","hubflag","major_int","MAX_PER_CAR","PLCS","URBAN_RURAL_SEGMENT","fogid","COMM_SALES_FLAG"]
feature_cols1=list(dat4.columns)
x_predict_backup=dat4[feature_cols1]
x_predict = x_predict_backup. drop(['source','item', 'store','SALES_QTY','TRNX','agemax',
 'min_model_yr','nxt_sales_qty','lks_comm_ratio','lks_shift_c2r','APP_LKUPS','lks','lks_Y',
 'HUB_STORE_KEY','mega_store_key','URBAN_RURAL', 'DATE_OPENED','weeks', 'rtl', 
 'open_coeff', 'open_coeff2'],axis=1)


# In[51]:


model = rf(n_jobs=10,random_state=10)
model.fit(x_train, y_train)
preds = model.predict(x_predict)
results = pd.DataFrame(x_predict_backup[["item","store","stkpds","SALES_QTY",'nxt_sales_qty']])
results["pred"] = preds


HOSES_results = results.copy()


HOSES_resultsagg = HOSES_results.copy()

HOSES_resultsagg['dif'] = (HOSES_resultsagg['pred'] - HOSES_resultsagg['nxt_sales_qty']).abs()
HOSES_resultsagg['t20percentnxtsale'] = HOSES_resultsagg['nxt_sales_qty']*0.2
H_accurate_predict20p = HOSES_resultsagg.loc[HOSES_resultsagg['dif'] < HOSES_resultsagg['t20percentnxtsale']]
H_p_t20p = round(((H_accurate_predict20p.shape[0]/HOSES_resultsagg.shape[0])*100),4)
print(f"This model has 20% accuracy as {H_p_t20p}%.")


# In[52]:


print(f"within 20% item_store count is  {H_accurate_predict20p.shape[0]}")
print(f"total item_store count is {HOSES_resultsagg.shape[0]}")
print(f"This model has 20% accuracy as {H_p_t20p}%.")

                        model = rf(n_jobs=10,n_estimators=100,max_depth=6,random_state=10, min_samples_leaf=5) 

                        model = rf(n_jobs=10,n_estimators=100,max_depth=10,random_state=10, min_samples_leaf=5) 
                        This model has 20% accuracy as 1.4976%.

                        model = rf(n_jobs=10,n_estimators=1000,max_depth=10,random_state=10, min_samples_leaf=5) 
                        This model has 20% accuracy as 1.4563%.

                        model = rf(n_jobs=10,n_estimators=1000,max_depth=10,random_state=10, min_samples_leaf=10) 
                        This model has 20% accuracy as 1.2704%.

                        model = rf(n_jobs=10,n_estimators=1000,random_state=10) 
                        This model has 20% accuracy as 4.0694%.

                        model = rf(n_jobs=10,n_estimators=2000,random_state=10) 
                        This model has 20% accuracy as 4.0074%.

                        model = rf(n_jobs=10,n_estimators=3000,random_state=10) 
                        This model has 20% accuracy as 4.0178%.

                        model = rf(n_jobs=10,n_estimators=1500,max_depth=20,random_state=10) 
                        This model has 20% accuracy as 3.8835%.

model = rf(n_jobs=10,random_state=10)
This model has 20% accuracy as 4.3999%.

                        model = rf(random_state=10) 
                        This model has 20% accuracy as 4.3999%.
# In[29]:


HOSES_predictlcval_plc2_act_trend  =round(HOSES_resultsagg['nxt_sales_qty'].sum()/HOSES_resultsagg['SALES_QTY'].sum() - 1, 3)
print(f'HOSES_predictlcval_plc2_act_trend is {HOSES_predictlcval_plc2_act_trend}')  

HOSES_predictlcval_plc2_pred_trend  =round(HOSES_resultsagg['pred'].sum()/HOSES_resultsagg['SALES_QTY'].sum() - 1, 3)
print(f'HOSES_predictlcval_plc2_pred_trend is {HOSES_predictlcval_plc2_pred_trend}')  


# In[30]:


HOSES_resultsagr = HOSES_resultsagg.copy()

agr1 = HOSES_resultsagr.groupby(['item','stkpds'])['pred',"SALES_QTY",'nxt_sales_qty'].sum().reset_index().sort_values(['item', 'stkpds'], ascending=True)
agr2 = HOSES_resultsagr.groupby(['item','stkpds']).size().to_frame('size').reset_index().sort_values(['item', 'stkpds'], ascending=True)

HOSES_agr = agr2.merge(agr1, on = ['item','stkpds'])
HOSES_agr = HOSES_agr.rename({'size':'count'}, axis = 1)
HOSES_agr['act_trend'] = round(HOSES_agr['nxt_sales_qty']/HOSES_agr['SALES_QTY']- 1, 3)
HOSES_agr['pred_trend'] = round(HOSES_agr['pred']/HOSES_agr['SALES_QTY']- 1, 3)
HOSES_agr.to_csv("/rapids/notebooks/my_code/LV_CAT_PLC2/HOSES_plc2_trend_stk13agr.csv")


# In[31]:


HOSES_agr


# In[64]:


HOSES_agr_itemagr = HOSES_agr.copy()

HOSES_agr_itemagr['dif'] = (HOSES_agr_itemagr['pred'] - HOSES_agr_itemagr['nxt_sales_qty']).abs()
HOSES_agr_itemagr['t20percentnxtsale_item'] = HOSES_agr_itemagr['nxt_sales_qty']*0.2
H_accurate_predict20p_item = HOSES_agr_itemagr.loc[HOSES_agr_itemagr['dif'] <= HOSES_agr_itemagr['t20percentnxtsale_item']]
H_p_t20p_item = round(((H_accurate_predict20p_item.shape[0]/HOSES_agr_itemagr.shape[0])*100),4)
print(f"within 20% item count is  {H_accurate_predict20p_item.shape[0]}")
print(f"total item count is {HOSES_agr_itemagr.shape[0]}")
print(f"This model has 20% accuracy as {H_p_t20p_item}%.")


# In[ ]:





# In[ ]:





# # CLUTCHES LC plc2 stk13 model

# In[32]:


train_lc_val  = pd.read_csv("/rapids/notebooks/team_data/personal/cjackso/FY20P07/CLUTCHES/train_lc_val.csv")
predict_lc_val  = pd.read_csv("/rapids/notebooks/team_data/personal/cjackso/FY20P07/CLUTCHES/predict_lc_val.csv")
train_lc_val_plc2 = train_lc_val.loc[train_lc_val['PLCS'] == 2]
train_lc_val_plc2_13 = train_lc_val_plc2.loc[train_lc_val_plc2['stkpds'] == 13]
predict_lc_val_plc2 = predict_lc_val.loc[predict_lc_val['PLCS'] == 2]
predict_lc_val_plc2_13 = predict_lc_val_plc2.loc[predict_lc_val_plc2['stkpds'] == 13]


# In[33]:


dat1=train_lc_val_plc2_13.copy()
cat_features=["CLIMATE_CODE","adv","ore","hubflag","major_int","MAX_PER_CAR","PLCS","URBAN_RURAL_SEGMENT","fogid","COMM_SALES_FLAG"]
dat2=pd.get_dummies(dat1,columns=cat_features)
feature_cols=list(dat2.columns)
x_train_backup=dat2[feature_cols]
y_train=dat2.nxt_sales_qty
x_train=x_train_backup.drop(['source','item', 'store','SALES_QTY','TRNX','agemax',
 'min_model_yr','nxt_sales_qty','lks_comm_ratio','lks_shift_c2r','APP_LKUPS','lks','lks_Y',
 'HUB_STORE_KEY','mega_store_key','URBAN_RURAL', 'DATE_OPENED','weeks', 'rtl', 
 'open_coeff', 'open_coeff2'],axis=1)



dat3 = predict_lc_val_plc2_13.copy()
dat4=pd.get_dummies(dat3, columns=cat_features)
cat_features=["CLIMATE_CODE","adv","ore","hubflag","major_int","MAX_PER_CAR","PLCS","URBAN_RURAL_SEGMENT","fogid","COMM_SALES_FLAG"]
feature_cols1=list(dat4.columns)
x_predict_backup=dat4[feature_cols1]
x_predict = x_predict_backup. drop(['source','item', 'store','SALES_QTY','TRNX','agemax',
 'min_model_yr','nxt_sales_qty','lks_comm_ratio','lks_shift_c2r','APP_LKUPS','lks','lks_Y',
 'HUB_STORE_KEY','mega_store_key','URBAN_RURAL', 'DATE_OPENED','weeks', 'rtl', 
 'open_coeff', 'open_coeff2'],axis=1)


# In[47]:


model = rf(n_jobs=10,random_state=10) 


model.fit(x_train, y_train)
preds = model.predict(x_predict)
results = pd.DataFrame(x_predict_backup[["item","store","stkpds","SALES_QTY",'nxt_sales_qty']])
results["pred"] = preds

CLUTCHES_results = results.copy()

CLUTCHES_resultsagg = CLUTCHES_results.copy()

CLUTCHES_resultsagg['dif'] = (CLUTCHES_resultsagg['pred'] - CLUTCHES_resultsagg['nxt_sales_qty']).abs()
CLUTCHES_resultsagg['t20percentnxtsale'] = CLUTCHES_resultsagg['nxt_sales_qty']*0.2
c_accurate_predict20p = CLUTCHES_resultsagg.loc[CLUTCHES_resultsagg['dif'] < CLUTCHES_resultsagg['t20percentnxtsale']]
c_p_t20p = round(((c_accurate_predict20p.shape[0]/CLUTCHES_resultsagg.shape[0])*100),4)
print(f"This model has 20% accuracy as {c_p_t20p}%.")


# In[48]:


print(f"within 20% item_store count is  {c_accurate_predict20p.shape[0]}")
print(f"total item_store count is {CLUTCHES_resultsagg.shape[0]}")
print(f"This model has 20% accuracy as {c_p_t20p}%.")

model = rf(n_jobs=10,n_estimators=2000,random_state=10)
This model has 20% accuracy as 0.3676%.

model = rf(n_jobs=10,random_state=10) 
This model has 20% accuracy as 0.445%.


model = rf(n_jobs=10,n_estimators=1500,max_depth=20,random_state=10) 
This model has 20% accuracy as 0.3289%.


model = rf(n_jobs=10,n_estimators=100,max_depth=10,random_state=10, min_samples_leaf=5)
This model has 20% accuracy as 0.0145%.
# In[39]:


CLUTCHES_predictlcval_plc2_act_trend  =round(CLUTCHES_resultsagg['nxt_sales_qty'].sum()/CLUTCHES_resultsagg['SALES_QTY'].sum() - 1, 3)
print(f'CLUTCHES_predictlcval_plc2_act_trend is {CLUTCHES_predictlcval_plc2_act_trend}')  

CLUTCHES_predictlcval_plc2_pred_trend  =round(CLUTCHES_resultsagg['pred'].sum()/CLUTCHES_resultsagg['SALES_QTY'].sum() - 1, 3)
print(f'CLUTCHES_predictlcval_plc2_pred_trend is {CLUTCHES_predictlcval_plc2_pred_trend}')  


# In[40]:


CLUTCHES_resultsagr = CLUTCHES_resultsagg.copy()

agr1 = CLUTCHES_resultsagr.groupby(['item','stkpds'])['pred',"SALES_QTY",'nxt_sales_qty'].sum().reset_index().sort_values(['item', 'stkpds'], ascending=True)
agr2 = CLUTCHES_resultsagr.groupby(['item','stkpds']).size().to_frame('size').reset_index().sort_values(['item', 'stkpds'], ascending=True)

CLUTCHES_agr = agr2.merge(agr1, on = ['item','stkpds'])
CLUTCHES_agr = CLUTCHES_agr.rename({'size':'count'}, axis = 1)
CLUTCHES_agr['act_trend'] = round(CLUTCHES_agr['nxt_sales_qty']/CLUTCHES_agr['SALES_QTY']- 1, 3)
CLUTCHES_agr['pred_trend'] = round(CLUTCHES_agr['pred']/CLUTCHES_agr['SALES_QTY'] - 1, 3)
CLUTCHES_agr.to_csv("/rapids/notebooks/my_code/LV_CAT_PLC2/CLUCTHES_plc2_trend_stk13agr.csv")


# In[41]:


CLUTCHES_agr


# In[63]:


CLUTCHES_agr_itemagr = CLUTCHES_agr.copy()

CLUTCHES_agr_itemagr['dif'] = (CLUTCHES_agr_itemagr['pred'] - CLUTCHES_agr_itemagr['nxt_sales_qty']).abs()
CLUTCHES_agr_itemagr['t20percentnxtsale_item'] = CLUTCHES_agr_itemagr['nxt_sales_qty']*0.2
c_accurate_predict20p_item = CLUTCHES_agr_itemagr.loc[CLUTCHES_agr_itemagr['dif'] <= CLUTCHES_agr_itemagr['t20percentnxtsale_item']]
c_p_t20p_item = round(((c_accurate_predict20p_item.shape[0]/CLUTCHES_agr_itemagr.shape[0])*100),4)
print(f"within 20% item count is  {c_accurate_predict20p_item.shape[0]}")
print(f"total item count is {CLUTCHES_agr_itemagr.shape[0]}")
print(f"This model has 20% accuracy as {c_p_t20p_item}%.")


# In[ ]:





# In[ ]:




