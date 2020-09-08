import os
import sys
import psutil
import time

print('NGF Nvidia Training and Predicting')
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))

max_ps_inc=100
max_mem_inc=40
max_mem=70

p0 = len(psutil.pids())
m0 = psutil.virtual_memory().percent
print('initial ps='+str(p0))
print('initial mem pct='+str(m0))

my_code_loc = '/rapids/notebooks/my_code/'

def kick_off(FOGID, FP):
    while len(psutil.pids())-p0>=max_ps_inc or psutil.virtual_memory().percent-m0>max_mem_inc or psutil.virtual_memory().percent>max_mem: 
        print('total ps='+str(len(psutil.pids())))
        print('mem pct='+str(psutil.virtual_memory().percent))
        time.sleep(120)
    os.system('python '+my_code_loc+'ngf_step1.py '+FOGID+' '+FP+' &')
    time.sleep(20)
    print('total ps='+str(len(psutil.pids())))
    print('mem pct='+str(psutil.virtual_memory().percent))
    
kick_off('REARAXLE','FY20P11') 
kick_off('RELAYCON','FY20P11') 
kick_off('SHOCKS','FY20P11') 
kick_off('SHOCK48','FY20P11') 
kick_off('STARTERS','FY20P11') 
kick_off('SUSPENSN','FY20P11') 
kick_off('SUSPLGBX','FY20P11') 
kick_off('SWITCHES','FY20P11') 
kick_off('TOWCONN','FY20P11') 
kick_off('TRANSFIL','FY20P11') 
kick_off('TRANSKIT','FY20P11') 
kick_off('TRNSFLAT','FY20P11') 
kick_off('TSTAT','FY20P11') 
kick_off('TURBODSL','FY20P11') 
kick_off('UJOINTS','FY20P11') 
kick_off('WATRPUMP','FY20P11') 
kick_off('WHEELHDW','FY20P11') 
kick_off('WINLIFT','FY20P11') 
kick_off('WIPMTR','FY20P11') 
 


print('All fogid were kicked off')
print('total ps='+str(len(psutil.pids())))
print('mem pct='+str(psutil.virtual_memory().percent))
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))