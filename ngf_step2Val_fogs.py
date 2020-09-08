import os
import sys
import psutil
import time

print('NGF Nvidia Training and Predicting - Combine Models')
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))

max_ps_inc=20
max_mem_inc=50
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
        time.sleep(10)
    os.system('python '+my_code_loc+'combine_results.py '+FOGID+' '+FP+' Val &')
    time.sleep(2)
    print('total ps='+str(len(psutil.pids())))
    print('mem pct='+str(psutil.virtual_memory().percent))
    
kick_off('ALTRNTR','FY20P11') 
kick_off('BOOSTERS','FY20P11') 
kick_off('BRAKPADS','FY20P11') 
kick_off('DRUMS','FY20P11') 
kick_off('ENGMGMT','FY20P11') 
kick_off('ENGPARTS','FY20P11') 
kick_off('FUELFILT','FY20P11') 
kick_off('FUELPUMP','FY20P11') 
kick_off('GASKETS','FY20P11') 
kick_off('IGNITION','FY20P11') 
kick_off('MASTRCYL','FY20P11') 
kick_off('SHOCKS','FY20P11') 
kick_off('SHOCK48','FY20P11') 
kick_off('STARTERS','FY20P11') 
kick_off('SUSPENSN','FY20P11') 

print('All fogid were kicked off')
print('total ps='+str(len(psutil.pids())))
print('mem pct='+str(psutil.virtual_memory().percent))
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))