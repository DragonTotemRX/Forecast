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
    os.system('python '+my_code_loc+'ngf_step1_CB.py '+FOGID+' '+FP+' &')
    time.sleep(20)
    print('total ps='+str(len(psutil.pids())))
    print('mem pct='+str(psutil.virtual_memory().percent))
    
kick_off('ACCOND','FY20P11') 
kick_off('AIRCOND','FY20P11') 
kick_off('AIRMANT','FY20P11') 
kick_off('ALTRNTR','FY20P11') 
kick_off('BEARSEAL','FY20P11') 
kick_off('BELTS','FY20P11') 
kick_off('BOOSTERS','FY20P11') 
kick_off('BRAKHDWE','FY20P11') 
kick_off('BRAKHOSE','FY20P11') 
kick_off('BRAKPADS','FY20P11') 
kick_off('BRAKSHOE','FY20P11') 
kick_off('BTENSION','FY20P11') 
kick_off('CALIPERS','FY20P11') 
kick_off('CARBKITS','FY20P11') 
kick_off('CLUTCHAC','FY20P11') 
kick_off('CLUTCHES','FY20P11') 
kick_off('COILS','FY20P11') 
kick_off('COLISION','FY20P11') 
kick_off('CONVERTR','FY20P11') 
kick_off('CVSHAFTS','FY20P11') 
kick_off('DISTRIB','FY20P11') 
kick_off('DRAGLINK','FY20P11') 
kick_off('DRUMS','FY20P11') 
kick_off('ENGCOMPS','FY20P11') 
kick_off('ENGMGMT','FY20P11') 
kick_off('ENGPARTS','FY20P11') 
kick_off('ENGTRANS','FY20P11') 
kick_off('FANASSY','FY20P11') 
kick_off('FANCLUCH','FY20P11') 
kick_off('FANLRGBX','FY20P11') 
kick_off('FUELFILT','FY20P11') 
kick_off('FUELIJO2','FY20P11') 
kick_off('FUELPUMP','FY20P11') 
kick_off('FUELTANK','FY20P11') 
kick_off('FUELNECK','FY20P11') 
kick_off('GASCAPS','FY20P11') 
kick_off('GASKETS','FY20P11') 
kick_off('GASKMISC','FY20P11') 
kick_off('HANGEXH','FY20P11') 
kick_off('HEATING','FY20P11') 
kick_off('HELP','FY20P11') 
kick_off('HITCHES','FY20P11') 
kick_off('HOSES','FY20P11') 
kick_off('HUBBRNG','FY20P11') 
kick_off('IGNITION','FY20P11') 
kick_off('INTHOSE','FY20P11') 
kick_off('INTRCOOL','FY20P11') 
kick_off('LOCKWORK','FY20P11') 
kick_off('MANUALS','FY20P11') 
kick_off('MASTRCYL','FY20P11') 
kick_off('MIRGLASS','FY20P11') 
kick_off('MISCHOSE','FY20P11') 
kick_off('MMOUNTS','FY20P11') 
kick_off('MTYLIFT','FY20P11') 
kick_off('MUFFLERS','FY20P11') 
kick_off('OILCAPS','FY20P11') 
kick_off('OILPANS','FY20P11') 
kick_off('PLUGS','FY20P11') 
kick_off('PLUGWIRE','FY20P11') 
kick_off('PSTEER','FY20P11') 
kick_off('RACKPIN','FY20P11') 
kick_off('RADCAPS','FY20P11') 
kick_off('RADIATOR','FY20P11') 


print('All fogid were kicked off')
print('total ps='+str(len(psutil.pids())))
print('mem pct='+str(psutil.virtual_memory().percent))
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))