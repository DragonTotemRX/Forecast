import os
import sys
import psutil
import time

max_mem=70

print('NGF Nvidia Training and Predicting')
print('Step 1) Run train and predict - BEGIN')
print('Arguments are [fog] and [period]'+str(sys.argv))

my_code_loc = '/rapids/notebooks/my_code/'

while psutil.virtual_memory().percent>max_mem: 
    print('total ps='+str(len(psutil.pids())))
    print('mem pct='+str(psutil.virtual_memory().percent))
    time.sleep(120)
os.system('{ python '+my_code_loc+'RF_HI_v2.py '+sys.argv[1]+' '+sys.argv[2]+' M1_P ; } > '+my_code_loc+'log/'+sys.argv[1]+'_rf_hi_stocked.log 2>&1 &')
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
time.sleep(60)

while psutil.virtual_memory().percent>max_mem: 
    print('total ps='+str(len(psutil.pids())))
    print('mem pct='+str(psutil.virtual_memory().percent))
    time.sleep(120)
os.system('{ python '+my_code_loc+'RF_HI_v2.py '+sys.argv[1]+' '+sys.argv[2]+' M2_P ; } > '+my_code_loc+'log/'+sys.argv[1]+'_rf_hi_full.log 2>&1 &')
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
time.sleep(60)

while psutil.virtual_memory().percent>max_mem: 
    print('total ps='+str(len(psutil.pids())))
    print('mem pct='+str(psutil.virtual_memory().percent))
    time.sleep(120)
os.system('{ python '+my_code_loc+'RF_LC_v2.py '+sys.argv[1]+' '+sys.argv[2]+' M1_P ; } > '+my_code_loc+'log/'+sys.argv[1]+'_rf_lc_stocked.log 2>&1 &')
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
time.sleep(60)

while psutil.virtual_memory().percent>max_mem: 
    print('total ps='+str(len(psutil.pids())))
    print('mem pct='+str(psutil.virtual_memory().percent))
    time.sleep(120)
os.system('{ python '+my_code_loc+'RF_LC_v2.py '+sys.argv[1]+' '+sys.argv[2]+' M2_P ; } > '+my_code_loc+'log/'+sys.argv[1]+'_rf_lc_full.log 2>&1 &')
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
time.sleep(60)

while psutil.virtual_memory().percent>max_mem: 
    print('total ps='+str(len(psutil.pids())))
    print('mem pct='+str(psutil.virtual_memory().percent))
    time.sleep(60)
os.system('{ python '+my_code_loc+'CB_v2.py '+sys.argv[1]+' '+sys.argv[2]+' M1_P ; } > '+my_code_loc+'log/'+sys.argv[1]+'_cb_stocked.log 2>&1 &')
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
time.sleep(120)

while psutil.virtual_memory().percent>max_mem: 
    print('total ps='+str(len(psutil.pids())))
    print('mem pct='+str(psutil.virtual_memory().percent))
    time.sleep(60)
os.system('{ python '+my_code_loc+'CB_v2.py '+sys.argv[1]+' '+sys.argv[2]+' M2_P ; } > '+my_code_loc+'log/'+sys.argv[1]+'_cb_full.log 2>&1 &')
print('clock='+time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))

print('NGF Step 1 Kicked Off')
