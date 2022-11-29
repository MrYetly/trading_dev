import os
from time import sleep
import pandas
import pandas_market_calendars as mcal
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

#parameters
start_date = '2021-05-13'
end_date = '2021-10-08'
max_trading_days = 100
starting_offset =5 
scan_value = '2941408'
file_prefix = 'up 20 OTD'

#find trading days
cal = mcal.get_calendar('NYSE')
bus_days = cal.schedule(start_date = start_date, end_date = end_date)
bus_days = bus_days.sort_values('market_open', ascending = False)
bus_days = bus_days.reset_index()
bus_days = bus_days.loc[:(max_trading_days-1),:]
bus_days = bus_days.rename(columns={'index':'date'})
bus_days['count'] = bus_days.index + starting_offset

#set directory to downloads
os.chdir('/users/ivananich/downloads')

#open webdriver
driver = webdriver.Chrome()

driver.get('https://stockcharts.com/login/')

#log in
email_field = driver.find_element(By.NAME, "form_UserID")
email_field.send_keys('ivanpiace123@yahoo.com')
pw_field = driver.find_element(By.NAME, 'form_UserPassword')
pw_field.send_keys('HANDOFF-KICK-492', Keys.ENTER)

#navigate to advanced scan work bench
driver.get('https://stockcharts.com/def/servlet/ScanUI')

#select scan
scans = driver.find_element(By.ID, 'favScans')
select_scan = Select(scans)
select_scan.select_by_value(scan_value)

name_change_dic = {}
downloaded = 0
for i, row in bus_days.iterrows():

    #input date offset
    offset = driver.find_element(By.ID, 'globalDateOffset')
    offset.clear()
    offset.send_keys(row['count'], Keys.ENTER)

    #run scan
    run_scan = driver.find_element(By.ID, 'runScan')
    run_scan.click()

    #switch tabs
    scan_tab = driver.window_handles[0]
    result_tab = driver.window_handles[1]
    driver.switch_to.window(result_tab)

    #download scan
    #check if results are zero
    n_results = driver.find_element(By.ID, 'resultcount')
    n_results = n_results.text[18:]
    n_results = int(n_results)

    if n_results != 0:
        dl_result = driver.find_element(By.ID, 'selectResult')
        select_csv = Select(dl_result)
        select_csv.select_by_value('csv')

        if downloaded == 0:
            f_name = 'SC.csv'
        else:
            f_name = f'SC ({downloaded}).csv'

        new_name = f'{file_prefix}_{str(row.date)[:-9]}.csv'
        name_change_dic[f_name] = new_name
        
        if len(name_change_dic) == 2:
            sleep(5)
        driver.switch_to.window(scan_tab)
        downloaded +=1
    else:
        driver.switch_to.window(scan_tab)

print('waiting for lagging downloads')
sleep(5)
driver.quit()

for old, new in name_change_dic.items():
    os.rename(old, new)
