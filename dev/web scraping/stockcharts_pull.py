import os
import pandas
import pandas_market_calendars as mcal
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

#find trading days
start_date = '2021-01-23'
end_date = '2021-07-15'
cal = mcal.get_calendar('NYSE')
bus_days = cal.schedule(start_date = start_date, end_date = end_date)
bus_days.reset_index()
#bus_days['count'] = bus_days.index + 6
print(bus_days)

#set directory to downloads
os.chdir('/users/ivananich/downloads')

#open webdriver
driver = webdriver.Chrome()

driver.get('https://stockcharts.com/login/')

#log in
email_field = driver.find_element(By.NAME, "form_UserID")
email_field.send_keys('mdd02018@gmail.com')
pw_field = driver.find_element(By.NAME, 'form_UserPassword')
pw_field.send_keys('Test123', Keys.ENTER)

#navigate to advanced scan work bench
driver.get('https://stockcharts.com/def/servlet/ScanUI')

#select scan
scans = driver.find_element(By.ID, 'favScans')
select_scan = Select(scans)
select_scan.select_by_value('2416747')

#input date offset
offset = driver.find_element(By.ID, 'globalDateOffset')
offset.clear()
offset.send_keys('100', Keys.ENTER)

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
    os.rename('SC.csv','test.csv')

#driver.quit()
