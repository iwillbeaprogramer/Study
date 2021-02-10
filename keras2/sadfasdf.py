from selenium import webdriver
from bs4 import BeautifulSoup
import time
import re 

driver = webdriver.Chrome()
driver.get('https://search.naver.com/search.naver?where=image&sm=tab_jum&query=%EC%95%84%EB%A9%94%EB%A6%AC%EC%B9%B4%EB%85%B8#')
time.sleep(5)
driver.find_element_by_xpath("//*[@id=\"main_pack\"]/section/div[2]/div[1]/div[1]/div[1]/div/div[1]/a").click()
driver.find_element_by_xpath("//*[@id=\"main_pack\"]/section/div[2]/div[2]/div/div[1]/div[1]/div[1]/div/div/div[1]/div[1]/img").