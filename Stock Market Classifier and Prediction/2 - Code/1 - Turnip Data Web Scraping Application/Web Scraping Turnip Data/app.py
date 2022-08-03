"""
Name: Sashen Moodley
Student number: 219006946
"""
import pandas as pd
from selenium import webdriver
from turnip_price_page import TurnipPage

dataSet = []

chrome = webdriver.Chrome(executable_path="D:\sashe\Downloads\chromedriver_win32_100\chromedriver.exe")  # <--- Change accordingly
chrome.get("https://turnip-price.vercel.app/")

page = TurnipPage(chrome)

page.search_button.click()
for _ in range(10000):
    dataSet.append(page.get_elements())
    page.search_button.click()

header = ['Buying Price', 'Pattern', 'Mon_AM', 'Mon_PM', 'Tues_AM', 'Tues_PM', 'Wed_AM', 'Wed_PM','Thurs_AM', 'Thurs_PM', 'Fri_AM', 'Fri_PM', 'Sat_AM', 'Sat_PM']
data = pd.DataFrame(dataSet, columns=header)
data.to_csv('TurnipsDS.csv', index=False)
