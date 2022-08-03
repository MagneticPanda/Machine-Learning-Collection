"""
Name: Sashen Moodley
Student Number: 219006946
"""

from selenium.webdriver.common.by import By


class TurnipPage:
    # Locators
    SEARCH_BUTTON = 'button#generate-seed'
    BUYING_PRICE = 'buying-price'
    NEXT_PATTERN = 'next-pattern'
    MON_AM = 'selling-price-0'
    MON_PM = 'selling-price-1'
    TUES_AM = 'selling-price-2'
    TUES_PM = 'selling-price-3'
    WED_AM = 'selling-price-4'
    WED_PM = 'selling-price-5'
    THURS_AM = 'selling-price-6'
    THURS_PM = 'selling-price-7'
    FRI_AM = 'selling-price-8'
    FRI_PM = 'selling-price-9'
    SAT_AM = 'selling-price-10'
    SAT_PM = 'selling-price-11'

    def __init__(self, browser):
        self.browser = browser

    @property
    def search_button(self):
        return self.browser.find_element(By.CSS_SELECTOR, 'button#generate-seed')

    @property
    def buying_price(self):
        return self.browser.find_element(By.ID, 'buying-price').text

    @property
    def pattern(self):
        return self.browser.find_element(By.ID, 'next-pattern').text

    @property
    def mon_am(self):
        return self.browser.find_element(By.ID, 'selling-price-0').text

    @property
    def mon_pm(self):
        return self.browser.find_element(By.ID, 'selling-price-1').text

    @property
    def tues_am(self):
        return self.browser.find_element(By.ID, 'selling-price-2').text

    @property
    def tues_pm(self):
        return self.browser.find_element(By.ID, 'selling-price-3').text

    @property
    def wed_am(self):
        return self.browser.find_element(By.ID, 'selling-price-4').text

    @property
    def wed_pm(self):
        return self.browser.find_element(By.ID, 'selling-price-5').text

    @property
    def thurs_am(self):
        return self.browser.find_element(By.ID, 'selling-price-6').text

    @property
    def thurs_pm(self):
        return self.browser.find_element(By.ID, 'selling-price-7').text

    @property
    def fri_am(self):
        return self.browser.find_element(By.ID, 'selling-price-8').text

    @property
    def fri_pm(self):
        return self.browser.find_element(By.ID, 'selling-price-9').text

    @property
    def sat_am(self):
        return self.browser.find_element(By.ID, 'selling-price-10').text

    @property
    def sat_pm(self):
        return self.browser.find_element(By.ID, 'selling-price-11').text

    def get_elements(self):
        if '0' in self.pattern:
            pattern = 'Fluctuating'
        elif '1' in self.pattern:
            pattern = 'High Spike'
        elif '2' in self.pattern:
            pattern = 'Decreasing'
        elif '3' in self.pattern:
            pattern = 'Small Spike'
        return [self.buying_price, pattern, self.mon_am, self.mon_pm, self.tues_am, self.tues_pm, self.wed_am,
                self.wed_pm, self.thurs_am, self.thurs_pm, self.fri_am, self.fri_pm, self.sat_am, self.sat_pm]
