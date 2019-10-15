
# coding: utf-8

# In[ ]:

# Libraries
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import pearsonr


# In[ ]:

class StockPriceForecasting:
    
    def __init__(self,stock_prices_file_location):
        self.__historical_quotes_df = pd.read_csv(stock_prices_file_location)
        self.__original_prices_df = pd.DataFrame()
        self.__first_dates_df = self.__historical_quotes_df.loc[:,('date','close')]
        self.__first_dates_df.date = self.__first_dates_df.date.apply(pd.to_datetime)
        self.__first_dates_df = self.__first_dates_df[( self.__first_dates_df.date > '2016-10-31') & ( self.__first_dates_df.date < '2017-07-01')]
        # Choose 8 required dates
        self.__first_dates_df = self.__first_dates_df.sort_values(by = 'date')
        dates = pd.DatetimeIndex(self.__first_dates_df.date)
        current_month = 0
        first_trading_dates = []

        for i in range(len(dates)):
            if dates[i].month != current_month:
                first_trading_dates.append(dates[i])
                current_month = dates[i].month

        # Prepare result data frame        
        self.__first_dates_df = self.__first_dates_df[self.__first_dates_df.date.isin(first_trading_dates)]
        self.__first_dates_df = self.__first_dates_df.close
        
    def saveFirstDates(self, file_name):
        """This fuction saves in file close prices of first trading days for 8 months"""
        
        self.__first_dates_df.to_csv(file_name, index=False, sep='\t', encoding='utf-8')
    
    def executeExponentialSmoothing(self,file_name):
        """The fuction reads stock prices from a file and executes exponential smoothing"""
        
        self.__original_prices_df = pd.read_csv(file_name, header = None, names = ['Price'])
        
        # Alpha input validation
        user_answer = 'y'
        while(user_answer == 'y'):

            while(True):
                try:
                    alpha = float(input("Please enter alpha: "))
                except ValueError:
                    print("Please enter a number")
                    continue
                else:
                    if (alpha < 0 or alpha > 1):
                        print('Alpha must be within [0,1] interval') 
                        continue
                    else:
                        break

            predicted_prices = []
            for i in range(len(self.__original_prices_df) + 1):
                if i == 0:
                    smoothed_value = self.__original_prices_df.Price[0]
                else:
                    smoothed_value = alpha * self.__original_prices_df.Price[i-1] + (1 - alpha) * predicted_prices[i-1]

                predicted_prices.append(smoothed_value)

            df_smoothed = pd.DataFrame()
            df_smoothed['Price'] = predicted_prices
            self.__original_prices_df.plot(title = 'Original Prices',legend = True)
            plt.show()
            df_smoothed.plot(title = 'Forecasted Prices',legend = True)
            plt.show()
            
            print("Forecasted price is:", predicted_prices[8])
            answer = input("Do you want to change alpha?('y'/'n')")
            if answer != 'y':
                print('Exponential smoothing successfully terminated')
                break
    
    def executeLinearRegression(self, file_name):
        """The fuction reads stock prices from a file and executes linear regression"""
        
        self.__original_prices_df = pd.read_csv(file_name, header = None, names = ['Price'])
        original_prices = self.__original_prices_df.Price.values
        df_lm = pd.DataFrame()
        df_lm['price'] = original_prices
        df_lm['lag_price'] = df_lm.price.shift(1)
        df_lm.dropna(how = 'any')
        # Manual linear regression implementation
        original_prices = df_lm.price.values
        original_prices = original_prices[1:]
        print("original prices:",original_prices)
        lag_prices = df_lm.lag_price.values
        lag_prices = lag_prices[1:]
        print("lag_prices:",lag_prices)
        # Mean Y
        mean_original = np.mean(original_prices)
        print("mean original:",mean_original)
        # Mean X
        mean_lag_price = np.mean(lag_prices)
        print("mean lag:",mean_lag_price)
        # Standard deviation original
        original_stdev = np.std(original_prices)
        print("original st. dev:", original_stdev)
        # Standard deviation lag
        lag_stdev = np.std(lag_prices)
        print("lag st. dev:", lag_stdev)
        #Sample Linear Correlation Coefficient
        corr_coeff = pearsonr(original_prices,lag_prices)
        print("corr coeff:", corr_coeff[0])
        # Slope
        slope = corr_coeff[0] * original_stdev / lag_stdev
        intercept = mean_original - slope * mean_lag_price
        # Use function
        lm = smf.ols(formula='price ~ lag_price', data=df_lm).fit()
        
        print("Calculated Slope:", slope)
        print("Build-in function Slope:",lm.params.values[1])
        print("Calculated Intercept:", intercept)
        print("Build-in function Intercept:",lm.params.values[0])
        
        # Forecasted price
        forecasted_price = intercept + slope * original_prices[6]
        print("forecasted_price 9 time period:",forecasted_price)

