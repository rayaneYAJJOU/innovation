#Importing necessary libraries
import streamlit as st
import sys
import pandas as pd
import quandl
import numpy as np
import matplotlib.pyplot as plt
from Source.exception import CustomException
from Source.logger import logging

#Title
st.title('Portfolio Optimization Tool')
st.write('1) Select the dates')
st.write('2) Enter tickers')

# Take user input for number of stocks
num_stocks = st.number_input('Enter number of stocks:',value = 0)

# Take user input for stock tickers
stocks = []
for i in range(num_stocks):
    stock = st.text_input(f'Enter stock ticker {i+1}:')
    stocks.append(stock.upper()) 
start_date = st.sidebar.date_input('Start date') #start date of analysis
end_date = st.sidebar.date_input('End date') #end date of analysis

def download(stocks): #Function to download stock data
    try:
        quandl.ApiConfig.api_key = 'wzJDT7oCsuxArANyYHcD' #API key
        data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                            qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                            date = { 'gte': start_date, 'lte': end_date }, paginate=True)#Download data
        data = data.dropna() #drop na values
        df = data.set_index('date') 
        table = df.pivot(columns='ticker') #pivot table from dataframe
        table.columns = [col[1] for col in table.columns]
        logging.debug("Downloaded data") #log event
    except Exception as e:
        raise CustomException(e,sys) #raise exception is arises
    return table 
 
def plot_download(stocks): #function to plot downloaded data
    try:
        table = download(stocks) #get the pivot table
        fig = plt.figure(figsize=(14, 7))  #plotting code
        for c in table.columns.values:
            plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c)
        plt.legend(loc='upper left', fontsize=12)
        plt.ylabel('price in $')
        st.pyplot(fig) #Display in webapp
        logging.debug('Data Plotted') #log event
    except Exception as e: 
        raise CustomException(e,sys) #raise exception if arises

def plot_returns(stocks): #function to plot stock returns
    try:
        table = download(stocks)
        returns = table.pct_change() #calculate returns
        fig = plt.figure(figsize=(14, 7)) #plotting code
        for c in returns.columns.values:
            plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
        plt.legend(loc='upper right', fontsize=12)
        plt.ylabel('daily returns')
        st.pyplot(fig) #display in webapp
        logging.debug('Returns plotted') #log event
    except Exception as e:
        raise CustomException(e,sys) #raise exception if arises
    
# This helper function calculates the annualized portfolio return and standard deviation 
# based on the given portfolio weights, mean returns, and covariance matrix of returns
def portfolio_annualised_performance(weights, mean_returns, cov_matrix): 
    returns = np.sum(mean_returns*weights ) *252 #252 trading days assumed
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

# This function simulates a given number of random portfolios and returns their performance metrics 
# (standard deviation, returns, and Sharpe ratio) and also records their weights
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate): 
    results = np.zeros((3,num_portfolios)) #initialize a 3 x num_portfolios matrix for storing results
    weights_record = [] #empty list to store weights record for each portfolio
    
    for i in range(num_portfolios): 
        # Generate random portfolio weights that sum to 1
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        
        # Record the weights for the current portfolio
        weights_record.append(weights)
        
        # Calculate the portfolio's annualized standard deviation, returns, and Sharpe ratio
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
    return results, weights_record


def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    '''This is a Python function that uses the mean returns and covariance matrix of a set of assets to simulate a
      number of portfolios with random weights, calculate their returns and volatility, and plot them on a scatter plot.
        It also identifies the portfolio with the maximum Sharpe ratio (the ratio of return to volatility) and the portfolio 
        with the minimum volatility, and prints their allocations and performance metrics.'''
    try:
        results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
        
        max_sharpe_idx = np.argmax(results[2])
        sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
        max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['Allocation'])
        max_sharpe_allocation.Allocation = [round(i*100,2)for i in max_sharpe_allocation.Allocation]
        max_sharpe_allocation = max_sharpe_allocation.T
        
        min_vol_idx = np.argmin(results[0])
        sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
        min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['Allocation'])
        min_vol_allocation.Allocation = [round(i*100,2)for i in min_vol_allocation.Allocation]
        min_vol_allocation = min_vol_allocation.T
        
        st.write("-"*80)
        st.write("Maximum Sharpe Ratio Portfolio Allocation\n")
        st.write("Annualised Return:", round(rp,2))
        st.write("Annualised Volatility:", round(sdp,2))
        st.write("\n")
        st.write(max_sharpe_allocation)
        st.write("-"*80)
        st.write("Minimum Volatility Portfolio Allocation\n")
        st.write("Annualised Return:", round(rp_min,2))
        st.write("Annualised Volatility:", round(sdp_min,2))
        st.write("\n")
        st.write(min_vol_allocation)
        
        fig = plt.figure(figsize=(10, 7))
        plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
        plt.colorbar()
        plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
        plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
        plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
        plt.xlabel('annualised volatility')
        plt.ylabel('annualised returns')
        plt.legend(labelspacing=0.8)
        st.pyplot(fig)
        logging.debug('Allocations done')
    except Exception as e:
        raise CustomException(e,sys)
table = download(stocks)
def app():
    '''This code defines a function app() that is called when the script is executed as the 
    main program'''
    if num_stocks:
        st.subheader('Plot shows how stock prices have evolved in the given time frame')
        plot_download(stocks)
        st.subheader('Plot of daily returns to see volatility')
        plot_returns(stocks)
        returns = table.pct_change()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_portfolios = 25000
        risk_free_rate = 0.0178
        display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
    else:
        st.subheader('Enter your stock tickers') 
if __name__ == "__main__":
    app()
