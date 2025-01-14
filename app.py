#Importing necessary libraries
import streamlit as st
import sys
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from Source.exception import CustomException
from Source.logger import logging
from math import comb

def bar():
    st.write("-"*80)

#Title
st.title('RiskLESS')
bar()

st.sidebar.image("logo.png")

st.sidebar.write('1) Select the dates')
start_date = st.sidebar.date_input('Start date') #start date of analysis
end_date = st.sidebar.date_input('End date') #end date of analysis

st.sidebar.write('2) Enter total number of stocks')
num_stocks = st.sidebar.number_input('',value = 0)

# Take user input for stock tickers
stocks = []

for i in range(num_stocks):
    stock = st.sidebar.text_input(f'Enter stock ticker {i+1}:')
    stocks.append(stock.upper())

st.sidebar.write("Ou charger un fichier en format csv")
data_uploaded = st.sidebar.file_uploader("Upload stock data")

st.sidebar.write('3) Enter number of stocks in basket')
num_basket = st.sidebar.number_input('', value = 0, max_value = num_stocks)

st.sidebar.write('4) Enter the investor\'s initial capital')
initial_capital = st.sidebar.number_input('Initial Capital ($)', min_value=1, value=100000)  # Default value of 100,000$

def comb_list(arr, k):
    n = len(arr)

    cnk = comb(n, k)

    combs = []

    ptr = 0

def download(stocks):  # Function to download stock data
    try:
        print(f"Downloading data for stocks: {stocks}")
        
        # Download data using yfinance
        data = yf.download(stocks, start=start_date, end=end_date, progress=False)

        print(data)
        
        if data.empty:
            print("No data returned. Check stock tickers or network connection.")
            return None
        
        # Keep only 'Adj Close' column
        print([d[1] for d in data.columns])
        data = data[[('Close', f'{stk}') for stk in stocks]]
        # Drop NA values
        data = data.dropna()
        print("Data shape after download:", data.shape)
        table = data.copy()
        table.columns = [[d[1] for d in table.columns]]
        # Reshape data to match the pivot table logic
        logging.debug("Downloaded and processed data")  # Log event
        return table
    except Exception as e:
        raise CustomException(e, sys) 


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

def get_correlation_matrix(cov_matrix):
    variances = np.diag(cov_matrix)

    # 2. Calculer les racines carrées des variances
    std_devs = np.sqrt(variances)

    # 3. Normaliser la matrice de covariance pour obtenir la matrice de corrélation
    corr_mat = cov_matrix / (std_devs[:, None] * std_devs[None, :])

    # Convertir en un numpy array si vous voulez
    corr_mat = corr_mat.to_numpy()
    
    return corr_mat

def show_correlation_matrix(cov_matrix, corr_mat):
    st.write("Correlation matrix:")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=cov_matrix.columns, yticklabels=cov_matrix.columns)
    st.pyplot(plt)

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate,table):
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

        bar()
        st.write("Maximum Sharpe Ratio Portfolio Allocation\n")
        st.write("Annualised Return:", round(rp,2))
        st.write("Annualised Volatility:", round(sdp,2))
        st.write("\n")
        #st.write(max_sharpe_allocation)

        fig, ax = plt.subplots()
        ax.pie(max_sharpe_allocation.loc["Allocation"].values, labels = max_sharpe_allocation.columns, autopct='%1.1f%%', startangle=90)
        ax.legend(max_sharpe_allocation.columns, title="Composants", loc="upper left", bbox_to_anchor=(1, 1))

        # Assurer que le graphique soit circulaire
        ax.axis('equal')

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)

        bar()
        st.write("Minimum Volatility Portfolio Allocation\n")
        st.write("Annualised Return:", round(rp_min,2))
        st.write("Annualised Volatility:", round(sdp_min,2))
        st.write("\n")

        fig, ax = plt.subplots()
        ax.pie(min_vol_allocation.loc["Allocation"].values, labels = min_vol_allocation.columns, autopct='%1.1f%%', startangle=90)
        ax.legend(min_vol_allocation.columns, title="Composants", loc="upper left", bbox_to_anchor=(1, 1))

        # Assurer que le graphique soit circulaire
        ax.axis('equal')

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)

        bar()
        
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

        return max_sharpe_allocation
    except Exception as e:
        raise CustomException(e,sys)

def sortino_ratio(returns, risk_free_rate=0.0178, target_return=0):
    """
    Cette fonction calcule le ratio de Sortino pour un portefeuille.
    
    :param returns: Liste ou tableau des rendements du portefeuille
    :param risk_free_rate: Taux sans risque (par défaut 1.78%)
    :param target_return: Rendement cible (par défaut 0, ce qui signifie aucun rendement cible)
    :return: Le ratio de Sortino
    """
    # Filtrer les rendements négatifs (en dessous du seuil ou de la moyenne)
    downside_returns = returns[returns < target_return]
    
    # Calculer la deviation négative (downside deviation)
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        # Si la downside deviation est nulle (pas de rendements en dessous du seuil), on retourne une valeur très élevée pour le ratio
        return np.inf
    
    # Calculer le rendement moyen du portefeuille
    portfolio_return = np.mean(returns)
    
    # Calculer le ratio de Sortino
    sortino = (portfolio_return - risk_free_rate) / downside_deviation
    
    return sortino

def monte_carlo_simulation(optimal_weights, risk_free_rate=0.0178, num_simulations=10000, num_days=252):
    """
    Effectue une simulation Monte Carlo pour évaluer un portefeuille d'actifs optimisé selon le modèle de Markowitz.
    
    :param num_simulations: Nombre de simulations Monte Carlo à effectuer
    :param num_days: Nombre de jours de trading simulés (par défaut 252 jours par an)
    
    :return: Dictionnaire avec les résultats des simulations (performance, volatilité, VaR)
    """

    # Téléchargement des données historiques
    data = yf.download(stocks, start=start_date, end=end_date)["Close"]
    
    # Calcul des rendements quotidiens
    returns = data.pct_change().dropna()
    
    # Calcul des rendements moyens et de la matrice de covariance
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Simulation des trajectoires de prix
    simulated_prices = np.zeros((num_simulations, num_days, len(stocks)))
    portfolio_returns = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        # Générer des rendements aléatoires basés sur la distribution des rendements historiques
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
        
        # Initialiser les prix simulés avec les derniers prix connus
        simulated_prices[i, 0, :] = data.iloc[-1].values
        
        for t in range(1, num_days):
            # Calculer le prix en tenant compte du rendement quotidien
            simulated_prices[i, t, :] = simulated_prices[i, t-1, :] * (1 + daily_returns[t])  # Appliquer le rendement pour chaque actif
        
        # Calculer les rendements du portefeuille pour chaque simulation
        weighted_returns = np.dot(daily_returns, optimal_weights)  # Rendement total du portefeuille par jour
        portfolio_returns[i] = np.sum(weighted_returns)  # Calculer le rendement global du portefeuille

    # Calcul des métriques du portefeuille
    mean_portfolio_return = np.mean(portfolio_returns)
    portfolio_volatility = np.std(portfolio_returns)
    VaR_95 = np.percentile(portfolio_returns, 5)

    discount_factor = 1 / ((1 + risk_free_rate) ** (num_days / 252))  # Actualisation pour l'année (252 jours)

    final_value = initial_capital * (1 + np.mean(portfolio_returns))  # Multiplication du capital initial par les rendements
    present_value = final_value * discount_factor

    sortino = sortino_ratio(portfolio_returns)

    # Visualisation de la distribution des rendements du portefeuille
    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_returns, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution des rendements du portefeuille simulé')
    plt.xlabel('Rendement')
    plt.ylabel('Fréquence')
    plt.grid(True)
    
    # Affichage avec Streamlit
    st.pyplot(plt)
    
    results = {
        'mean_portfolio_return': mean_portfolio_return,
        'portfolio_volatility': portfolio_volatility,
        'VaR_95': VaR_95,
        'present_value': present_value,
        'final_value': final_value,
        'sortino_ratio': sortino
    }

    return results

def app():
    '''This code defines a function app() that is called when the script is executed as the 
    main program'''
    if num_stocks>1:

        st.subheader('Plot shows how stock prices have evolved in the given time frame')
        plot_download(stocks)
        st.subheader('Plot of daily returns to see volatility')
        plot_returns(stocks)
        table = download(stocks)

        

        returns = table.pct_change()
        mean_returns = returns.mean()
        print(f"start_date : {start_date}")
        print(f"mean_returns : {mean_returns}")
        cov_matrix = returns.cov()
        num_portfolios = 25000
        risk_free_rate = 0.0178
        bar()
        max_sharpe = display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate,table)

        optimal_weights = max_sharpe.loc["Allocation"].values / 100

        bar()
        results = monte_carlo_simulation(optimal_weights)
        st.write(f"Performance moyenne du portefeuille : {results['mean_portfolio_return']:.2f}%")
        st.write(f"Volatilité du portefeuille : {results['portfolio_volatility']:.2f}%")
        st.write(f"VaR à 95% : {results['VaR_95']:.2f}%")
        st.write(f"Ratio de Sortino : {results['sortino_ratio']:.2f}")

        bar()

        st.subheader("Pricing du portefeuille (Valeur actuelle nette)")

        st.write(f"Le prix théorique actuel du portefeuille est : ${results['present_value']:.2f}")
        st.write(f"Valeur finale du portefeuille après la période simulée : ${results['final_value']:.2f}")

        bar()

    else:
        st.subheader('Enter your stock tickers, there must be atleast 2 tickers') 
if __name__ == "__main__":
    app()
