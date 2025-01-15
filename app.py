#Importing necessary libraries
import streamlit as st
import sys
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from Source.exception import CustomException
from Source.logger import logging
from itertools import combinations
from sklearn.cluster import KMeans
from math import floor
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

def train_lstm_model(data, ticker, num_epochs=50, batch_size=32):
    """
    Entraîne un modèle LSTM pour prédire les prix des actions.

    :param data: DataFrame contenant les prix historiques des actions
    :param ticker: Symbole de l'action à utiliser pour l'entraînement
    :param num_epochs: Nombre d'époques d'entraînement
    :param batch_size: Taille du batch
    :return: Modèle LSTM entraîné
    """
    # Extraire les prix de l'action spécifique
    prices = data[[ticker]].values

    # Normaliser les données pour les mettre dans la plage [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Créer les séquences de données pour LSTM
    X, y = [], []
    lookback = 60  # Utiliser les 60 derniers jours pour prédire le suivant
    for i in range(lookback, len(scaled_prices)):
        X.append(scaled_prices[i - lookback:i, 0])
        y.append(scaled_prices[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshaper X pour correspondre aux dimensions attendues par LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Construire le modèle LSTM
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50, return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1))  # Prédire une valeur de sortie

    # Compiler le modèle
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entraîner le modèle
    model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=1)

    return model, scaler

def predict_future_prices(model, data, ticker, scaler, days_to_predict=252):
    """
    Utilise le modèle LSTM pour prédire les prix futurs.

    :param model: Modèle LSTM entraîné
    :param data: DataFrame contenant les prix historiques
    :param ticker: Symbole de l'action à prédire
    :param scaler: Scaler utilisé pour normaliser les données
    :param days_to_predict: Nombre de jours à prédire
    :return: Liste des prix prédits
    """
    # Extraire les données récentes pour la prédiction
    recent_data = data[[ticker]].values[-60:]  # Prendre les 60 derniers jours
    scaled_recent_data = scaler.transform(recent_data)

    # Préparer l'entrée pour la prédiction
    inputs = scaled_recent_data.reshape(1, -1, 1)

    predicted_prices = []
    for _ in range(days_to_predict):
        # Prédire le prix du jour suivant
        predicted_price = model.predict(inputs, verbose=0)
        predicted_prices.append(predicted_price[0, 0])

        # Ajouter la prédiction au jeu de données
        inputs = np.append(inputs[:, 1:, :], [predicted_price], axis=1)

    # Dé-normaliser les prédictions
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices.flatten()

def black_scholes_basket(weights, prices, volatilities, correlations, strike, T, r, option_type="call"):
    """
    Calcule le prix d'une option basket en utilisant le modèle de Black-Scholes.
    
    :param weights: Liste des poids pour chaque actif dans le basket
    :param prices: Liste des prix actuels des actifs
    :param volatilities: Liste des volatilitités des actifs
    :param correlations: Matrice de corrélation entre les actifs
    :param strike: Prix d'exercice (strike price) de l'option
    :param T: Temps jusqu'à l'échéance (en années)
    :param r: Taux sans risque (annuel)
    :param option_type: "call" pour une option d'achat, "put" pour une option de vente
    :return: Prix de l'option basket
    """
    # Calcul du prix synthétique du basket
    S_basket = np.dot(weights, prices)
    
    # Calcul de la volatilité synthétique
    vol_basket = np.sqrt(
        np.dot(weights, np.dot(correlations * np.outer(volatilities, volatilities), weights))
    )
    
    # Calcul des paramètres d1 et d2
    d1 = (np.log(S_basket / strike) + (r + 0.5 * vol_basket**2) * T) / (vol_basket * np.sqrt(T))
    d2 = d1 - vol_basket * np.sqrt(T)
    
    if option_type == "call":
        # Formule pour une option call
        price = S_basket * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        # Formule pour une option put
        price = strike * np.exp(-r * T) * norm.cdf(-d2) - S_basket * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    return price

def cluster_stocks(returns, num_clusters, metric="mean_return"):
    # Calcul de la matrice de corrélation
    corr_matrix = returns.corr()
    
    # Application de KMeans sur la matrice de corrélation
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(corr_matrix)
    labels = model.labels_
    
    # Regrouper les actions par clusters
    clustered_stocks = {i: [] for i in range(num_clusters)}
    for stock, label in zip(returns.columns, labels):
        clustered_stocks[label].append(stock)
    
    for label in clustered_stocks.keys():
        clustered_stocks.update({label: list(map(lambda x : x[0], clustered_stocks[label]))})
    
    # Sélectionner une action aléatoire par cluster non vide
    selected_stocks = []
    for label, st in clustered_stocks.items():
        if len(st) > 0:  # Vérification que le cluster n'est pas vide
            if metric == 'mean_return':
                # Calcul de la moyenne des rendements pour chaque action
                mean_returns = returns[st].mean()
                best_stock = mean_returns.idxmax()  # Action avec le meilleur rendement moyen
            elif metric == 'volatility':
                # Calcul de la volatilité pour chaque action
                volatilities = returns[st].std()
                best_stock = volatilities.idxmin()  # Action avec la volatilité la plus faible
            elif metric == 'sharpe_ratio':
                # Calcul du ratio de Sharpe pour chaque action
                mean_returns = returns[st].mean()
                volatilities = returns[st].std()
                sharpe_ratios = mean_returns / volatilities
                best_stock = sharpe_ratios.idxmax()  # Action avec le meilleur ratio de Sharpe
            else:
                raise ValueError(f"Métrique inconnue : {metric}")
            selected_stocks.append(best_stock[0])
    
    return list(map(str, selected_stocks))

def bar():
    st.write("-"*80)

risk_free_rate = 0.0178

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

T = st.sidebar.number_input("Années jusqu'à l'échéance", value=1, min_value=1, step=1)
strike = st.sidebar.number_input("Strike (Prix d'exercice)", value=initial_capital, step=1)
r = st.sidebar.number_input("Taux sans risque (%)", value=risk_free_rate * 100, step=0.1) / 100

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
def random_portfolios(basket, num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios)) #initialize a 3 x num_portfolios matrix for storing results
    weights_record = [] #empty list to store weights record for each portfolio
    
    for i in range(num_portfolios): 
        # Generate random portfolio weights that sum to 1
        weights = np.random.random(len(basket))
        weights /= np.sum(weights)
        
        # Record the weights for the current portfolio
        weights_record.append(weights)
        
        # Calculate the portfolio's annualized standard deviation, returns, and Sharpe ratio
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
    return results, weights_record

def get_correlation_matrix_df(cov_matrix):
    variances = np.diag(cov_matrix)

    # 2. Calculer les racines carrées des variances
    std_devs = np.sqrt(variances)

    # 3. Normaliser la matrice de covariance pour obtenir la matrice de corrélation
    corr_mat = cov_matrix / (std_devs[:, None] * std_devs[None, :])

    return corr_mat

def get_correlation_matrix(cov_matrix):
    
    return get_correlation_matrix_df(cov_matrix).to_numpy()

def show_correlation_matrix(cov_matrix, corr_mat):
    st.write("Correlation matrix:")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=cov_matrix.columns, yticklabels=cov_matrix.columns)
    st.pyplot(plt)

def display_simulated_ef_with_random(basket, mean_returns, cov_matrix, num_portfolios, risk_free_rate, table):
    '''This is a Python function that uses the mean returns and covariance matrix of a set of assets to simulate a
      number of portfolios with random weights, calculate their returns and volatility, and plot them on a scatter plot.
        It also identifies the portfolio with the maximum Sharpe ratio (the ratio of return to volatility) and the portfolio 
        with the minimum volatility, and prints their allocations and performance metrics.'''
    try:
        results, weights = random_portfolios(basket, num_portfolios,mean_returns, cov_matrix, risk_free_rate)
        
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

        return max_sharpe_allocation, rp, sdp, min_vol_allocation, rp_min, sdp_min, results
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

def monte_carlo_simulation(basket, optimal_weights, num_simulations=10000, num_days=252):
    """
    Effectue une simulation Monte Carlo pour évaluer un portefeuille d'actifs optimisé selon le modèle de Markowitz.
    
    :param num_simulations: Nombre de simulations Monte Carlo à effectuer
    :param num_days: Nombre de jours de trading simulés (par défaut 252 jours par an)
    
    :return: Dictionnaire avec les résultats des simulations (performance, volatilité, VaR)
    """

    # Téléchargement des données historiques
    data = yf.download(basket, start=start_date, end=end_date)["Close"]
    
    # Calcul des rendements quotidiens
    returns = data.pct_change().dropna()
    
    # Calcul des rendements moyens et de la matrice de covariance
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Simulation des trajectoires de prix
    simulated_prices = np.zeros((num_simulations, num_days, len(basket)))
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
        'mean_portfolio_return': mean_portfolio_return * 100,
        'portfolio_volatility': portfolio_volatility * 100,
        'VaR_95': VaR_95 * 100,
        'sortino_ratio': sortino
    }

    return results

def show_alloc(max_sharpe_allocation, rp, sdp, min_vol_allocation, rp_min, sdp_min, res):
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
    plt.scatter(res[0,:], res[1,:], c = res[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    st.pyplot(fig)

def app():
    '''This code defines a function app() that is called when the script is executed as the 
    main program'''
    if num_stocks > 1 and num_stocks >= num_basket >= 1:
        st.subheader('Plot shows how stock prices have evolved in the given time frame')
        plot_download(stocks)
        st.subheader('Plot of daily returns to see volatility')
        plot_returns(stocks)
        table = download(stocks)

        bar()

        returns = table.pct_change()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        corr_matrix = get_correlation_matrix(cov_matrix)

        st.subheader("Matrice de corrélation")

        show_correlation_matrix(cov_matrix, corr_matrix)

        print(f"start_date : {start_date}")
        print(f"mean_returns : {mean_returns}")
        num_portfolios = 25000

        best_sharpe = -np.inf
        best_basket = []
        best_ret = 0
        best_vol = 1

        reduced_comb = combinations(cluster_stocks(returns, floor(.7 * num_basket + .3 * num_stocks)), num_basket)

        for basket in reduced_comb:
            tab = table[list(basket)]

            ret = tab.pct_change()
            mean_ret = ret.mean()
            cov_mat = ret.cov()

            max_sharpe, rp, sdp, min_vol, rp_min, sdp_min, _ = display_simulated_ef_with_random(basket, mean_ret, cov_mat, num_portfolios, risk_free_rate, tab)

            if rp > best_ret and rp_min < best_vol:
                best_sharpe = max_sharpe
                best_basket = basket
                best_ret = rp
                best_vol = rp_min

        bar()

        tab = table[list(best_basket)]

        optimal_weights = best_sharpe.loc["Allocation"].values / 100

        ret_opt = tab.pct_change()
        mean_opt = ret_opt.mean()
        cov_opt = ret_opt.cov()

        max_sharpe, rp, sdp, min_vol, rp_min, sdp_min, res = display_simulated_ef_with_random(best_basket, mean_opt, cov_opt, num_portfolios, risk_free_rate, table[list(best_basket)])

        show_correlation_matrix(cov_opt, get_correlation_matrix(cov_opt))

        show_alloc(max_sharpe, rp, sdp, min_vol, rp_min, sdp_min, res)

        optimal_weights = max_sharpe.loc["Allocation"].values / 100

        results = monte_carlo_simulation(best_basket, optimal_weights)
        st.write(f"Performance moyenne du portefeuille : {results['mean_portfolio_return']:.2f}%")
        st.write(f"Volatilité du portefeuille : {results['portfolio_volatility']:.2f}%")
        st.write(f"VaR à 95% : {results['VaR_95']:.2f}%")
        st.write(f"Ratio de Sortino : {results['sortino_ratio']:.2f}")

        bar()

        st.subheader("Gain du portefeuille")

        st.write("Option Best-of")
        st.write(f"Le gain sera basé sur l'action ayant un rendement maximal : \${initial_capital} + {np.max(ret_opt) * 100:.2f}\% = \${(initial_capital * (1 + np.max(ret_opt))):.2f}")

        st.write("Option Worst-of")
        st.write(f"Le gain sera basé sur l'action ayant un rendement minimal : \${initial_capital} - {-np.min(ret_opt) * 100:.2f}\% = \${(initial_capital * (1 + np.min(ret_opt))):.2f}")
        
        st.write("Option Average")
        avg = np.mean(ret_opt)

        if avg >= 0:
            st.write(f"Le gain sera basé sur la moyenne des rendements : \${initial_capital} + {avg * 100:.2f}\% = \${(initial_capital * (1 + avg)):.2f}")
        else:
            st.write(f"Le gain sera basé sur la moyenne des rendements : \${initial_capital} - {-avg * 100:.2f}\% = \${(initial_capital * (1 + avg)):.2f}")
        
        bar()

        st.subheader("Pricing du produit structuré")

        # Derniers prix, volatilités, et matrice de corrélation pour le basket
        prices = tab.iloc[-1].to_numpy()
        correlations = get_correlation_matrix(cov_opt)

        log_ret = np.log(prices[1:] / prices[:-1])
        sigma =  log_ret.std()
        volatilities = sigma * np.sqrt(252)

        # Calcul du prix d'option basket call
        basket_call_price = black_scholes_basket(
            weights=optimal_weights,
            prices=prices,
            volatilities=volatilities,
            correlations=correlations,
            strike=strike,
            T=T,
            r=r,
            option_type="call"
        )

        basket_put_price = black_scholes_basket(
            weights=optimal_weights,
            prices=prices,
            volatilities=volatilities,
            correlations=correlations,
            strike=strike,
            T=T,
            r=r,
            option_type="put"
        )

        st.write(f"Prix d'option basket call : ${basket_call_price:.2f}")
        st.write(f"Prix d'option basket put : ${basket_put_price:.2f}")

        bar()

        num_days = 252 * T

        for ticker in best_basket:
            model, scaler = train_lstm_model(tab, ticker)
            print(f"Model trained successfully for {ticker}!")

            predicted_prices = predict_future_prices(model, tab, ticker, scaler, days_to_predict=num_days)

            # Afficher les prédictions
            st.subheader(f"Prix prédits pour {ticker} ({T} années)")
            fig = plt.figure(figsize=(14, 7))
            plt.plot(range(len(predicted_prices)), predicted_prices, label='Prix prédits')
            plt.title("Prédictions basées sur LSTM")
            plt.xlabel("Jour")
            plt.ylabel("Prix")
            plt.legend()
            st.pyplot(fig)

    else:
        if num_stocks <= 1:
            st.subheader('Enter your stock tickers, there must be atleast 2 tickers')
        if num_stocks < num_basket:
            st.subheader("Number of basket items must be less than number of stocks.")
        if num_basket < 1:
            st.subheader("Number of baskets must not be zero.")

if __name__ == "__main__":
    app()
