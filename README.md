
Introduction:

This is a web-based Portfolio Optimization Tool, which helps investors to make informed decisions based on the historical price of assets. It is built using
Python and Streamlit, a Python library for building web applications for machine learning and data science. The tool uses a Mean-Variance Optimization technique 
to optimize the portfolio, which maximizes the expected return for a given level of risk.

Features:

The Portfolio Optimization Tool has the following features:

• Allows users to input a list of stock tickers and download the corresponding stock price data using Quandl API.

• Plots the stock price data for the selected time period.

• Calculates and plots daily returns for the selected time period.

• Optimizes the portfolio and identifies the allocation of assets that maximize the Sharpe Ratio, which is the ratio of expected portfolio return to portfolio risk.

• Plots the efficient frontier for the given set of assets.


Installation:

To use the Portfolio Optimization Tool, follow these steps:


Install Python version 3.8 or higher on your system.

Clone this repository to your local machine.

Open a command prompt or terminal and navigate to the root directory of the project.

Install the required Python packages by running the following command:

py -m pip install -r requirements.txt

After installing the packages, run the following command:

py -m streamlit run app.py  (si ça ne marche pas utiliser le chemin complet de app.py)

This should open a new tab in your default browser, displaying the web-based Portfolio Optimization Tool.

Usage:

The Portfolio Optimization Tool has a simple user interface that allows users to input stock tickers, and start and end date. It simulates 25000 portfolios 
with a risk free rate of 0.0178. The tool will then download the corresponding stock price data and use it to calculate the daily returns for each asset. 
The efficient frontier for the given set of assets will then be plotted, along with the optimal allocation of assets that maximizes the Sharpe Ratio.

Contributing:

If you would like to contribute to the Portfolio Optimization Tool, feel free to fork the repository and submit a pull request with your changes.

Contact:

If you have any questions or suggestions regarding the Portfolio Optimization Tool, please feel free to contact me at shubhan.kamat@gmail.com
