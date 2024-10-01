import argparse
import os
import re
import requests
import json
import datetime


import pandas as pd
from kucoin.client import Market


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", required=True)
args = parser.parse_args()

folder = args.folder
#folder = "/home/eisti/python_distrib/script_srt/folder/"




def test_lib_kc():
	client = Market(url='https://api.kucoin.com')

	symbols = client.get_market_list()
	print(symbols)

	tickers = client.get_all_tickers()
	tickers = pd.DataFrame(tickers['ticker'])
	tickers.set_index('symbol', inplace=True)
	print(tickers.head().T)



def api_ticker_info():

	currencies = requests.get(url + '/api/v1/currencies')
	currencies = currencies.json()
	currencies = pd.DataFrame(currencies['data'])
	currencies.set_index('currency', inplace=True)
	print('There are {} currencies'.format(len(currencies)))
	print(currencies.head().T)



def live_price_data():
	#order book
	ticker = requests.get(url + '/api/v1/market/orderbook/level1?symbol=BTC-USDT').json()
	#print(ticker['data'])

	#market last 24h
	ticker_24h = requests.get(url + '/api/v1/market/stats?symbol=BTC-USDT').json()
	print(ticker_24h['data'])
	"""for x in ticker_24h:
		print(ticker_24h[x])"""

	#prices
	fiat = requests.get(url + '/api/v1/prices?base=USD').json()
	print(fiat['data'])
	#print(len(fiat['data'].keys()))
	#print(fiat['data']["BTC"])
	#print(fiat['data'])
	#print(fiat['data'])


def historical_data():
	"""
	history = requests.get(url + '/api/v1/market/histories?symbol=BTC-USDT')
	history = history.json()
	history = pd.DataFrame(history['data'])
	#history['time'] = pd.to_datetime(history['time'])
	history.set_index('time', inplace=True)
	#print(history)
	print()
	"""

	"""
	kline = requests.get(url + '/api/v1/market/candles?type=1min&symbol=BTC-USDT&startAt=1566703297&endAt=1566789757')
	kline = kline.json()
	kline = pd.DataFrame(kline['data'])
	kline = kline.rename({0:"Time",1:"Open",
	                2:"Close",3:"High",4:"Low",5:"Amount",6:"Volume"}, axis='columns')
	kline.set_index('Time', inplace=True)
	kline.head()"""


	timestamp_start = 1508371200  #1508371200
	timestamp_end = timestamp_start + 1500 * 86400


	kline = requests.get(url + '/api/v1/market/candles?type=1day&symbol=RUNE-USDT&startAt=' + str(timestamp_start) + '&endAt=' + str(timestamp_end))
	kline = kline.json()
	kline = pd.DataFrame(kline['data'])
	kline = kline.rename({0:"Time",1:"Open",
	                2:"Close",3:"High",4:"Low",5:"Amount",6:"Volume"}, axis='columns')
	kline.insert(1, "Timestamp", kline["Time"], True)
	kline['Time'] = pd.to_datetime(kline['Time'], unit="s")
	kline.set_index('Time', inplace=True)
	print(kline)


"""
btc listing kucoin
2017-10-19 -> 1508371200


pour chaque symbole, avec la requete avec '&startAt &endAt' :

si head n'est pas la date d'aujourd'hui, continuer
-> timestamp_start = timestamp_end

attention les bornes seront en double

"""



if __name__ == "__main__":

	url='https://api.kucoin.com'

	#test_lib_kc()
	#api_ticker_info()
	#live_price_data()
	historical_data()






