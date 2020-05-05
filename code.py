import os
import sys
import tweepy
import requests
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# First we login into twitter
consumer_key = '' # Enter Consumer Key
consumer_secret = ''# Enter consumer_secret
access_token = ''# Enter access_token
access_token_secret = ''# Enter access_token_secret
alphaapi = ''# Enter alphaapi 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
user = tweepy.API(auth)

# Where the csv file will live
FILE_NAME = 'historical.csv'


def stock_sentiment(quote, num_tweets):
    # Checks if the sentiment for our quote is
    # positive or negative, returns True if
    # majority of valid tweets have positive sentiment
    list_of_tweets = user.search(quote, count=num_tweets)
    positive, null = 0, 0

    for tweet in list_of_tweets:
        blob = TextBlob(tweet.text).sentiment
        if blob.subjectivity == 0:
            null += 1
            next
        if blob.polarity > 0:
            positive += 1

    if positive > ((num_tweets - null)/2):
        return True


def get_historical(quote):
    # Download our file from google finance
    #url = 'http://www.google.com/finance/historical?q=NASDAQ%3A'+quote+'&output=csv'
    #url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+quote+'&interval=5min&outputsize=full&apikey='+alphaapi
    #url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+quote+'&interval=5min&apikey='+alphaapi+'&datatype=csv'
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+quote+'&apikey='+alphaapi+'&datatype=csv'
    
    r = requests.get(url, stream=True)

    if r.status_code != 400:
        with open(FILE_NAME, 'wb') as f:
            for chunk in r:
                f.write(chunk)

        return True


def stock_prediction():
    # Collect data points from csv
    dataset = []
    dates = []

    with open(FILE_NAME) as f:
        for n, line in enumerate(f):
            if n != 0:
              dataset.append(float(line.split(',')[1]))
              dates.append((line.split(',')[0]))

    dataset = np.array(dataset)
    dates = [dates[n+1] for n in range(len(dates)-2)]
    step = len(dates)//6
    datex = dates[::step]
    #datex.pop()
    #dates = np.array(dates).reshape(-1,1)
    '''
    datex = []
    for date in dates:
      datex.append(date[-2:])
    '''  
    # Create dataset matrix (X=t and Y=t+1)
    def create_dataset(dataset):
        dataX = [dataset[n+1] for n in range(len(dataset)-2)]
        dataX=(np.array(dataX)).reshape(-1,1)
        return dataX, dataset[2:]
        
    trainX, trainY = create_dataset(dataset)

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(trainX, trainY)  # perform linear regression
    prediction = linear_regressor.predict(trainX)
    result = 'The price will move from %s to %s' % (dataset[-1], prediction[-1])

    xAxis = [i for i in range(1,len(trainX)+1)]
    xAxis = np.array(xAxis) 
    plt.style.use('fivethirtyeight')

    y = np.polyfit(xAxis, trainX, 3)
    y = np.squeeze(y)
    p3 = np.poly1d(y)
    xp = np.linspace(0, len(trainX), 100)
    plt.plot(xAxis, trainX, '.', xp, p3(xp), '--', lw=1.8)
    locs, labels=plt.xticks()
    x_ticks = []
    datex.reverse()
    plt.xticks(locs,datex, rotation=45, horizontalalignment='right')
    #plt.xticks(datex, trainX)

    plt.show()

    return result

    
# Ask user for a stock quote
stock_quote = input('Enter a stock quote from NASDAQ (e.j: AAPL, FB, GOOGL): ').upper()

# Check if the stock sentiment is positve
if not stock_sentiment(stock_quote, num_tweets=100):
    print ('This stock has bad sentiment, please re-run the script')
    sys.exit()

# Check if we got te historical data
if not get_historical(stock_quote):
    print ('Google returned a 404, please re-run the script and')
    print ('enter a valid stock quote from NASDAQ')
    sys.exit()

# We have our file so we create the neural net and get the prediction
print (stock_prediction())

# We are done so we delete the csv file
os.remove(FILE_NAME)