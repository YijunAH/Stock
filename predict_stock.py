import pandas as pd
import numpy as np

from fbprophet import Prophet
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt

def get_historical_stock_price(stock):
    data = yf.download(stock, start="2013-01-01", end="2017-08-07")
    return data

def main():
    stock = "DOW"
    df_whole = get_historical_stock_price(stock)
    #print(df_whole)
    df = df_whole.filter(['Date', 'Close'], axis=1)
    #print(df)
    df.reset_index(inplace=True)
    df.columns = ['ds', 'y']
    ##df['y'] = np.log(df['y'])

    m = Prophet()
    m.fit(df)

    num_days = 30 ##int(raw_input("Enter no of days to predict stock price for: "))
    future = m.make_future_dataframe(periods=num_days)
    forecast = m.predict(future)


    #plt.tight_layout(pad=7)
    m.plot(forecast)
    plt.show()
    m.plot_components(forecast)
    #plt.waitforbuttonpress()
    plt.show()
    plt.savefig('predictionDOW.png')


if __name__ == "__main__":
    ##print "hello"
    main()

