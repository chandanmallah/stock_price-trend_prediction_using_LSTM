import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data  # to scrap data from yahoo website
from keras.models import load_model  # to load keras model
import streamlit as st
from datetime import date
from plotly import graph_objs as go
import yfinance as yf
import streamlit.components.v1 as com

com.html("""
<div>
<style>
h1.heading{
    color: Black;
    border-radius: 20px;
    text-align: center;
}
</style>
<h1 class = "heading">
Stock Trend Prediction
</h1>
</div>
""")
def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-vector/stock-market-concept-with-bull-bear_1017-9634.jpg?w=740&t=st=1666281587~exp=1666282187~hmac=d93af01596488bf0116a41cfa7ef807358aec6459db1bde46695729d5cf6ee15");
             background-attachment: fixed;
             background-size: cover
         }}
         [data-testid="stHeader"] {{
         background-color: rgba(0,0,0,0);
         }}
         [data-testid="stSidebar"] {{
         background-image: url("https://seekingalpha.com/samw/static/images/media-x1.593a5718.png");
         background-size: cover;
         }}
         </style>
         <iframe width="700" height="80" src="https://rss.app/embed/v1/ticker/twZiEqz20wuRDhRA" frameborder="0"></iframe>

         """,
        unsafe_allow_html=True
    )


add_bg_from_url()


start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")

with st.sidebar:
    st.markdown(f'<h1 style="color:#000000;font-size:24px;">{"Stock Trend Prediction"}</h1>', unsafe_allow_html=True)
    user_input = st.text_input('Enter stock Ticker for prediction', 'TSLA')
    df = data.DataReader(user_input, 'yahoo', start, end)


def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data


data = load_data(user_input)
st.subheader('Raw data')
st.write(data.tail())

# describing data

st.subheader('Data from 2010-2022')
df = df.drop(['Adj Close'], axis=1)
st.write(df.describe())

# visualization
st.subheader('Opening Price vs Closing Price with Rangeslider')


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    fig.update_layout(
        autosize=False,
        width=600,
        height=400,
    )
    st.plotly_chart(fig)


plot_raw_data()

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(6, 4))
plt.plot(ma100)
plt.plot(df.Close)
st.plotly_chart(fig1)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(6, 4))
plt.plot(ma100, 'r', label='100 days moving average')
plt.plot(ma200, 'g', label='200 days moving average')
plt.plot(df.Close, 'b', label='closing price')
plt.ylabel('Price')
plt.xlabel('Year')
plt.legend()
st.plotly_chart(fig2)

st.write("If 100 Days Moving Average Crosses 200 Days Moving Average Then it's an uptrend/downtrend")

# splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler

Scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = Scaler.fit_transform(data_training)

# load model
model = load_model('keras+model.h5')

# testing_part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = Scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

Scaler = Scaler.scale_
scale_factor = 1 / Scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# predicted price
st.subheader('Predictions for Tomorrow')
fig4 = plt.figure(figsize=(6, 4))
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.ylabel('Time')
plt.xlabel('Price')
plt.legend()
st.plotly_chart(fig4)

def add_bg_from_url2():
    st.markdown(
        f"""
         <style>
         
         [data-testid="stHeader"] {{
         background-color: rgba(0,0,0,0);
         }}
         [data-testid="stSidebar"] {{
         background-size: cover;
         }}
         </style>
         <iframe width="600" height="350" src="https://rss.app/embed/v1/wall/twPMShVhqoRGtKz7" frameborder="0"></iframe>

         """,
        unsafe_allow_html=True
    )


add_bg_from_url2()
