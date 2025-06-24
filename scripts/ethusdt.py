# In[1]:


import ccxt
import pandas as pd
import time

exchange = ccxt.binance()
symbol = 'ETH/USDT'
timeframe = '1m'
# Fetch data for the last 3 days to ensure enough data for various MAs
since = exchange.parse8601('2025-06-01T00:00:00Z')
all_ohlcv = []
limit = 1000
while True:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    if len(ohlcv) == 0:
        break
    all_ohlcv.extend(ohlcv)
    since = ohlcv[-1][0] + 1
    time.sleep(exchange.rateLimit / 1000) # Rate limiting

df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df = df.set_index('timestamp')
print(df.head())


# In[2]:


try:
    ticker = exchange.fetch_ticker(symbol)
    if 'last' in ticker and ticker['last'] is not None:
        last_price = ticker['last']
        current_time = pd.to_datetime('now', utc=True)
        # Create a DataFrame with the latest price
        live_data_df = pd.DataFrame([{
            'timestamp': current_time,
            'open': 0, # Placeholder
            'high': 0, # Placeholder
            'low': 0,  # Placeholder
            'close': last_price,
            'volume': 0 # Placeholder
        }])
        live_data_df = live_data_df.set_index('timestamp')
        print("Successfully fetched live price data:")
        print(live_data_df)
    else:
        print("Could not retrieve 'last' price from ticker data.")
except Exception as e:
    print(f"Failed to fetch live data: {e}")


# In[3]:


# Check if df and live_data_df are not empty before concatenating
if not df.empty and not live_data_df.empty:
    combined_df = pd.concat([df, live_data_df])
elif not df.empty:
    combined_df = df.copy()
elif not live_data_df.empty:
    combined_df = live_data_df.copy()
else:
    print("Both historical and live data DataFrames are empty.")
    combined_df = pd.DataFrame() # Create an empty DataFrame if both are empty

if not combined_df.empty:
    # Ensure the index is a DatetimeIndex before sorting
    if not isinstance(combined_df.index, pd.DatetimeIndex):
         combined_df.index = pd.to_datetime(combined_df.index)

    combined_df = combined_df.sort_index(ascending=True)
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    print(combined_df.head())
    print(combined_df.tail())
else:
    print("Cannot process an empty combined DataFrame.")


# In[4]:


if combined_df.empty:
    print("Resampling cannot be performed due to the absence of data.")
else:
    # Resample to 5-minute frequency
    df_5min = combined_df['close'].resample('5min').last().dropna()
    # Resample to 15-minute frequency
    df_15min = combined_df['close'].resample('15min').last().dropna()
    # Resample to 1-hour frequency
    df_1h = combined_df['close'].resample('1h').last().dropna()

    ma_5min = df_5min.rolling(window=5).mean()
    ma_15min = df_15min.rolling(window=5).mean()
    ma_1h = df_1h.rolling(window=5).mean()

    # Display the head of the resampled DataFrames
    print("5-minute resampled data:")
    print(df_5min.head())
    print("\n15-minute resampled data:")
    print(df_15min.head())
    print("\n1-hour resampled data:")
    print(df_1h.head())


# In[5]:


try:
    # Check if moving average variables are defined and not empty
    if 'ma_5min' in locals() and ma_5min is not None and not ma_5min.empty:
        print(f"Current 5-minute Moving Average: {ma_5min.iloc[-1]}")
    else:
        print("5-minute Moving Average could not be calculated due to missing data.")

    if 'ma_15min' in locals() and ma_15min is not None and not ma_15min.empty:
        print(f"Current 15-minute Moving Average: {ma_15min.iloc[-1]}")
    else:
        print("15-minute Moving Average could not be calculated due to missing data.")

    if 'ma_1h' in locals() and ma_1h is not None and not ma_1h.empty:
        print(f"Current 1-hour Moving Average: {ma_1h.iloc[-1]}")
    else:
        print("1-hour Moving Average could not be calculated due to missing data.")

except NameError:
    print("Moving average variables are not defined. Calculation likely failed in previous steps.")
except Exception as e:
    print(f"An error occurred while displaying moving averages: {e}")


# In[6]:


import plotly.graph_objects as go

def plot_resampled_with_ma(df, ma, title):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df,
            mode='lines',
            name='Close Price'
        ))

        fig.add_trace(go.Scatter(
            x=ma.index,
            y=ma,
            mode='lines',
            name='Moving Average',
            line=dict(dash='dot', color='orange')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Price',
            legend=dict(x=0, y=1),
            template='plotly_dark',
            height=500
        )

        fig.show()


# In[7]:


plot_resampled_with_ma(df_5min, ma_5min, "5-Minute Close Price & MA")
plot_resampled_with_ma(df_15min, ma_15min, "15-Minute Close Price & MA")
plot_resampled_with_ma(df_1h, ma_1h, "1-Hour Close Price & MA")


# In[8]:


df_1h = combined_df['close'].resample('1h').ohlc().dropna()
df_15m = combined_df['close'].resample('15min').last().dropna()

# Make sure the timestamps align
df_1h = df_1h[df_1h.index + pd.Timedelta(minutes=45) <= df_15m.index[-1]]

results = []

for ts in df_1h.index:
    open_price = df_1h.loc[ts, 'open']
    close_price = df_1h.loc[ts, 'close']

    # Find 45-min mark price
    ts_45 = ts + pd.Timedelta(minutes=45)
    if ts_45 not in df_15m.index:
        continue
    price_45 = df_15m.loc[ts_45]

    # Direction at 45m and at end
    delta_45 = price_45 - open_price
    delta_close = close_price - open_price

    direction_45 = 'up' if delta_45 > 0 else 'down' if delta_45 < 0 else 'flat'
    direction_close = 'up' if delta_close > 0 else 'down' if delta_close < 0 else 'flat'
    flipped = direction_45 != direction_close

    results.append({
        'timestamp': ts,
        'open': open_price,
        'price_at_45min': price_45,
        'close': close_price,
        'delta_45_pct': (delta_45 / open_price) * 100,
        'direction_45': direction_45,
        'direction_close': direction_close,
        'flipped': flipped
    })

# Turn into DataFrame
flip_df = pd.DataFrame(results)

# Example: probability of flipping if candle is UP more than 0.5% at 45min
threshold = 0.5
subset = flip_df[flip_df['delta_45_pct'] > threshold]
prob_flip_from_up = subset['flipped'].mean()

print(f"\nProbability of a candle flipping from UP (> {threshold}%) at 45min to DOWN at close: {prob_flip_from_up:.2%}")


# In[9]:


import plotly.express as px

# Histogram: delta_45_pct vs flipped
fig = px.histogram(
    flip_df, 
    x='delta_45_pct', 
    color='flipped', 
    nbins=40, 
    title='Flip Probabilities by 45-Minute Candle Change (%)'
)
fig.show()


# In[10]:


subset_down = flip_df[flip_df['delta_45_pct'] < -threshold]
prob_flip_from_down = subset_down['flipped'].mean()
print(f"Probability of a candle flipping from DOWN (< -{threshold}%) at 45min to UP at close: {prob_flip_from_down:.2%}")
