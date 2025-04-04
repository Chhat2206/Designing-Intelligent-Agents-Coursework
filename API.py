import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random

# Fetch S&P 500 constituents from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {'User-Agent': 'Mozilla/5.0'}  # Avoid 403 Forbidden error

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the first wikitable (constituents table)
    table = soup.find('table', {'class': 'wikitable'})

    # Extract ticker symbols from first column
    tickers = []
    for row in table.find_all('tr')[1:]:  # Skip header row
        cols = row.find_all('td')
        if len(cols) > 0:
            ticker = cols[0].text.strip()
            tickers.append(ticker)

except requests.exceptions.RequestException as e:
    print(f"Failed to fetch Wikipedia page: {e}")
    tickers = []

# Define date range and parameters
start_date = "2020-01-01"
end_date = "2025-04-04"
all_data = pd.DataFrame()
# DELAY_RANGE = (0.5, 2)  # Seconds between requests

# Download historical data for each ticker
for i, ticker in enumerate(tickers, 1):
    try:
        # Clean ticker symbols
        clean_ticker = ticker.replace('.', '-')

        # Download data with progress disabled
        data = yf.download(
            clean_ticker,
            start=start_date,
            end=end_date,
            progress=False
        )

        # Add company identifier
        data['Ticker'] = clean_ticker

        # Append to main dataframe
        all_data = pd.concat([all_data, data])

        print(f"({i}/{len(tickers)}) Success: {clean_ticker} | Rows: {len(data)}")

        # Random delay to avoid rate limiting
        # time.sleep(random.uniform(*DELAY_RANGE))

    except Exception as e:
        print(f"({i}/{len(tickers)}) Failed: {ticker} | Error: {str(e)[:50]}")

# Save results
if not all_data.empty:
    all_data.to_csv("sp500_full_dataset.csv")
    print(f"Data saved for {len(tickers)} companies")
else:
    print("No data collected")
