import yfinance as yf
import pandas as pd

# try:
#     # Load the Excel file
#     file_name = 'scraped.xlsx'
#     data = pd.read_excel(file_name)
#
#     # Extract the first column (A)
#     column_a = data.iloc[:, 0]
#
#     print(column_a)
# except:
#     print(f"Failure")

# List of S&P 500 companies (tickers)
companies = ["MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A"]

# Define the date range
start_date = "2020-01-01"
end_date = "2025-04-04"

# Create an empty DataFrame to store all data
all_data = pd.DataFrame()

# Loop through each company ticker
for company in companies:
    try:
        # Download historical data for the company
        data = yf.download(company, start=start_date, end=end_date)

        # Add a column for the company ticker
        data['Company'] = company

        # Append the company's data to the main DataFrame
        all_data = pd.concat([all_data, data])

        print(f"Data fetched for {company}")
    except Exception as e:
        print(f"Failed to fetch data for {company}: {e}")

# Save the combined data to a CSV file
all_data.to_csv("s&p500_individual_companies.csv")

print("All data saved to s&p500_individual_companies.csv")
