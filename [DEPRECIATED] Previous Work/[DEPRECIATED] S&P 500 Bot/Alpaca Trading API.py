# chhatseh@gmail.com
# diaproject520520

from alpaca.trading.client import TradingClient

client = TradingClient(
    'YOUR_API_KEY_ID',
    'YOUR_API_SECRET_KEY',
    paper=True  # Set False for live trading
)
