import yfinance as yf
import pandas as pd

# Список тикеров, которые нам нужны
tickers = {
    'DX-Y.NYB': 'DXY',          # Индекс доллара
    'GC=F': 'Gold_USD',         # Золото
    'BZ=F': 'Brent_Oil',        # Нефть Brent
    'BTC-USD': 'BTC_USD',       # Биткоин
    'EURUSD=X': 'EUR_USD',      # Евро к доллару
    'RUB=X': 'USD_RUB'          # Евро к доллару
}

# 2 за последние 16 лет с месячным интервалом
# только колонку 'Close' (цена закрытия месяца)
data = yf.download(list(tickers.keys()), start="2010-01-01", end="2026-10-16", interval="1mo")['Close']

data = data.rename(columns=tickers)

# сохраняем в Excel
data = data.reset_index()
# Приводим дату к формату ГГГГ-ММ-ДД (последний день месяца)
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

data.to_excel('yahoo_features.xlsx', index=False)
print(data.head())