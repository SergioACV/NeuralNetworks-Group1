import pandas as pd
import numpy as np
import yfinance as yf
import simfin as sf

# =====================================================================
# 1. CONFIGURACIÓN DE SIMFIN (Usando variante Trimestral Gratuita)
# =====================================================================
sf.set_api_key('3acf3f87-3043-4c6a-accf-36d178179d7f')
sf.set_data_dir('~/simfin_data/')

print("Descargando ratios históricos trimestrales de SimFin...")
# Cambiamos variant='daily' por 'quarterly' para evitar el error 500 de pago
df_simfin_quarterly = sf.load_derived(variant='quarterly', market='us')

# =====================================================================
# 2. SELECCIÓN DE TICKERS Y PARAMETRIZACIÓN TEMPORAL (S&P 500)
# =====================================================================
tickers_sp500 = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL']
start_date = "2018-01-01"
end_date = "2024-04-01"

# Filtrar SimFin para nuestros tickers
df_simfin_filtered = df_simfin_quarterly.loc[df_simfin_quarterly.index.get_level_values('Ticker').isin(tickers_sp500)]

# =====================================================================
# 3. CONVERSIÓN DE TRIMESTRAL A DIARIO (Forward Fill)
# =====================================================================
print("Indexando y expandiendo ratios trimestrales a frecuencia diaria...")
df_simfin_filtered = df_simfin_filtered.reset_index()
df_simfin_filtered['Date'] = pd.to_datetime(df_simfin_filtered['Date'])

# Reindexamos por Ticker y Fecha para rellenar los días intermedios con el último dato conocido (Forward Fill)
df_list = []
for ticker, group in df_simfin_filtered.groupby('Ticker'):
    group = group.set_index('Date').resample('D').ffill()
    group['Ticker'] = ticker
    df_list.append(group)

df_simfin_daily = pd.concat(df_list).reset_index()
df_simfin_daily = df_simfin_daily.query(f'"{start_date}" <= Date <= "{end_date}"')
df_simfin_daily.set_index(['Ticker', 'Date'], inplace=True)

# =====================================================================
# 4. DESCARGA Y CÁLCULO DE INDICADORES TÉCNICOS (Yahoo Finance)
# =====================================================================
print("Descargando precios y calculando indicadores técnicos de Yahoo Finance...")
lista_df_yahoo = []

for ticker in tickers_sp500:
    df_yf = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df_yf.empty:
        continue
        
    df_yf = df_yf.reset_index()
    df_yf['Ticker'] = ticker
    df_yf['Date'] = pd.to_datetime(df_yf['Date'])
    
    # Variables del Paper
    df_yf['Log_Return'] = np.log(df_yf['Close'] / df_yf['Close'].shift(1))
    df_yf['Amount'] = df_yf['Close'] * df_yf['Volume']
    df_yf['Change'] = df_yf['Close'].pct_change()
    
    # EMA y MACD (Estrategia de trading)
    df_yf['EMA_12'] = df_yf['Close'].ewm(span=12, adjust=False).mean()
    df_yf['EMA_26'] = df_yf['Close'].ewm(span=26, adjust=False).mean()
    df_yf['MACD_DIF'] = df_yf['EMA_12'] - df_yf['EMA_26']
    df_yf['MACD_DEA'] = df_yf['MACD_DIF'].ewm(span=9, adjust=False).mean()
    df_yf['MACD'] = 2 * (df_yf['MACD_DIF'] - df_yf['MACD_DEA'])
    
    df_yf_features = df_yf[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 
                            'Volume', 'Amount', 'Change', 'Log_Return', 
                            'MACD', 'MACD_DIF', 'MACD_DEA']]
    lista_df_yahoo.append(df_yf_features)

df_yahoo_total = pd.concat(lista_df_yahoo)
df_yahoo_total.set_index(['Ticker', 'Date'], inplace=True)

# =====================================================================
# 5. FUSIÓN FINAL DEL DATASET
# =====================================================================
print("Fusionando fuentes de datos...")
dataset_final = df_yahoo_total.join(df_simfin_daily, how='inner')

# Limpieza de NaNs iniciales por lags e indicadores técnicos
dataset_final.dropna(subset=['Log_Return', 'MACD'], inplace=True)

# Tratar P/E negativos como nulos y removerlos (Instrucción explícita del Paper) 
if 'Price to Earnings' in dataset_final.columns:
    dataset_final.loc[dataset_final['Price to Earnings'] < 0, 'Price to Earnings'] = np.nan
    dataset_final.dropna(subset=['Price to Earnings'], inplace=True)

dataset_final = dataset_final.reset_index()

print(f"\n¡Dataset generado exitosamente! Forma actual: {dataset_final.shape}")
print(dataset_final[['Ticker', 'Date', 'Close', 'Log_Return', 'Return on Equity', 'Price to Earnings']].head())