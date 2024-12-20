from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv("1979-2021.csv")

# Préparer les données pour Prophet
df['Date'] = pd.to_datetime(df['Date'])
prophet_df = df[['Date', 'United States(USD)']].dropna().rename(columns={'Date': 'ds', 'United States(USD)': 'y'})

# Ajuster le modèle Prophet
model = Prophet()
model.fit(prophet_df)

# Faire des prévisions sur 24 mois
future = model.make_future_dataframe(periods=24, freq='M')
forecast = model.predict(future)

# Tracer les prévisions
model.plot(forecast)
plt.title("Prévisions des prix de l'or (Prophet)")
plt.show()

# Visualisation des composantes (tendance, saisonnalité)
model.plot_components(forecast)
plt.show()
