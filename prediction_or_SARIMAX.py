import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Chargement de notre fichier CSV des prix de l'or
df = pd.read_csv("1979-2021.csv")

# On convertit les dates du CSV en datetime pour pouvoir les exploiter
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# On garde seulement les colonnes USD et EURO
data = df[['United States(USD)', 'Europe(EUR)']].dropna()

# --- 1. Visualisation initiale ---
plt.figure(figsize=(14, 6))
plt.plot(data['United States(USD)'], label="Prix en USD", color="blue")
plt.plot(data['Europe(EUR)'], label="Prix en EURO", color="orange")
plt.title("Évolution des prix de l'or (USD et EUR)")
plt.xlabel("Année")
plt.ylabel("Prix de l'or")
plt.legend()
plt.show()

# --- 2. Analyse statistique initiale ---
print("Résumé des données :\n")
print(data.describe())

# --- 3. Test de stationnarité (ADF) ---
def test_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"\nTest ADF pour {name} :")
    print(f"  - ADF Statistic: {result[0]}")
    print(f"  - p-value: {result[1]}")
    if result[1] > 0.05:
        print(f"  -> {name} n'est PAS stationnaire (p-value > 0.05). Différenciation nécessaire.")
    else:
        print(f"  -> {name} est stationnaire (p-value <= 0.05).")

# Tester USD et EURO
test_stationarity(data['United States(USD)'], "USD")
test_stationarity(data['Europe(EUR)'], "EURO")

# --- 4. Différenciation pour stationnarité ---
usd_diff = data['United States(USD)'].diff().dropna()
eur_diff = data['Europe(EUR)'].diff().dropna()

# Re-tester après différenciation
test_stationarity(usd_diff, "USD différencié")
test_stationarity(eur_diff, "EURO différencié")

# Visualisation des données différenciées
plt.figure(figsize=(14, 6))
plt.plot(usd_diff, label="USD différencié", color="blue")
plt.plot(eur_diff, label="EURO différencié", color="orange")
plt.title("Séries temporelles différenciées (USD et EURO)")
plt.xlabel("Année")
plt.ylabel("Variation des prix")
plt.legend()
plt.show()

# --- 5. ACF et PACF pour choisir les paramètres SARIMAX ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ACF et PACF pour USD
plot_acf(usd_diff, ax=axes[0, 0], lags=40)
axes[0, 0].set_title("ACF - USD")
plot_pacf(usd_diff, ax=axes[0, 1], lags=40)
axes[0, 1].set_title("PACF - USD")

# ACF et PACF pour EURO
plot_acf(eur_diff, ax=axes[1, 0], lags=40)
axes[1, 0].set_title("ACF - EURO")
plot_pacf(eur_diff, ax=axes[1, 1], lags=40)
axes[1, 1].set_title("PACF - EURO")

plt.tight_layout()
plt.show()

# --- 6. Ajustement d'un modèle SARIMAX ---
# Choisir les paramètres en fonction des ACF/PACF et itérer pour optimiser.
model_usd = SARIMAX(data['United States(USD)'], order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
results_usd = model_usd.fit(disp=False)

model_eur = SARIMAX(data['Europe(EUR)'], order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
results_eur = model_eur.fit(disp=False)

# Résumé des résultats
print("\nRésumé du modèle SARIMAX pour USD :")
print(results_usd.summary())

print("\nRésumé du modèle SARIMAX pour EURO :")
print(results_eur.summary())

# --- 7. Prévisions ---
forecast_steps = 48  # Prévisions pour 24 mois
forecast_usd = results_usd.get_forecast(steps=forecast_steps)
forecast_eur = results_eur.get_forecast(steps=forecast_steps)

forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Obtenir les intervalles de confiance
usd_forecast_ci = forecast_usd.conf_int()
eur_forecast_ci = forecast_eur.conf_int()

# Visualisation des prévisions USD et EURO
plt.figure(figsize=(14, 6))

# USD
plt.plot(data['United States(USD)'], label="USD Réel", color="blue")
plt.plot(forecast_index, forecast_usd.predicted_mean, label="USD Prévision", color="skyblue")
plt.fill_between(forecast_index, usd_forecast_ci.iloc[:, 0], usd_forecast_ci.iloc[:, 1], color="skyblue", alpha=0.3)

# EURO
plt.plot(data['Europe(EUR)'], label="EURO Réel", color="orange")
plt.plot(forecast_index, forecast_eur.predicted_mean, label="EURO Prévision", color="gold")
plt.fill_between(forecast_index, eur_forecast_ci.iloc[:, 0], eur_forecast_ci.iloc[:, 1], color="gold", alpha=0.3)

plt.title("Prévisions SARIMAX des prix de l'or (USD et EURO)")
plt.xlabel("Année")
plt.ylabel("Prix de l'or")
plt.legend()
plt.show()
