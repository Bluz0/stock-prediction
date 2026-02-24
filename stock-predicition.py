import os
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# Import Alpaca & ML
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from forex_python.converter import CurrencyRates

# Configuration Rich
console = Console()

c = CurrencyRates()
rate = c.get_rate('USD', 'EUR')
TAUX_DE_CHANGE = rate  # Taux de change USD -> EUR

def run_prediction():
    load_dotenv()
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

    console.print(Panel(f"[bold cyan]Bourse Prediction (Taux de change : {TAUX_DE_CHANGE:.2f} EUR/USD)[/bold cyan]", expand=False))
    
    symbol = console.input("[bold yellow]➤ Entrez le symbole de l'action (ex: NVDA, ASML): [/bold yellow]").upper()

    # --- DEBUT DE L'ANIMATION ---
    with console.status(f"[bold green]Récupération des données pour {symbol}...", spinner="dots"):
        
        START_DATE = '2024-01-01'
        END_DATE = datetime.now().strftime('%Y-%m-%d')
        
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=START_DATE,
            end=END_DATE,
            feed=DataFeed.IEX
        )

        bars = client.get_stock_bars(request_params)
        df = bars.df.reset_index()
        df = df[df['symbol'] == symbol]

        # Feature Engineering
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(window=5).std()
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['target_return'] = df['return'].shift(-1)
        df = df.dropna()

        features = ['close', 'volume', 'volatility', 'ma_5', 'ma_20']
        X = df[features]
        y = df['target_return']

    with console.status("[bold blue]Machine Learning en cours de réfléxion...", spinner="bouncingBar"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42)
        model.fit(X_train, y_train)

        # Prédictions
        predicted_returns = model.predict(X_test)
        predicted_prices = X_test['close'].values * (1 + predicted_returns)
        actual_prices = df.loc[y_test.index, 'close'].values
        
        # Calcul MAE
        mae = mean_absolute_error(actual_prices, predicted_prices)

    # --- FIN DES ANIMATIONS, AFFICHAGE DES RESULTATS ---
    
    # Prédiction pour demain (Correction du warning feature names)
    latest_data = X.tail(1) 
    next_return_pred = model.predict(latest_data)[0]
    current_price = X.iloc[-1]['close']
    next_day_price = current_price * (1 + next_return_pred)
    
    # Tableau de bord
    table = Table(title=f"\n📊 Résultats : {symbol}", title_style="bold magenta", border_style="bright_blue")
    table.add_column("Indicateur", style="cyan")
    table.add_column("Valeur", justify="right", style="white")

    variation_color = "green" if next_return_pred > 0 else "red"
    variation_icon = "📈" if next_return_pred > 0 else "📉"

    table.add_row("Marge d'erreur (MAE)", f"{mae:.2f} $")
    table.add_row("Prix de clôture", f"{current_price:.2f} $")
    table.add_row("Tendance prédite", f"[{variation_color}]{next_return_pred*100:.2f}% {variation_icon}[/]")
    
    console.print(table)

    # Panneau final
    conclusion = (
        f"Prix cible ($) : [bold yellow]{next_day_price:.2f} $[/]\n"
        f"Prix cible (€) : [bold green]{next_day_price * TAUX_DE_CHANGE:.2f} €[/]"
    )
    console.print(Panel(conclusion, title="🚀 Prédiction Lendemain", border_style="bright_green"))

    # Affichage graphique
    console.print("\n[dim italic]Ouverture du graphique... (Vérifiez votre barre des tâches)[/dim italic]")
    dates_test = df.loc[y_test.index, 'timestamp']
    plt.figure(figsize=(10, 5))
    plt.plot(dates_test, actual_prices, label='Réel', color='#1f77b4')
    plt.plot(dates_test, predicted_prices, label='Prédit', color='#d62728', linestyle='--')
    plt.title(f"Historique vs Prédiction - {symbol}")
    plt.xticks(rotation=35)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_prediction()