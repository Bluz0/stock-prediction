## IN ENGLISH
# Predict the price of a stock on the stock exchange - Alpaca & ML

A console application to predict stock price movements (like NVIDIA or ASML) using machine learning (**Random Forest**) and the **Alpaca Markets* API.

## Features
- **Real-time recovery** : Connect to the Alpaca API to get historical data (free IEX feed).
- **Artificial Intelligence** : RandomForestRegressor model trained on returns to avoid extrapolation errors.
- **Premium Terminal Interface**: Use of the `Rich` library for colorful dashboards and animations (spinners).
- **Visualization** : Matplotlib graphs comparing actual prices and predictions.

## Installation

1. **Clone the project** :
   ```bash
   git clone [https://github.com/Bluz0/bourse-prediction.git](https://github.com/Bluz0/bourse-prediction.git)
   cd stock exchange-prediction

2. **Install dependencies**
    ```bash
    pip install pandas numpy matplotlib scikit-learn alpaca-py python-dotenv rich

3. **Create an Alpaca account and put its API_KEY in . env**
    (create a .env in /bourse-prediction)
    ```bash
    ALPACA_API_KEY=your_key
    ALPACA_SECRET_KEY=secrey_key

4. **ENJOY**
    ```bash
    python stock-predicition.py

## The Model

The model predicts the percentage change (Rt ) rather than the gross price for better accuracy on high growth stocks:

<p>Rt = Price_t Price_t 1 / Price_t-1  </p>


## EN FRANCAIS
# Predire le prix d'une action en bourse - Alpaca & ML

Une application console pour prédire les mouvements de prix des actions (comme NVIDIA ou ASML) en utilisant l'apprentissage automatique (**Random Forest**) et l'API **Alpaca Markets**.

## Fonctionnalités
- **Récupération en temps réel** : Connexion à l'API Alpaca pour obtenir les données historiques (flux IEX gratuit).
- **Intelligence Artificielle** : Modèle `RandomForestRegressor` entraîné sur les rendements (returns) pour éviter les erreurs d'extrapolation.
- **Interface Terminal Premium** : Utilisation de la librairie `Rich` pour des tableaux de bord colorés et des animations (spinners).
- **Visualisation** : Graphiques Matplotlib comparant les prix réels et les prédictions.

## Installation

1. **Cloner le projet** :
   ```bash
   git clone [https://github.com/Bluz0/bourse-prediction.git](https://github.com/Bluz0/bourse-prediction.git)
   cd bourse-prediction

2. **Installer les dépendances**
    ```bash
    pip install pandas numpy matplotlib scikit-learn alpaca-py python-dotenv rich

3. **Crée un compte Alpaca et mettre son API_KEY dans .env**
    (créer un .env dans /bourse-prediction)
    ```bash
    ALPACA_API_KEY=votre_cle_ici
    ALPACA_SECRET_KEY=votre_secret_ici

4. **ENJOY**
    ```bash
    python stock-predicition.py

## Le Modèle

Le modèle prédit la variation en pourcentage (Rt​) plutôt que le prix brut pour une meilleure précision sur les actions en forte croissance :

<p>Rt​=​Price_t ​− Price_t−1 / Price_t-1​​</p>
