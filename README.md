# Predire le prix d'une action en bourse - Alpaca & ML

Une application console élégante pour prédire les mouvements de prix des actions (comme NVDA ou ASML) en utilisant l'apprentissage automatique (**Random Forest**) et l'API **Alpaca Markets**.

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