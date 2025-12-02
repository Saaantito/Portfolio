
"""
Portfolio Visualizer - Application Streamlit
Version Finale - Bug yfinance complètement résolu
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(page_title="Portfolio Visualizer", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# FONCTIONS DE CACHE POUR TÉLÉCHARGER LES DONNÉES
# ============================================================================

@st.cache_data(ttl=3600)
def download_data(tickers, start_date, end_date):
    """Télécharge les données de prix ajustés - VERSION ULTRA ROBUSTE"""
    try:
        # Convertit en string
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

        # Crée un DataFrame vide pour stocker les résultats
        all_prices = pd.DataFrame()

        # Télécharge ticker par ticker pour éviter les problèmes de MultiIndex
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True)

                if df.empty:
                    st.warning(f"⚠️ Aucune donnée pour {ticker}")
                    continue

                # Utilise Close (qui est déjà ajusté avec auto_adjust=True)
                if 'Close' in df.columns:
                    all_prices[ticker] = df['Close']
                else:
                    st.warning(f"⚠️ Colonne Close introuvable pour {ticker}")
                    continue

            except Exception as e:
                st.error(f"❌ Erreur pour {ticker}: {str(e)}")
                continue

        # Vérifie qu'on a au moins un ticker valide
        if all_prices.empty:
            return None

        # Supprime les lignes avec des NaN
        all_prices = all_prices.dropna()

        if all_prices.empty:
            return None

        return all_prices

    except Exception as e:
        st.error(f"❌ Erreur générale: {str(e)}")
        return None

# ============================================================================
# FONCTION DE SIMULATION DU PORTEFEUILLE
# ============================================================================

def simulate_portfolio(prices_df, allocations, initial_capital, contribution_amount, 
                       contribution_freq, rebalance_freq):
    """
    Simule l'évolution jour par jour du portefeuille avec DCA et rebalancement.
    """

    dates = prices_df.index
    portfolio_value = pd.Series(index=dates, dtype=float)

    # Initialisation: Achat des parts au jour 0
    shares = {}
    for ticker, weight in allocations.items():
        amount_to_invest = initial_capital * weight
        shares[ticker] = amount_to_invest / prices_df.loc[dates[0], ticker]

    # Tracking pour DCA et rebalancement
    last_contribution_date = dates[0]
    last_rebalance_year = dates[0].year

    for i, date in enumerate(dates):
        # Calcul de la valeur totale du portefeuille
        total_value = sum(shares[ticker] * prices_df.loc[date, ticker] for ticker in allocations.keys())
        portfolio_value[date] = total_value

        # ===== GESTION DES CONTRIBUTIONS (DCA) =====
        add_contribution = False
        if contribution_freq == 'Monthly':
            if (date - last_contribution_date).days >= 30:
                add_contribution = True
        elif contribution_freq == 'Quarterly':
            if (date - last_contribution_date).days >= 90:
                add_contribution = True
        elif contribution_freq == 'Annual':
            if (date - last_contribution_date).days >= 365:
                add_contribution = True

        if add_contribution and contribution_amount > 0:
            # Ajoute le capital selon l'allocation cible
            for ticker, weight in allocations.items():
                amount_to_invest = contribution_amount * weight
                shares[ticker] += amount_to_invest / prices_df.loc[date, ticker]
            last_contribution_date = date

        # ===== GESTION DU REBALANCEMENT =====
        if rebalance_freq == 'Annual' and date.year > last_rebalance_year:
            # Recalcule la valeur totale après contribution éventuelle
            total_value = sum(shares[ticker] * prices_df.loc[date, ticker] for ticker in allocations.keys())

            # Vend tout et rachète selon les poids cibles
            for ticker, weight in allocations.items():
                target_value = total_value * weight
                shares[ticker] = target_value / prices_df.loc[date, ticker]

            last_rebalance_year = date.year

    return portfolio_value

# ============================================================================
# CALCUL DES MÉTRIQUES FINANCIÈRES
# ============================================================================

def calculate_cagr(portfolio_values):
    """Calcule le CAGR (Compound Annual Growth Rate)"""
    start_value = portfolio_values.iloc[0]
    end_value = portfolio_values.iloc[-1]
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    if years > 0 and start_value > 0:
        cagr = (end_value / start_value) ** (1 / years) - 1
        return cagr * 100
    return 0

def calculate_volatility(returns):
    """Calcule la volatilité annualisée"""
    return returns.std() * np.sqrt(252) * 100

def calculate_max_drawdown(portfolio_values):
    """Calcule le drawdown maximum"""
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    return drawdown.min() * 100

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """Calcule le ratio de Sharpe"""
    excess_returns = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    if volatility > 0:
        return excess_returns / volatility
    return 0

def calculate_beta(portfolio_returns, benchmark_returns):
    """Calcule le Bêta du portefeuille par rapport au benchmark"""
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    benchmark_variance = np.var(benchmark_returns)
    if benchmark_variance > 0:
        return covariance / benchmark_variance
    return 0

def calculate_treynor_ratio(portfolio_returns, beta, risk_free_rate=0.03):
    """Calcule le ratio de Treynor"""
    portfolio_return = portfolio_returns.mean() * 252
    if beta != 0:
        return (portfolio_return - risk_free_rate) / beta
    return 0

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

st.title("📊 Portfolio Visualizer Pro")
st.markdown("### Simulez la performance de votre portefeuille avec DCA et Rebalancement")

# ============================================================================
# SIDEBAR - PARAMÈTRES
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration du Portefeuille")

    # 1. Sélection des actifs
    st.subheader("1. Actifs")
    tickers_input = st.text_input(
        "Tickers (séparés par des virgules)",
        value="AAPL,MSFT,SPY",
        help="Ex: AAPL, MSFT, SPY"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    # 2. Période
    st.subheader("2. Période")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Date de début",
            value=datetime.now() - timedelta(days=5*365),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "Date de fin",
            value=datetime.now() - timedelta(days=30),
            max_value=datetime.now()
        )

    # 3. Capital initial
    st.subheader("3. Capital")
    initial_capital = st.number_input(
        "Capital initial ($)",
        min_value=100,
        value=10000,
        step=1000
    )

    # 4. Allocation
    st.subheader("4. Allocation")
    allocations = {}
    total_weight = 0

    for ticker in tickers:
        weight = st.slider(
            f"{ticker} (%)",
            min_value=0,
            max_value=100,
            value=int(100/len(tickers)),
            key=f"weight_{ticker}"
        )
        allocations[ticker] = weight / 100
        total_weight += weight

    # Vérification de la somme
    if total_weight != 100:
        st.error(f"⚠️ La somme des poids est {total_weight}%. Elle doit être égale à 100%.")
        st.stop()
    else:
        st.success(f"✅ Allocation totale: {total_weight}%")

    # 5. Paramètres DCA
    st.subheader("5. Dollar Cost Averaging (DCA)")
    contribution_amount = st.number_input(
        "Contribution périodique ($)",
        min_value=0,
        value=500,
        step=100
    )
    contribution_freq = st.selectbox(
        "Fréquence",
        options=['None', 'Monthly', 'Quarterly', 'Annual'],
        index=1
    )

    # 6. Rebalancement
    st.subheader("6. Rebalancement")
    rebalance_freq = st.selectbox(
        "Fréquence de rebalancement",
        options=['None', 'Annual'],
        index=0
    )

    st.markdown("---")
    run_simulation = st.button("🚀 Lancer la Simulation", type="primary")

# ============================================================================
# ZONE PRINCIPALE - RÉSULTATS
# ============================================================================

if run_simulation:
    with st.spinner("Téléchargement des données..."):
        # Télécharge les données + SPY pour le benchmark
        all_tickers = list(set(tickers + ['SPY']))
        prices_data = download_data(all_tickers, start_date, end_date)

        if prices_data is None or prices_data.empty:
            st.error("❌ Impossible de récupérer les données.")
            st.info("💡 Essayez avec des dates plus anciennes (ex: 2020-2024)")
            st.stop()

        # Vérifie que tous les tickers sont présents
        missing_tickers = [t for t in tickers if t not in prices_data.columns]
        if missing_tickers:
            st.error(f"❌ Tickers introuvables: {', '.join(missing_tickers)}")
            st.info("💡 Vérifiez l'orthographe sur finance.yahoo.com")
            st.stop()

        # Vérifie SPY
        if 'SPY' not in prices_data.columns:
            st.error("❌ Impossible de charger le benchmark SPY")
            st.stop()

    with st.spinner("Simulation du portefeuille..."):
        # Simule le portefeuille utilisateur
        portfolio_values = simulate_portfolio(
            prices_data[tickers],
            allocations,
            initial_capital,
            contribution_amount,
            contribution_freq,
            rebalance_freq
        )

        # Simule le benchmark SPY avec le même capital
        spy_initial_shares = initial_capital / prices_data['SPY'].iloc[0]
        spy_values = prices_data['SPY'] * spy_initial_shares

    # ========================================================================
    # AFFICHAGE DES RÉSULTATS
    # ========================================================================

    st.success("✅ Simulation terminée!")

    # SECTION 1: Graphique d'évolution
    st.markdown("## 📈 Évolution du Portefeuille")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Normalise pour comparaison
    portfolio_norm = (portfolio_values / portfolio_values.iloc[0]) * 100
    spy_norm = (spy_values / spy_values.iloc[0]) * 100

    ax.plot(portfolio_norm.index, portfolio_norm.values, label='Votre Portefeuille', linewidth=2, color='#2E86AB')
    ax.plot(spy_norm.index, spy_norm.values, label='Benchmark (SPY)', linewidth=2, color='#A23B72', linestyle='--')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Valeur Normalisée (Base 100)', fontsize=12)
    ax.set_title('Performance du Portefeuille vs Benchmark', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig)

    # SECTION 2: KPIs
    st.markdown("## 📊 Métriques de Performance")

    # Calcul des rendements
    portfolio_returns = portfolio_values.pct_change().dropna()
    spy_returns = spy_values.pct_change().dropna()

    # Calcul des métriques
    cagr = calculate_cagr(portfolio_values)
    volatility = calculate_volatility(portfolio_returns)
    max_dd = calculate_max_drawdown(portfolio_values)
    sharpe = calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.03)

    # Beta et Treynor
    beta = calculate_beta(portfolio_returns.values, spy_returns.values)
    treynor = calculate_treynor_ratio(portfolio_returns, beta, risk_free_rate=0.03)

    # Affichage en colonnes
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("CAGR", f"{cagr:.2f}%")
        st.metric("Volatilité (Ann.)", f"{volatility:.2f}%")

    with col2:
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
        st.metric("Ratio de Sharpe", f"{sharpe:.2f}")

    with col3:
        st.metric("Bêta (vs SPY)", f"{beta:.2f}")
        st.metric("Ratio de Treynor", f"{treynor:.2f}")

    # SECTION 3: Matrice de corrélation
    if len(tickers) > 1:
        st.markdown("## 🔥 Matrice de Corrélation des Actifs")

        returns_df = prices_data[tickers].pct_change().dropna()
        correlation_matrix = returns_df.corr()

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax2
        )
        ax2.set_title('Corrélation des Rendements Quotidiens', fontsize=14, fontweight='bold')
        plt.tight_layout()

        st.pyplot(fig2)

    # SECTION 4: Détails finaux
    st.markdown("## 💰 Résumé Final")

    final_value = portfolio_values.iloc[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**Valeur Initiale:** ${initial_capital:,.2f}")
        st.info(f"**Valeur Finale:** ${final_value:,.2f}")
    with col_b:
        st.success(f"**Rendement Total:** {total_return:,.2f}%")
        st.success(f"**Gain/Perte:** ${final_value - initial_capital:,.2f}")

else:
    st.info("👈 Configurez votre portefeuille dans la barre latérale et cliquez sur 'Lancer la Simulation'")

    # Affiche un exemple visuel
    st.markdown("### 🎯 Fonctionnalités")
    st.markdown("""
    - ✅ **Simulation historique** avec données réelles via Yahoo Finance
    - 💵 **Dollar Cost Averaging (DCA)** configurable
    - ⚖️ **Rebalancement automatique** annuel
    - 📈 **Métriques avancées**: CAGR, Sharpe, Treynor, Drawdown
    - 🔗 **Comparaison** avec benchmark SPY
    - 🎨 **Corrélations** entre actifs
    """)
