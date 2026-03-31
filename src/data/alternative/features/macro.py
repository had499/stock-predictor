"""
Standalone macro feature engineering.

Transforms raw expanded FRED data (plus cross-asset and ETF flow data)
into a rich set of ticker-agnostic macro features.  The result has one
row per business day and is designed to be merged into stock DataFrames
on the ``date`` column.

Exports: build_macro_features
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_macro_features(raw_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw expanded FRED data into a rich set of macro features
    that are **ticker-agnostic** – merge once on ``date`` across all stocks.

    Feature categories produced:

    1. **Yield curve shape** – slope, curvature, inversion flags, steepness z-score
    2. **Rate momentum** – 5d / 20d / 60d changes in key rates
    3. **Real rates & inflation** – real yield, breakeven inflation momentum
    4. **Volatility regimes** – VIX level, term structure, regime buckets
    5. **Credit conditions** – HY spread, BBB spread, TED spread, momentum
    6. **Labor market** – claims momentum, payroll surprise proxies
    7. **Consumer & business** – sentiment momentum, inflation expectations gap
    8. **Currency** – USD strength, cross-rate momentum
    9. **Commodities** – oil/gold momentum, oil-gold ratio
    10. **Housing** – mortgage-treasury spread, starts momentum
    11. **Liquidity & financial conditions** – M2 growth, Fed balance sheet, NFCI
    12. **Leading indicators** – composite regime, ISM-like proxies
    13. **Cross-indicator interactions** – risk-on/off composite, macro surprise

    Parameters
    ----------
    raw_macro : pd.DataFrame
        Output of ``scrape_macro_data()`` with date + indicator columns.

    Returns
    -------
    pd.DataFrame
        Columns: date, <macro features …>.  All numeric, no ticker column.
    """
    df = raw_macro.copy()

    if df.empty or "date" not in df.columns:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Helper: safe diff / pct_change (returns NaN if column missing)
    def _diff(col: str, periods: int = 1) -> pd.Series:
        return df[col].diff(periods) if col in df.columns else pd.Series(np.nan, index=df.index)

    def _pct(col: str, periods: int = 1) -> pd.Series:
        return df[col].pct_change(periods) if col in df.columns else pd.Series(np.nan, index=df.index)

    def _zscore(col: str, window: int = 60) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        rolling_mean = df[col].rolling(window, min_periods=10).mean()
        rolling_std = df[col].rolling(window, min_periods=10).std().clip(lower=1e-8)
        return (df[col] - rolling_mean) / rolling_std

    # ── 1. Yield Curve Shape ──
    if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
        df["yc_slope_10y2y"] = df["treasury_10y"] - df["treasury_2y"]
        df["yc_inverted_10y2y"] = (df["yc_slope_10y2y"] < 0).astype(int)
        df["yc_slope_10y2y_z"] = _zscore("yc_slope_10y2y", 252)

    if "treasury_10y" in df.columns and "treasury_3m" in df.columns:
        df["yc_slope_10y3m"] = df["treasury_10y"] - df["treasury_3m"]
        df["yc_inverted_10y3m"] = (df["yc_slope_10y3m"] < 0).astype(int)

    if "treasury_30y" in df.columns and "treasury_2y" in df.columns:
        df["yc_slope_30y2y"] = df["treasury_30y"] - df["treasury_2y"]

    # Curvature: 2y vs (10y+1y)/2 – "butterfly"
    if all(c in df.columns for c in ["treasury_1y", "treasury_2y", "treasury_10y"]):
        df["yc_curvature"] = df["treasury_2y"] - (df["treasury_10y"] + df["treasury_1y"]) / 2

    # Term premium proxy (30y - 5y)
    if "treasury_30y" in df.columns and "treasury_5y" in df.columns:
        df["term_premium_proxy"] = df["treasury_30y"] - df["treasury_5y"]

    # ── 2. Rate Momentum ──
    for tenor, col in [("10y", "treasury_10y"), ("2y", "treasury_2y"),
                       ("ff", "fed_funds_rate")]:
        if col in df.columns:
            df[f"rate_{tenor}_5d_chg"] = _diff(col, 5)
            df[f"rate_{tenor}_20d_chg"] = _diff(col, 20)
            df[f"rate_{tenor}_60d_chg"] = _diff(col, 60)

    # Yield curve slope momentum
    if "yc_slope_10y2y" in df.columns:
        df["yc_slope_5d_chg"] = df["yc_slope_10y2y"].diff(5)
        df["yc_slope_20d_chg"] = df["yc_slope_10y2y"].diff(20)
        df["yc_flattening"] = (df["yc_slope_5d_chg"] < 0).astype(int)

    # ── 3. Real Rates & Inflation ──
    if "real_yield_10y" in df.columns:
        df["real_yield_5d_chg"] = _diff("real_yield_10y", 5)
        df["real_yield_20d_chg"] = _diff("real_yield_10y", 20)
        df["real_yield_z"] = _zscore("real_yield_10y", 252)

    if "breakeven_inflation_10y" in df.columns:
        df["breakeven_5d_chg"] = _diff("breakeven_inflation_10y", 5)
        df["breakeven_20d_chg"] = _diff("breakeven_inflation_10y", 20)
        df["inflation_expectations_elevated"] = (
            df["breakeven_inflation_10y"] > df["breakeven_inflation_10y"].rolling(252, min_periods=60).quantile(0.75)
        ).astype(int)

    if "inflation_expectations" in df.columns:
        df["infl_expect_5d_chg"] = _diff("inflation_expectations", 5)

    # CPI momentum (monthly, so diff across ~22 biz days)
    if "cpi" in df.columns:
        df["cpi_mom"] = _pct("cpi", 22)   # approx monthly

    if "core_cpi" in df.columns:
        df["core_cpi_mom"] = _pct("core_cpi", 22)

    # ── 4. Volatility Regimes ──
    if "vix" in df.columns:
        df["vix_5d_chg"] = _diff("vix", 5)
        df["vix_20d_chg"] = _diff("vix", 20)
        df["vix_5d_pct"] = _pct("vix", 5)
        df["vix_ma20"] = df["vix"].rolling(20, min_periods=5).mean()
        df["vix_z"] = _zscore("vix", 252)
        df["vix_above_ma20"] = (df["vix"] > df["vix_ma20"]).astype(int)
        df["vix_regime"] = pd.cut(
            df["vix"], bins=[0, 15, 20, 25, 35, 100],
            labels=["low", "normal", "elevated", "high", "crisis"],
        ).astype(str)
        # VIX term structure proxy: level vs MA (contango/backwardation)
        df["vix_term_structure"] = df["vix"] / df["vix_ma20"].clip(lower=1)

    # ── 5. Credit Conditions ──
    for spread_name, col in [("hy", "hy_spread"), ("bbb", "bbb_spread"), ("ted", "ted_spread")]:
        if col in df.columns:
            df[f"{spread_name}_spread_5d_chg"] = _diff(col, 5)
            df[f"{spread_name}_spread_20d_chg"] = _diff(col, 20)
            df[f"{spread_name}_spread_z"] = _zscore(col, 252)

    # Credit risk composite
    credit_cols = [c for c in ["hy_spread", "bbb_spread", "ted_spread"] if c in df.columns]
    if credit_cols:
        # Average z-score of available credit spreads
        credit_z = pd.concat([_zscore(c, 252) for c in credit_cols], axis=1)
        df["credit_risk_composite"] = credit_z.mean(axis=1)
        df["credit_stress"] = (df["credit_risk_composite"] > 1.0).astype(int)

    # ── 6. Labor Market ──
    if "initial_claims" in df.columns:
        df["claims_4w_ma"] = df["initial_claims"].rolling(20, min_periods=5).mean()
        df["claims_4w_chg"] = df["claims_4w_ma"].pct_change(20)
        df["claims_z"] = _zscore("initial_claims", 252)
        df["claims_rising"] = (df["claims_4w_chg"] > 0.05).astype(int)

    if "continued_claims" in df.columns:
        df["continued_claims_mom"] = _pct("continued_claims", 20)

    if "nonfarm_payrolls" in df.columns:
        df["payrolls_mom"] = _pct("nonfarm_payrolls", 22)

    if "unemployment_rate" in df.columns:
        df["unemp_3m_chg"] = _diff("unemployment_rate", 63)
        # Sahm Rule proxy: 3-month avg unemployment rise > 0.5pp from 12m low
        df["unemp_3m_avg"] = df["unemployment_rate"].rolling(63, min_periods=22).mean()
        df["unemp_12m_low"] = df["unemployment_rate"].rolling(252, min_periods=63).min()
        df["sahm_rule_indicator"] = (df["unemp_3m_avg"] - df["unemp_12m_low"]).clip(lower=0)
        df["sahm_recession_flag"] = (df["sahm_rule_indicator"] >= 0.5).astype(int)
        df.drop(columns=["unemp_3m_avg", "unemp_12m_low"], inplace=True)

    # ── 7. Consumer & Business Sentiment ──
    if "consumer_sentiment" in df.columns:
        df["sentiment_mom"] = _diff("consumer_sentiment", 22)
        df["sentiment_z"] = _zscore("consumer_sentiment", 252)

    if "inflation_expectations" in df.columns and "breakeven_inflation_10y" in df.columns:
        df["infl_expect_gap"] = df["inflation_expectations"] - df["breakeven_inflation_10y"]

    # ── 8. Currency ──
    if "usd_index_broad" in df.columns:
        df["usd_5d_ret"] = _pct("usd_index_broad", 5)
        df["usd_20d_ret"] = _pct("usd_index_broad", 20)
        df["usd_z"] = _zscore("usd_index_broad", 252)
        df["usd_strong"] = (df["usd_z"] > 1.0).astype(int)

    for ccy, col in [("eur", "usd_eur"), ("jpy", "jpy_usd"), ("gbp", "usd_gbp")]:
        if col in df.columns:
            df[f"{ccy}_5d_ret"] = _pct(col, 5)
            df[f"{ccy}_20d_ret"] = _pct(col, 20)

    # ── 9. Commodities ──
    if "oil_wti" in df.columns:
        df["oil_5d_ret"] = _pct("oil_wti", 5)
        df["oil_20d_ret"] = _pct("oil_wti", 20)
        df["oil_60d_ret"] = _pct("oil_wti", 60)
        df["oil_z"] = _zscore("oil_wti", 252)

    if "oil_brent" in df.columns and "oil_wti" in df.columns:
        df["brent_wti_spread"] = df["oil_brent"] - df["oil_wti"]

    if "gold_price" in df.columns:
        df["gold_5d_ret"] = _pct("gold_price", 5)
        df["gold_20d_ret"] = _pct("gold_price", 20)
        df["gold_z"] = _zscore("gold_price", 252)

    if "gold_price" in df.columns and "oil_wti" in df.columns:
        df["gold_oil_ratio"] = df["gold_price"] / df["oil_wti"].clip(lower=0.01)

    if "gas_price" in df.columns:
        df["gas_5d_ret"] = _pct("gas_price", 5)

    # ── 10. Housing ──
    if "mortgage_30y" in df.columns:
        df["mortgage_5d_chg"] = _diff("mortgage_30y", 5)
        df["mortgage_20d_chg"] = _diff("mortgage_30y", 20)

    if "mortgage_30y" in df.columns and "treasury_10y" in df.columns:
        df["mortgage_spread"] = df["mortgage_30y"] - df["treasury_10y"]

    if "housing_starts" in df.columns:
        df["housing_starts_mom"] = _pct("housing_starts", 22)

    if "building_permits" in df.columns:
        df["permits_mom"] = _pct("building_permits", 22)

    # ── 11. Liquidity & Financial Conditions ──
    if "m2_money_supply" in df.columns:
        df["m2_yoy_growth"] = _pct("m2_money_supply", 252)
        df["m2_3m_growth"] = _pct("m2_money_supply", 63)

    if "fed_balance_sheet" in df.columns:
        df["fed_bs_yoy_growth"] = _pct("fed_balance_sheet", 252)
        df["fed_bs_tightening"] = (df["fed_bs_yoy_growth"] < 0).astype(int)

    if "nfci" in df.columns:
        df["nfci_5d_chg"] = _diff("nfci", 5)
        df["nfci_tightening"] = (df["nfci"] > 0).astype(int)  # >0 = tighter than avg

    if "stress_index" in df.columns:
        df["stress_5d_chg"] = _diff("stress_index", 5)
        df["stress_elevated"] = (df["stress_index"] > 0).astype(int)

    # ── 12. Leading Indicators ──
    if "leading_index" in df.columns:
        df["leading_index_mom"] = _pct("leading_index", 22)
        df["leading_index_negative"] = (df["leading_index_mom"] < 0).astype(int)

    if "industrial_production" in df.columns:
        df["ip_mom"] = _pct("industrial_production", 22)

    if "retail_sales_ex_food" in df.columns:
        df["retail_sales_mom"] = _pct("retail_sales_ex_food", 22)

    if "durable_goods_orders" in df.columns:
        df["durables_mom"] = _pct("durable_goods_orders", 22)

    if "inventory_sales_ratio" in df.columns:
        df["inv_sales_z"] = _zscore("inventory_sales_ratio", 252)

    # Defragment before adding derived composite columns
    df = df.copy()

    # ── 13. Cross-Indicator Interactions ──

    # Risk-on / Risk-off composite
    risk_signals = []
    if "vix_z" in df.columns:
        risk_signals.append(-df["vix_z"])  # low VIX = risk-on
    if "credit_risk_composite" in df.columns:
        risk_signals.append(-df["credit_risk_composite"])
    if "yc_slope_10y2y_z" in df.columns:
        risk_signals.append(df["yc_slope_10y2y_z"])  # steep curve = risk-on
    if "usd_z" in df.columns:
        risk_signals.append(-df["usd_z"])  # weak USD = risk-on
    if risk_signals:
        df["risk_appetite_composite"] = pd.concat(risk_signals, axis=1).mean(axis=1)
        df["risk_on_regime"] = (df["risk_appetite_composite"] > 0.5).astype(int)
        df["risk_off_regime"] = (df["risk_appetite_composite"] < -0.5).astype(int)

    # Monetary policy stance composite
    policy_signals = []
    if "rate_ff_20d_chg" in df.columns:
        policy_signals.append(df["rate_ff_20d_chg"])
    if "fed_bs_yoy_growth" in df.columns:
        policy_signals.append(-df["fed_bs_yoy_growth"])  # shrinking = tightening
    if "real_yield_10y" in df.columns:
        policy_signals.append(_zscore("real_yield_10y", 252))
    if policy_signals:
        df["policy_tightness"] = pd.concat(policy_signals, axis=1).mean(axis=1)

    # Growth vs inflation regime
    growth_cols = [c for c in ["ip_mom", "payrolls_mom", "retail_sales_mom"] if c in df.columns]
    infl_cols = [c for c in ["cpi_mom", "core_cpi_mom", "breakeven_5d_chg"] if c in df.columns]
    if growth_cols and infl_cols:
        df["growth_composite"] = df[growth_cols].mean(axis=1)
        df["inflation_composite"] = df[infl_cols].mean(axis=1)
        # Quadrant: stagflation = low growth + high inflation
        df["stagflation_risk"] = (
            (df["growth_composite"] < df["growth_composite"].rolling(252, min_periods=60).quantile(0.25)) &
            (df["inflation_composite"] > df["inflation_composite"].rolling(252, min_periods=60).quantile(0.75))
        ).astype(int)

    # ── 14. Cross-Asset Signals ──
    if "btc_price" in df.columns:
        df["btc_5d_ret"] = _pct("btc_price", 5)
        df["btc_20d_ret"] = _pct("btc_price", 20)
        df["btc_60d_ret"] = _pct("btc_price", 60)
        df["btc_z"] = _zscore("btc_price", 252)
        # BTC as risk appetite proxy
        df["btc_risk_signal"] = (df["btc_5d_ret"] > 0).astype(int)

    if "copper_price" in df.columns:
        df["copper_5d_ret"] = _pct("copper_price", 5)
        df["copper_20d_ret"] = _pct("copper_price", 20)
        df["copper_z"] = _zscore("copper_price", 252)

    if "copper_price" in df.columns and "gold_price" in df.columns:
        # Copper/gold ratio: rising = risk-on (industrial demand), falling = risk-off
        df["copper_gold_ratio"] = df["copper_price"] / df["gold_price"].clip(lower=0.01)
        df["copper_gold_ratio_z"] = _zscore("copper_gold_ratio", 252)
        df["copper_gold_momentum"] = df["copper_gold_ratio"].pct_change(20)

    # ── 15. ETF Fund Flow Signals ──
    flow_cols = [c for c in df.columns if c.endswith("_abnormal_volume")]
    if flow_cols:
        # Aggregate flow signal: average abnormal volume across key ETFs
        df["aggregate_flow_signal"] = df[flow_cols].mean(axis=1)
        df["flow_risk_on"] = (df["aggregate_flow_signal"] > 0.5).astype(int)

    if "spy_flow_momentum" in df.columns:
        df["spy_flow_5d"] = df["spy_flow_momentum"]

    if "hyg_abnormal_volume" in df.columns:
        # HYG (high yield bonds) flow = credit appetite proxy
        df["credit_flow_signal"] = df["hyg_abnormal_volume"]

    if "tlt_abnormal_volume" in df.columns:
        # TLT (long treasuries) flow = flight-to-safety proxy
        df["safety_flow_signal"] = df["tlt_abnormal_volume"]

    # Risk appetite v2: augment with cross-asset + flow data
    risk_v2 = []
    if "btc_z" in df.columns:
        risk_v2.append(df["btc_z"])
    if "copper_gold_ratio_z" in df.columns:
        risk_v2.append(df["copper_gold_ratio_z"])
    if "aggregate_flow_signal" in df.columns:
        risk_v2.append(df["aggregate_flow_signal"])
    if risk_v2:
        df["risk_appetite_v2"] = pd.concat(risk_v2, axis=1).mean(axis=1)

    # Convert date back to string for consistency
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    logger.info(
        "Macro features: %d rows × %d columns (%d raw → %d derived)",
        len(df), df.shape[1], raw_macro.shape[1], df.shape[1] - raw_macro.shape[1],
    )
    return df
