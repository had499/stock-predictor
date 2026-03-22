"""
Alternative Data Feature Engineering

Transforms raw scraped alternative data into model-ready features
aligned to a daily (date, ticker) grid.

Key feature groups:
  - **Insider signals**   – buy ratio, net flow, rolling activity
  - **Institutional**     – holder count, ownership %
  - **Earnings**          – surprise magnitude, beat/miss flags, days since
  - **Analyst**           – consensus score, distribution skew
  - **Short interest**    – short %, ratio, squeeze signal
  - **Options flow**      – put/call ratio, volume imbalance
  - **Congress trades**   – net flow, recent activity flag
  - **Attention**         – Wikipedia pageview spikes & trends
  - **Macro regime**      – yield curve, VIX regime, rate momentum

Author: Stock Predictor Project
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


def build_alternative_features(
    raw_data: Dict[str, pd.DataFrame],
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Merge all alternative data sources into a single daily DataFrame
    for one ticker.

    Parameters
    ----------
    raw_data : dict[str, DataFrame]
        Output of ``scrape_alternative_data()``.
    ticker : str
    start_date, end_date : str (ISO format)

    Returns
    -------
    DataFrame with columns (date, ticker, <features…>).
    All features are numeric and forward-filled where appropriate.
    """
    # Base: daily date grid (business days)
    dates = pd.bdate_range(start=start_date, end=end_date)
    base = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "ticker": ticker.upper(),
    })

    # ── Merge each source ──

    if "insider" in raw_data:
        base = _merge_insider_features(base, raw_data["insider"])

    if "earnings" in raw_data:
        base = _merge_earnings_features(base, raw_data["earnings"])

    if "congress" in raw_data:
        base = _merge_congress_features(base, raw_data["congress"])

    if "wiki" in raw_data:
        base = _merge_wiki_features(base, raw_data["wiki"])

    if "fred" in raw_data:
        base = _merge_macro_features(base, raw_data["fred"])

    if "earnings_revisions" in raw_data:
        base = _merge_earnings_revision_features(base, raw_data["earnings_revisions"])

    return base


# ---------------------------------------------------------------------------
# Merge helpers – each creates + joins features for one source
# ---------------------------------------------------------------------------

def _merge_insider_features(
    base: pd.DataFrame, insider_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Insider trading features from EDGAR Form 4 filings.

    Signals:
      - insider_buy_ratio         : fraction of buys among all insider trades
      - insider_net_transactions   : buys − sells
      - insider_activity_5d       : any insider trade in the last 5 biz days
      - insider_buys_30d          : rolling 30-day buy count
      - insider_sells_30d         : rolling 30-day sell count
      - insider_net_value_30d     : rolling 30-day net $ value (buy - sell)
      - insider_buy_value_30d     : rolling 30-day $ value of purchases
      - insider_officer_buy_ratio : fraction of officer buys in 30-day window
    """
    feature_cols = [
        "insider_buy_ratio", "insider_net_transactions",
        "insider_activity_5d", "insider_buys_30d", "insider_sells_30d",
        "insider_net_value_30d", "insider_buy_value_30d",
        "insider_officer_buy_ratio",
    ]
    if insider_df.empty:
        for col in feature_cols:
            base[col] = 0.0
        return base

    # Select available columns (gracefully handle old-format data)
    keep_cols = ["date", "insider_buys", "insider_sells",
                 "insider_net_transactions", "insider_buy_ratio"]
    value_cols = ["insider_buy_value", "insider_sell_value",
                  "insider_net_value", "officer_buys", "officer_sells"]
    for vc in value_cols:
        if vc in insider_df.columns:
            keep_cols.append(vc)

    df = insider_df[keep_cols].copy()
    merged = base.merge(df, on="date", how="left")

    # Fill missing days with 0 (no trades = no signal)
    fill_cols = ["insider_buys", "insider_sells", "insider_net_transactions",
                 "insider_buy_ratio"]
    fill_cols += [c for c in value_cols if c in merged.columns]
    for col in fill_cols:
        merged[col] = merged[col].fillna(0)

    # Rolling count windows
    merged["insider_buys_30d"] = merged["insider_buys"].rolling(30, min_periods=1).sum()
    merged["insider_sells_30d"] = merged["insider_sells"].rolling(30, min_periods=1).sum()
    merged["insider_activity_5d"] = (
        (merged["insider_buys"] + merged["insider_sells"]).rolling(5, min_periods=1).sum() > 0
    ).astype(int)

    # Value-based features (stronger signal than count-based)
    if "insider_buy_value" in merged.columns:
        merged["insider_buy_value_30d"] = merged["insider_buy_value"].rolling(30, min_periods=1).sum()
        merged["insider_net_value_30d"] = merged["insider_net_value"].rolling(30, min_periods=1).sum()
    else:
        merged["insider_buy_value_30d"] = 0.0
        merged["insider_net_value_30d"] = 0.0

    # Officer buy ratio (officers buying is a much stronger signal)
    if "officer_buys" in merged.columns:
        officer_buys_30d = merged["officer_buys"].rolling(30, min_periods=1).sum()
        total_30d = merged["insider_buys_30d"].clip(lower=1)
        merged["insider_officer_buy_ratio"] = officer_buys_30d / total_30d
    else:
        merged["insider_officer_buy_ratio"] = 0.0

    # Drop raw columns, keep derived
    drop_cols = ["insider_buys", "insider_sells"] + [c for c in value_cols if c in merged.columns]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged


def _merge_earnings_features(
    base: pd.DataFrame, earnings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Earnings surprise features.

    Signals:
      - earnings_surprise_pct     : % surprise (actual vs estimate)
      - earnings_beat             : 1 if beat, 0 if miss, NaN if no report
      - days_since_earnings       : business days since last earnings report
      - days_to_next_earnings     : business days until next earnings report
      - earnings_surprise_ma      : rolling mean of last 4 surprises
    """
    if earnings_df.empty:
        for col in ["earnings_surprise_pct", "earnings_beat",
                     "days_since_earnings", "days_to_next_earnings",
                     "earnings_surprise_ma"]:
            base[col] = np.nan
        return base

    # Separate reported surprises and scheduled future dates
    edf_report = earnings_df[["date", "earnings_surprise_pct"]].dropna().copy()
    edf_report = edf_report.sort_values("date").drop_duplicates(subset="date", keep="last")

    edf_sched = earnings_df[earnings_df.get("scheduled", False)][["date"]].copy()
    edf_sched = edf_sched.sort_values("date").drop_duplicates(subset="date", keep="last")

    # Merge reported surprises into the base
    merged = base.merge(edf_report, on="date", how="left")

    # Beat flag
    merged["earnings_beat"] = np.where(
        merged["earnings_surprise_pct"].notna(),
        (merged["earnings_surprise_pct"] > 0).astype(float),
        np.nan,
    )

    # Days since last earnings (only reported days count)
    is_report_day = merged["earnings_surprise_pct"].notna()
    merged["_earnings_day_num"] = np.where(is_report_day, range(len(merged)), np.nan)
    merged["_earnings_day_num"] = merged["_earnings_day_num"].ffill()
    merged["days_since_earnings"] = (
        pd.Series(range(len(merged))) - merged["_earnings_day_num"]
    )
    merged = merged.drop(columns=["_earnings_day_num"])

    # Days to next earnings: consider both reported and scheduled dates
    # Build an index marker of next event (report or scheduled)
    marker = pd.Series(np.nan, index=merged.index)
    if not edf_report.empty:
        report_idx = merged[merged["date"].isin(edf_report["date"])].index
        marker.loc[report_idx] = report_idx
    if not edf_sched.empty:
        sched_idx = merged[merged["date"].isin(edf_sched["date"])].index
        marker.loc[sched_idx] = sched_idx

    # Replace marker positions with integer range numbers for bfill
    marker_pos = marker.notna()
    marker_nums = pd.Series(np.nan, index=merged.index)
    if marker_pos.any():
        # assign increasing indices where markers exist
        positions = np.where(marker_pos)[0]
        for i, p in enumerate(positions):
            marker_nums.iloc[p] = p

    marker_nums = marker_nums.bfill()
    merged["days_to_next_earnings"] = marker_nums - pd.Series(range(len(merged)))

    # Forward-fill surprise for inter-earnings periods
    merged["earnings_surprise_pct"] = merged["earnings_surprise_pct"].ffill()

    # Rolling average of last 4 surprises
    surprise_at_report = earnings_df[["date", "earnings_surprise_pct"]].dropna()
    if not surprise_at_report.empty:
        surprise_at_report = surprise_at_report.sort_values("date")
        surprise_at_report["earnings_surprise_ma"] = (
            surprise_at_report["earnings_surprise_pct"].rolling(4, min_periods=1).mean()
        )
        merged = merged.merge(
            surprise_at_report[["date", "earnings_surprise_ma"]],
            on="date", how="left",
        )
        merged["earnings_surprise_ma"] = merged["earnings_surprise_ma"].ffill()
    else:
        merged["earnings_surprise_ma"] = np.nan

    return merged


def _merge_congress_features(
    base: pd.DataFrame, congress_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Congress trades features.

    Signals:
      - congress_net          : buys − sells
      - congress_trade_count  : number of trades
      - congress_activity_30d : any trades in last 30 days
      - congress_buy_signal   : net > 0 in last 30 days
    """
    if congress_df.empty:
        for col in ["congress_net", "congress_trade_count",
                     "congress_activity_30d", "congress_buy_signal"]:
            base[col] = 0
        return base

    df = congress_df[["date", "congress_net", "congress_trade_count"]].copy()
    merged = base.merge(df, on="date", how="left")
    merged["congress_net"] = merged["congress_net"].fillna(0)
    merged["congress_trade_count"] = merged["congress_trade_count"].fillna(0)

    merged["congress_activity_30d"] = (
        merged["congress_trade_count"].rolling(30, min_periods=1).sum() > 0
    ).astype(int)
    merged["congress_buy_signal"] = (
        merged["congress_net"].rolling(30, min_periods=1).sum() > 0
    ).astype(int)

    return merged


def _merge_wiki_features(
    base: pd.DataFrame, wiki_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wikipedia pageview features (retail attention proxy).

    Signals:
      - wiki_pageviews       : daily views
      - wiki_pageviews_ma7   : 7-day moving average
      - wiki_attention_spike : >2σ above 30-day mean
      - wiki_attention_trend : 7d MA / 30d MA (>1 = rising attention)
    """
    if wiki_df.empty:
        for col in ["wiki_pageviews", "wiki_pageviews_ma7",
                     "wiki_attention_spike", "wiki_attention_trend"]:
            base[col] = np.nan
        return base

    df = wiki_df[["date", "wiki_pageviews", "wiki_pageviews_ma7",
                   "wiki_attention_spike"]].copy()
    merged = base.merge(df, on="date", how="left")

    # Forward-fill weekends/gaps
    merged["wiki_pageviews"] = merged["wiki_pageviews"].ffill().fillna(0)
    merged["wiki_pageviews_ma7"] = merged["wiki_pageviews_ma7"].ffill().fillna(0)
    merged["wiki_attention_spike"] = merged["wiki_attention_spike"].ffill().fillna(0)

    # Attention trend
    ma30 = merged["wiki_pageviews"].rolling(30, min_periods=1).mean()
    merged["wiki_attention_trend"] = (
        merged["wiki_pageviews_ma7"] / ma30.clip(lower=1)
    )

    return merged


def _merge_earnings_revision_features(
    base: pd.DataFrame, revisions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyst earnings revision features (historical time series).

    Signals:
      - revision_upgrades_30d   : rolling 30-day upgrade count
      - revision_downgrades_30d : rolling 30-day downgrade count
      - revision_net_30d        : net upgrades − downgrades (30d)
      - revision_momentum       : recent vs longer-term revision pace
      - revision_activity       : any revision in last 5 business days
      - revision_sentiment      : net / total revisions (30d), −1 to +1
    """
    cols = ["revision_upgrades_30d", "revision_downgrades_30d",
            "revision_net_30d", "revision_momentum",
            "revision_activity", "revision_sentiment"]

    if revisions_df.empty:
        for col in cols:
            base[col] = 0.0
        return base

    df = revisions_df[["date", "upgrades", "downgrades", "net_revisions",
                        "total_revisions"]].copy()
    merged = base.merge(df, on="date", how="left")

    # Fill missing days with 0 (no revisions = no signal)
    for col in ["upgrades", "downgrades", "net_revisions", "total_revisions"]:
        merged[col] = merged[col].fillna(0)

    # Rolling windows
    merged["revision_upgrades_30d"] = merged["upgrades"].rolling(30, min_periods=1).sum()
    merged["revision_downgrades_30d"] = merged["downgrades"].rolling(30, min_periods=1).sum()
    merged["revision_net_30d"] = merged["net_revisions"].rolling(30, min_periods=1).sum()

    # Momentum: 30d net vs 90d net (normalised to 30d equivalent)
    net_30 = merged["net_revisions"].rolling(30, min_periods=1).sum()
    net_90 = merged["net_revisions"].rolling(90, min_periods=1).sum()
    merged["revision_momentum"] = net_30 - net_90 / 3

    # Recent activity flag
    merged["revision_activity"] = (
        merged["total_revisions"].rolling(5, min_periods=1).sum() > 0
    ).astype(int)

    # Sentiment: net / total in 30d window, clipped to [-1, +1]
    total_30 = merged["total_revisions"].rolling(30, min_periods=1).sum().clip(lower=1)
    merged["revision_sentiment"] = np.clip(
        merged["revision_net_30d"] / total_30, -1, 1
    )

    # Drop raw columns
    merged = merged.drop(columns=["upgrades", "downgrades", "net_revisions",
                                   "total_revisions"], errors="ignore")
    return merged


def _merge_macro_features(
    base: pd.DataFrame, macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    FRED macro features.

    Merged on date. Also adds rate-of-change features:
      - vix_5d_change       : 5-day VIX change
      - fed_funds_momentum  : 20-day change in fed funds rate
      - oil_5d_return       : 5-day % change in WTI crude
    """
    if macro_df.empty:
        return base

    # Select columns to merge (skip 'ticker' from macro since it's global)
    macro_cols = [c for c in macro_df.columns if c != "ticker"]
    merged = base.merge(macro_df[macro_cols], on="date", how="left")

    # Forward-fill macro data (published with lag)
    numeric_cols = merged.select_dtypes(include="number").columns
    merged[numeric_cols] = merged[numeric_cols].ffill()

    # Rate-of-change features
    if "vix" in merged.columns:
        merged["vix_5d_change"] = merged["vix"].diff(5)
        merged["vix_20d_change"] = merged["vix"].diff(20)

    if "fed_funds_rate" in merged.columns:
        merged["fed_funds_momentum"] = merged["fed_funds_rate"].diff(20)

    if "oil_wti" in merged.columns:
        merged["oil_5d_return"] = merged["oil_wti"].pct_change(5)

    if "high_yield_spread" in merged.columns:
        merged["credit_stress_5d"] = merged["high_yield_spread"].diff(5)

    return merged


# ============================================================================
# Standalone Macro Feature Engineering
# ============================================================================

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

    # De-fragment (many column inserts cause fragmentation)
    df = df.copy()

    logger.info(
        "Macro features: %d rows × %d columns (%d raw → %d derived)",
        len(df), df.shape[1], raw_macro.shape[1], df.shape[1] - raw_macro.shape[1],
    )
    return df
