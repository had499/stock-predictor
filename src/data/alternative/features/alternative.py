"""
Per-ticker alternative data feature engineering.

Transforms raw scraped alternative data (insider trades, earnings, congress
trades, Wikipedia pageviews, FRED macro, analyst revisions) into a
model-ready daily (date, ticker) DataFrame via build_alternative_features()
and its _merge_*() helper functions.

Exports: build_alternative_features
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

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

    if "short_interest" in raw_data:
        base = _merge_short_interest_features(base, raw_data["short_interest"])

    if "price_targets" in raw_data:
        base = _merge_price_target_features(base, raw_data["price_targets"])

    return base


# ---------------------------------------------------------------------------
# Merge helpers – each creates + joins features for one source
# ---------------------------------------------------------------------------

def _merge_insider_features(
    base: pd.DataFrame, insider_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Insider trading features from EDGAR Form 4 filings.

    Open-market trade signals (P / S codes):
      - insider_buy_ratio         : fraction of buys among P+S trades
      - insider_net_transactions  : buys − sells
      - insider_activity_5d       : any P/S trade in last 5 biz days
      - insider_buys_30d          : rolling 30-day buy count
      - insider_sells_30d         : rolling 30-day sell count
      - insider_net_value_30d     : rolling 30-day net $ value (buy − sell)
      - insider_buy_value_30d     : rolling 30-day $ value of purchases
      - insider_officer_buy_ratio : fraction of officer buys in 30-day window

    Compensation structure signals (A / F / M / G codes):
      - grant_intensity_30d       : rolling 30-day grant count (A codes)
      - tax_event_30d             : rolling 30-day tax-withholding count (F codes)
      - option_exercise_30d       : rolling 30-day option exercise count (M codes)
      - total_disposal_30d        : sells + tax-withheld in 30d (all shares leaving
                                    insider hands, voluntary or forced)
      - compensation_style_90d    : grants / (grants + exercises) over 90 days,
                                    0 = option-heavy, 1 = RSU-heavy, NaN if no activity
    """
    feature_cols = [
        "insider_buy_ratio", "insider_net_transactions",
        "insider_activity_5d", "insider_buys_30d", "insider_sells_30d",
        "insider_net_value_30d", "insider_buy_value_30d",
        "insider_officer_buy_ratio",
        "grant_intensity_30d", "tax_event_30d", "option_exercise_30d",
        "total_disposal_30d", "compensation_style_90d",
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
    comp_cols = ["grant_count", "grant_shares", "tax_withheld_count",
                 "tax_withheld_shares", "option_exercises", "gift_count"]
    for vc in value_cols + comp_cols:
        if vc in insider_df.columns:
            keep_cols.append(vc)

    df = insider_df[keep_cols].copy()
    merged = base.merge(df, on="date", how="left")

    # Fill missing days with 0 (no trades = no signal)
    fill_cols = ["insider_buys", "insider_sells", "insider_net_transactions",
                 "insider_buy_ratio"]
    fill_cols += [c for c in value_cols + comp_cols if c in merged.columns]
    for col in fill_cols:
        merged[col] = merged[col].fillna(0)

    # ── Open-market rolling features ──
    merged["insider_buys_30d"] = merged["insider_buys"].rolling(30, min_periods=1).sum()
    merged["insider_sells_30d"] = merged["insider_sells"].rolling(30, min_periods=1).sum()
    merged["insider_activity_5d"] = (
        (merged["insider_buys"] + merged["insider_sells"]).rolling(5, min_periods=1).sum() > 0
    ).astype(int)

    if "insider_buy_value" in merged.columns:
        merged["insider_buy_value_30d"] = merged["insider_buy_value"].rolling(30, min_periods=1).sum()
        merged["insider_net_value_30d"] = merged["insider_net_value"].rolling(30, min_periods=1).sum()
    else:
        merged["insider_buy_value_30d"] = 0.0
        merged["insider_net_value_30d"] = 0.0

    if "officer_buys" in merged.columns:
        officer_buys_30d = merged["officer_buys"].rolling(30, min_periods=1).sum()
        total_30d = merged["insider_buys_30d"].clip(lower=1)
        merged["insider_officer_buy_ratio"] = officer_buys_30d / total_30d
    else:
        merged["insider_officer_buy_ratio"] = 0.0

    # ── Compensation structure features ──
    if "grant_count" in merged.columns:
        merged["grant_intensity_30d"] = merged["grant_count"].rolling(30, min_periods=1).sum()
    else:
        merged["grant_intensity_30d"] = 0.0

    if "tax_withheld_count" in merged.columns:
        merged["tax_event_30d"] = merged["tax_withheld_count"].rolling(30, min_periods=1).sum()
    else:
        merged["tax_event_30d"] = 0.0

    if "option_exercises" in merged.columns:
        merged["option_exercise_30d"] = merged["option_exercises"].rolling(30, min_periods=1).sum()
    else:
        merged["option_exercise_30d"] = 0.0

    # Total disposal = voluntary sells + forced tax-withholding sells
    sell_col = merged["insider_sells"] if "insider_sells" in merged.columns else pd.Series(0, index=merged.index)
    tax_col = merged["tax_withheld_count"] if "tax_withheld_count" in merged.columns else pd.Series(0, index=merged.index)
    merged["total_disposal_30d"] = (sell_col + tax_col).rolling(30, min_periods=1).sum()

    # Compensation style: rolling 90d grants / (grants + exercises)
    # Ranges 0 (all option exercises) → 1 (all grants/RSUs), NaN if no activity
    if "grant_count" in merged.columns and "option_exercises" in merged.columns:
        grants_90d = merged["grant_count"].rolling(90, min_periods=1).sum()
        exercises_90d = merged["option_exercises"].rolling(90, min_periods=1).sum()
        total_comp_90d = grants_90d + exercises_90d
        merged["compensation_style_90d"] = np.where(
            total_comp_90d > 0,
            grants_90d / total_comp_90d,
            np.nan,
        )
    else:
        merged["compensation_style_90d"] = np.nan

    # Drop raw columns, keep derived
    drop_cols = (["insider_buys", "insider_sells"]
                 + [c for c in value_cols if c in merged.columns]
                 + [c for c in comp_cols if c in merged.columns])
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


def _merge_short_interest_features(
    base: pd.DataFrame, si_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    FINRA short interest features (bi-monthly, forward-filled to daily).

    Signals:
      - days_to_cover          : short ratio (shares short / avg daily volume)
      - short_pct_chg          : % change in short interest vs prior period
      - short_squeeze_risk     : days_to_cover rising while short_pct_chg > 0
    """
    feature_cols = ["days_to_cover", "short_pct_chg", "short_squeeze_risk"]
    if si_df.empty:
        for col in feature_cols:
            base[col] = np.nan
        return base

    df = si_df[["date", "days_to_cover", "short_change_pct"]].copy()
    df = df.rename(columns={"short_change_pct": "short_pct_chg"})
    merged = base.merge(df, on="date", how="left")

    # Forward-fill between bi-monthly readings
    merged["days_to_cover"] = merged["days_to_cover"].ffill()
    merged["short_pct_chg"] = merged["short_pct_chg"].ffill()

    # Squeeze risk: short interest grew AND days-to-cover is elevated (>5)
    merged["short_squeeze_risk"] = (
        (merged["short_pct_chg"] > 0) & (merged["days_to_cover"] > 5)
    ).astype(float)
    # Mask where we have no data at all
    no_data = merged["days_to_cover"].isna()
    merged.loc[no_data, "short_squeeze_risk"] = np.nan

    return merged


def _merge_price_target_features(
    base: pd.DataFrame, pt_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    FMP analyst price target features (forward-filled to daily).

    Signals:
      - pt_mean                : consensus mean price target (forward-filled)
      - pt_count               : number of analyst targets on record
      - pt_revision_direction  : last revision direction (+1 raised / -1 lowered)
      - pt_dispersion          : (pt_high - pt_low) / pt_mean — analyst disagreement
      - pt_momentum_60d        : change in pt_mean over 60 trading days
    """
    feature_cols = ["pt_mean", "pt_count", "pt_revision_direction",
                    "pt_dispersion", "pt_momentum_60d"]
    if pt_df.empty:
        for col in feature_cols:
            base[col] = np.nan
        return base

    df = pt_df[["date", "pt_mean", "pt_high", "pt_low",
                 "pt_count", "pt_revision_direction"]].copy()
    merged = base.merge(df, on="date", how="left")

    # Forward-fill — targets remain valid until the next revision
    for col in ["pt_mean", "pt_high", "pt_low", "pt_count", "pt_revision_direction"]:
        merged[col] = merged[col].ffill()

    # Analyst disagreement: normalised spread between high and low targets
    merged["pt_dispersion"] = np.where(
        merged["pt_mean"] > 0,
        (merged["pt_high"] - merged["pt_low"]) / merged["pt_mean"],
        np.nan,
    )

    # Momentum: change in consensus target over 60 trading days
    merged["pt_momentum_60d"] = merged["pt_mean"].diff(60)

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
