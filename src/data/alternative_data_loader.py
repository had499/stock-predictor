"""
Alternative Data Loader Module

This module provides functionality to load alternative data sources useful for
stock price estimation. Alternative data complements traditional price and
financial statement data with non-traditional signals such as news sentiment,
macroeconomic indicators, insider transactions, and institutional holdings.

Data sources used:
  - yfinance          : news headlines, insider transactions, institutional holdings
  - pandas_datareader : FRED macroeconomic indicators (GDP, CPI, interest rates, …)

Author: Stock Predictor Project
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Union, List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# pandas_datareader is optional – used only for FRED macroeconomic data
try:
    import pandas_datareader.data as web
    _DATAREADER_AVAILABLE = True
except (ImportError, Exception):  # noqa: BLE001
    _DATAREADER_AVAILABLE = False
    logger.warning(
        "pandas_datareader is not installed or incompatible with the current environment. "
        "Macroeconomic data from FRED will not be available. "
        "Install it with: pip install pandas-datareader"
    )


# ---------------------------------------------------------------------------
# FRED series used for macroeconomic features
# ---------------------------------------------------------------------------
FRED_SERIES: Dict[str, str] = {
    # Growth
    'gdp_growth': 'A191RL1Q225SBEA',        # Real GDP growth rate (quarterly, %)
    # Inflation
    'cpi': 'CPIAUCSL',                       # Consumer Price Index – All Urban
    'core_cpi': 'CPILFESL',                  # CPI ex Food & Energy
    'pce_inflation': 'PCEPI',                # PCE Price Index
    # Interest rates
    'fed_funds_rate': 'FEDFUNDS',            # Effective Federal Funds Rate
    'treasury_10y': 'GS10',                  # 10-Year Treasury Constant Maturity Rate
    'treasury_2y': 'GS2',                    # 2-Year Treasury Constant Maturity Rate
    'yield_curve_spread': 'T10Y2Y',          # 10Y – 2Y Treasury Spread
    # Labour market
    'unemployment_rate': 'UNRATE',           # Civilian Unemployment Rate
    'non_farm_payrolls': 'PAYEMS',           # All Employees: Total Non-Farm
    # Consumer / Business sentiment
    'consumer_sentiment': 'UMCSENT',         # U. of Michigan Consumer Sentiment
    'ism_manufacturing': 'MANEMP',           # Manufacturing Employment (proxy)
    # Credit & Liquidity
    'credit_spread': 'BAA10Y',               # Moody's Baa – 10Y Treasury spread
    'm2_money_supply': 'M2SL',               # M2 Money Stock
    # Housing
    'housing_starts': 'HOUST',              # Housing Starts
    # Oil
    'wti_oil': 'DCOILWTICO',                 # Crude Oil Prices: WTI (daily)
    # Volatility proxy
    'vix': 'VIXCLS',                         # CBOE Volatility Index
}


class AlternativeDataLoader(BaseEstimator, TransformerMixin):
    """
    Loads alternative data sources for stock price estimation.

    Supported data categories:
      - News headlines and metadata from yfinance
      - FRED macroeconomic indicators (requires pandas_datareader)
      - Insider transactions from yfinance
      - Institutional holder data from yfinance

    All data categories support temporal filtering via ``start_date`` and
    ``end_date``.  When set, only records whose date falls within that range
    are returned:

    - **news**: filtered by ``publish_time``
    - **macro**: filtered by the FRED observation date
    - **insider_transactions**: filtered by the transaction date
    - **institutional_holders**: filtered by the date reported

    Implements sklearn's BaseEstimator and TransformerMixin for easy integration
    with machine learning pipelines.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        include_news: bool = True,
        include_macro: bool = True,
        include_insider_transactions: bool = True,
        include_institutional_holders: bool = True,
        macro_series: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the AlternativeDataLoader.

        Args:
            symbols (List[str], optional): Stock symbols for company-specific data.
            start_date (str or datetime, optional): Start of the date range (inclusive).
                Applied to all data categories:
                - news: articles published on or after this date
                - macro: FRED observations on or after this date
                - insider_transactions: transactions on or after this date
                - institutional_holders: positions reported on or after this date
                Defaults to 5 years ago for macro data when ``None``.
            end_date (str or datetime, optional): End of the date range (inclusive).
                Applied to all data categories (see ``start_date``).
                Defaults to today for macro data when ``None``.
            include_news (bool): Fetch recent news headlines from yfinance.
            include_macro (bool): Fetch FRED macroeconomic indicators.
            include_insider_transactions (bool): Fetch insider transaction data.
            include_institutional_holders (bool): Fetch institutional holder data.
            macro_series (Dict[str, str], optional): Custom mapping of
                {label: FRED_series_id} to fetch. Defaults to FRED_SERIES.
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.include_news = include_news
        self.include_macro = include_macro
        self.include_insider_transactions = include_insider_transactions
        self.include_institutional_holders = include_institutional_holders
        self.macro_series = macro_series if macro_series is not None else FRED_SERIES

        warnings.filterwarnings('ignore', category=FutureWarning)

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X=None, y=None):
        """Fit the transformer (no-op). Required by sklearn interface."""
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for the transformer."""
        return {
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'include_news': self.include_news,
            'include_macro': self.include_macro,
            'include_insider_transactions': self.include_insider_transactions,
            'include_institutional_holders': self.include_institutional_holders,
            'macro_series': self.macro_series,
        }

    def set_params(self, **params) -> 'AlternativeDataLoader':
        """Set parameters for the transformer."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def transform(self, X=None) -> Dict[str, Any]:
        """
        Load all configured alternative data.

        Args:
            X: Ignored (required by sklearn interface).

        Returns:
            Dict with any combination of the following keys depending on
            the loader's configuration:
            {
                'news':                   Dict[str, pd.DataFrame],
                'macro':                  pd.DataFrame,
                'insider_transactions':   Dict[str, pd.DataFrame],
                'institutional_holders':  Dict[str, pd.DataFrame],
            }
        """
        result: Dict[str, Any] = {}

        if self.include_news and self.symbols:
            result['news'] = self._load_news_for_all_symbols()

        if self.include_macro:
            result['macro'] = self.get_macro_indicators()

        if self.include_insider_transactions and self.symbols:
            result['insider_transactions'] = self._load_insider_transactions_for_all()

        if self.include_institutional_holders and self.symbols:
            result['institutional_holders'] = self._load_institutional_holders_for_all()

        return result

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------

    def get_news(self, symbol: str) -> pd.DataFrame:
        """
        Fetch recent news headlines for a stock symbol via yfinance.

        The returned DataFrame contains news metadata that can be used as a
        basis for sentiment analysis or as a signal for model features.
        Articles are filtered to [start_date, end_date] when those are set
        on the loader.

        Args:
            symbol (str): Stock ticker symbol (e.g. 'AAPL').

        Returns:
            pd.DataFrame: News articles with columns:
                uuid, title, publisher, link, publish_time,
                type, related_tickers, symbol.
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                logger.warning(f"No news found for {symbol}")
                return pd.DataFrame()

            records = []
            for item in news:
                content = item.get('content', {})
                records.append({
                    'symbol': symbol,
                    'uuid': item.get('id') or content.get('id'),
                    'title': content.get('title') or item.get('title'),
                    'publisher': (content.get('provider') or {}).get('displayName') or item.get('publisher'),
                    'link': (content.get('canonicalUrl') or {}).get('url') or item.get('link'),
                    'publish_time': (
                        content.get('pubDate')
                        or (
                            datetime.fromtimestamp(item['providerPublishTime']).isoformat()
                            if item.get('providerPublishTime') else None
                        )
                    ),
                    'type': content.get('contentType') or item.get('type'),
                    'related_tickers': ', '.join(
                        t.get('symbol', '') for t in content.get('finance', {}).get('stockTickers', [])
                    ) or ', '.join(item.get('relatedTickers', [])),
                })

            df = pd.DataFrame(records)

            # Apply temporal filtering on publish_time
            df = self._filter_by_date(df, 'publish_time')

            logger.info(f"Successfully fetched {len(df)} news articles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return pd.DataFrame([{'symbol': symbol, 'error': str(e)}])

    def get_macro_indicators(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        series: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators from FRED via pandas_datareader.

        Includes GDP growth, CPI, federal funds rate, unemployment rate,
        Treasury yields, consumer sentiment, credit spreads, and more.

        Args:
            start_date (str or datetime, optional): Override the loader's start_date.
            end_date (str or datetime, optional): Override the loader's end_date.
            series (Dict[str, str], optional): Custom {label: FRED_series_id} mapping.

        Returns:
            pd.DataFrame: Macroeconomic data indexed by date, one column per series.
                Returns an empty DataFrame with an explanatory 'error' column if
                pandas_datareader is not installed.
        """
        if not _DATAREADER_AVAILABLE:
            logger.warning("pandas_datareader is not available; skipping macro data.")
            return pd.DataFrame(
                columns=['date', 'error'],
                data=[['N/A', 'pandas_datareader is not installed']],
            )

        start = start_date or self.start_date or (datetime.today() - timedelta(days=365 * 5))
        end = end_date or self.end_date or datetime.today()
        fred_series = series or self.macro_series

        frames: Dict[str, pd.Series] = {}
        for label, series_id in fred_series.items():
            try:
                s = web.DataReader(series_id, 'fred', start, end)
                frames[label] = s.iloc[:, 0]
                logger.info(f"Fetched FRED series '{series_id}' as '{label}'")
            except Exception as e:
                logger.warning(f"Could not fetch FRED series '{series_id}' ({label}): {e}")

        if not frames:
            logger.warning("No FRED series could be fetched.")
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index.name = 'date'
        df = df.reset_index()
        logger.info(
            f"Successfully fetched {len(df.columns) - 1} macroeconomic series "
            f"({len(df)} rows)"
        )
        return df

    def get_insider_transactions(self, symbol: str) -> pd.DataFrame:
        """
        Fetch insider transaction data for a stock symbol via yfinance.

        Insider buying and selling activity can serve as an alternative signal
        for stock price direction.  Results are filtered to [start_date, end_date]
        when those are set on the loader.

        Args:
            symbol (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: Insider transactions with date, insider name, shares,
                value, and transaction type.
        """
        try:
            ticker = yf.Ticker(symbol)
            transactions = ticker.insider_transactions

            if transactions is None or transactions.empty:
                logger.warning(f"No insider transactions found for {symbol}")
                return pd.DataFrame()

            df = transactions.copy()
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
            df.insert(0, 'symbol', symbol)

            # Detect the date column (yfinance uses 'start_date' or 'date')
            date_col = next(
                (c for c in df.columns if c in ('date', 'start_date', 'transaction_date')),
                None,
            )
            if date_col:
                df = self._filter_by_date(df, date_col)

            logger.info(f"Successfully fetched {len(df)} insider transactions for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching insider transactions for {symbol}: {str(e)}")
            return pd.DataFrame([{'symbol': symbol, 'error': str(e)}])

    def get_institutional_holders(self, symbol: str) -> pd.DataFrame:
        """
        Fetch institutional holder data for a stock symbol via yfinance.

        Large institutional ownership changes can signal shifts in market
        sentiment and be used as alternative data features.  Results are
        filtered to [start_date, end_date] when those are set on the loader.

        Args:
            symbol (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: Institutional holders with holder name, shares held,
                date reported, percent of shares out, and value.
        """
        try:
            ticker = yf.Ticker(symbol)
            holders = ticker.institutional_holders

            if holders is None or holders.empty:
                logger.warning(f"No institutional holders found for {symbol}")
                return pd.DataFrame()

            df = holders.copy()
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
            df.insert(0, 'symbol', symbol)

            # Apply temporal filtering on the 'date_reported' column
            date_col = next(
                (c for c in df.columns if c in ('date_reported', 'date')),
                None,
            )
            if date_col:
                df = self._filter_by_date(df, date_col)

            logger.info(f"Successfully fetched {len(df)} institutional holders for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching institutional holders for {symbol}: {str(e)}")
            return pd.DataFrame([{'symbol': symbol, 'error': str(e)}])

    def get_all_alternative_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available alternative data for a single symbol.

        Args:
            symbol (str): Stock ticker symbol.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys 'news',
            'insider_transactions', 'institutional_holders'.
        """
        result: Dict[str, pd.DataFrame] = {}

        if self.include_news:
            result['news'] = self.get_news(symbol)
        if self.include_insider_transactions:
            result['insider_transactions'] = self.get_insider_transactions(symbol)
        if self.include_institutional_holders:
            result['institutional_holders'] = self.get_institutional_holders(symbol)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        date_column: str,
    ) -> pd.DataFrame:
        """
        Filter *df* so only rows where *date_column* falls within
        [self.start_date, self.end_date] are kept.

        If both ``start_date`` and ``end_date`` are ``None``, the DataFrame
        is returned unchanged.  Timezone-aware timestamps in *date_column* are
        normalised to UTC and then made timezone-naive before comparison so
        that user-supplied dates (which may be naive strings like '2023-01-01')
        are always comparable.

        Args:
            df (pd.DataFrame): DataFrame to filter.
            date_column (str): Name of the column holding datetime values.

        Returns:
            pd.DataFrame: Filtered DataFrame (copy).
        """
        if self.start_date is None and self.end_date is None:
            return df

        if date_column not in df.columns or df.empty:
            return df

        dates = pd.to_datetime(df[date_column], errors='coerce')

        # Normalise tz-aware datetimes to UTC then strip timezone so that
        # naive user-supplied dates are always comparable.
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert('UTC').dt.tz_localize(None)

        # Floor to day so that date-only bounds (e.g. '2024-01-14') include
        # all records on that calendar day regardless of their intraday time.
        dates = dates.dt.normalize()

        mask = pd.Series(True, index=df.index)
        if self.start_date is not None:
            start = pd.Timestamp(self.start_date)
            mask &= dates >= start
        if self.end_date is not None:
            end = pd.Timestamp(self.end_date)
            mask &= dates <= end

        return df[mask].reset_index(drop=True)

    def _load_news_for_all_symbols(self) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            results[symbol] = self.get_news(symbol)
        return results

    def _load_insider_transactions_for_all(self) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            results[symbol] = self.get_insider_transactions(symbol)
        return results

    def _load_institutional_holders_for_all(self) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            results[symbol] = self.get_institutional_holders(symbol)
        return results


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def load_macro_indicators(
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    series: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to load FRED macroeconomic indicators.

    Args:
        start_date (str or datetime, optional): Start date for data retrieval.
        end_date (str or datetime, optional): End date for data retrieval.
        series (Dict[str, str], optional): Custom {label: FRED_series_id} mapping.

    Returns:
        pd.DataFrame: Macroeconomic data indexed by date.
    """
    loader = AlternativeDataLoader(start_date=start_date, end_date=end_date)
    return loader.get_macro_indicators(series=series)


def load_news(symbol: str) -> pd.DataFrame:
    """
    Convenience function to load recent news for a symbol.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: News articles with title, publisher, publish_time, and link.
    """
    loader = AlternativeDataLoader()
    return loader.get_news(symbol)


def load_insider_transactions(symbol: str) -> pd.DataFrame:
    """
    Convenience function to load insider transactions for a symbol.

    Args:
        symbol (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: Insider transaction records.
    """
    loader = AlternativeDataLoader()
    return loader.get_insider_transactions(symbol)
