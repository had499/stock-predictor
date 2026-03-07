"""
Financial Data Loader Module

This module provides functionality to load companies' financial statement data
(income statement, balance sheet, cash flow statement) and key financial metrics
using the yfinance library. It is designed to complement the stock price data
loader with fundamental analysis data useful for stock price estimation.

All statement-fetching methods support temporal filtering via ``start_date``
and ``end_date``, so only reporting periods that fall within the requested
range are returned.

Author: Stock Predictor Project
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Union, List, Optional, Dict, Any
import logging
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataLoader(BaseEstimator, TransformerMixin):
    """
    A comprehensive financial statement data loader that retrieves fundamental
    data for stock prediction tasks. Fetches income statements, balance sheets,
    cash flow statements, and key financial ratios via yfinance.

    Implements sklearn's BaseEstimator and TransformerMixin for easy integration
    with machine learning pipelines.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        frequency: str = 'annual',
        include_income_statement: bool = True,
        include_balance_sheet: bool = True,
        include_cash_flow: bool = True,
        include_key_metrics: bool = True,
    ):
        """
        Initialize the FinancialDataLoader.

        Args:
            symbols (List[str], optional): Stock symbols to load (can be set later).
            start_date (str or datetime, optional): Earliest reporting period to include
                (inclusive).  Periods before this date are excluded.  If ``None``,
                no lower bound is applied.
            end_date (str or datetime, optional): Latest reporting period to include
                (inclusive).  Periods after this date are excluded.  If ``None``,
                no upper bound is applied.
            frequency (str): Data frequency – 'annual' or 'quarterly'.
            include_income_statement (bool): Whether to fetch income statement data.
            include_balance_sheet (bool): Whether to fetch balance sheet data.
            include_cash_flow (bool): Whether to fetch cash flow statement data.
            include_key_metrics (bool): Whether to fetch key financial metrics/ratios.
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.include_income_statement = include_income_statement
        self.include_balance_sheet = include_balance_sheet
        self.include_cash_flow = include_cash_flow
        self.include_key_metrics = include_key_metrics

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
            'frequency': self.frequency,
            'include_income_statement': self.include_income_statement,
            'include_balance_sheet': self.include_balance_sheet,
            'include_cash_flow': self.include_cash_flow,
            'include_key_metrics': self.include_key_metrics,
        }

    def set_params(self, **params) -> 'FinancialDataLoader':
        """Set parameters for the transformer."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def transform(self, X=None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load financial statement data for all configured symbols.

        Args:
            X: Ignored (required by sklearn interface).

        Returns:
            Dict mapping each symbol to a dict of DataFrames:
            {
                'AAPL': {
                    'income_statement': pd.DataFrame,
                    'balance_sheet':    pd.DataFrame,
                    'cash_flow':        pd.DataFrame,
                    'key_metrics':      pd.DataFrame,
                }
            }

        Raises:
            ValueError: If symbols is not set.
        """
        if self.symbols is None:
            raise ValueError("Symbols must be set before calling transform()")

        return self._load_all_financial_data()

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------

    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """
        Fetch the income statement for a single symbol.

        Only reporting periods within [start_date, end_date] are returned.
        If both are ``None``, all available periods are returned.

        Args:
            symbol (str): Stock symbol (e.g. 'AAPL').

        Returns:
            pd.DataFrame: Income statement data with periods as rows.
        """
        return self._fetch_statement(symbol, 'income_statement')

    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """
        Fetch the balance sheet for a single symbol.

        Only reporting periods within [start_date, end_date] are returned.
        If both are ``None``, all available periods are returned.

        Args:
            symbol (str): Stock symbol.

        Returns:
            pd.DataFrame: Balance sheet data with periods as rows.
        """
        return self._fetch_statement(symbol, 'balance_sheet')

    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """
        Fetch the cash flow statement for a single symbol.

        Only reporting periods within [start_date, end_date] are returned.
        If both are ``None``, all available periods are returned.

        Args:
            symbol (str): Stock symbol.

        Returns:
            pd.DataFrame: Cash flow statement data with periods as rows.
        """
        return self._fetch_statement(symbol, 'cash_flow')

    def get_key_metrics(self, symbol: str) -> pd.DataFrame:
        """
        Fetch key financial metrics and valuation ratios for a single symbol.

        Metrics include P/E ratio, EPS, dividend yield, profit margin, ROE,
        ROA, debt-to-equity, current ratio, and more.

        Args:
            symbol (str): Stock symbol.

        Returns:
            pd.DataFrame: Single-row DataFrame with key financial metrics.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            metrics = {
                'symbol': symbol,
                # Valuation
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'enterprise_value': info.get('enterpriseValue'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                'ev_to_revenue': info.get('enterpriseToRevenue'),
                # Profitability
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'ebitda': info.get('ebitda'),
                'ebitda_margins': info.get('ebitdaMargins'),
                # Growth
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
                # Per-share metrics
                'eps_trailing_12m': info.get('trailingEps'),
                'eps_forward': info.get('forwardEps'),
                'book_value_per_share': info.get('bookValue'),
                'revenue_per_share': info.get('revenuePerShare'),
                # Dividends
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'payout_ratio': info.get('payoutRatio'),
                # Leverage / Liquidity
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'total_debt': info.get('totalDebt'),
                'total_cash': info.get('totalCash'),
                'total_cash_per_share': info.get('totalCashPerShare'),
                # Size
                'market_cap': info.get('marketCap'),
                'total_revenue': info.get('totalRevenue'),
                'float_shares': info.get('floatShares'),
                'shares_outstanding': info.get('sharesOutstanding'),
                # Analyst estimates
                'target_high_price': info.get('targetHighPrice'),
                'target_low_price': info.get('targetLowPrice'),
                'target_mean_price': info.get('targetMeanPrice'),
                'recommendation_mean': info.get('recommendationMean'),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions'),
            }

            df = pd.DataFrame([metrics])
            logger.info(f"Successfully fetched key metrics for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching key metrics for {symbol}: {str(e)}")
            return pd.DataFrame([{'symbol': symbol, 'error': str(e)}])

    def get_all_financial_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available financial data for a single symbol.

        Args:
            symbol (str): Stock symbol.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys
            'income_statement', 'balance_sheet', 'cash_flow', 'key_metrics'.
        """
        result: Dict[str, pd.DataFrame] = {}

        if self.include_income_statement:
            result['income_statement'] = self.get_income_statement(symbol)
        if self.include_balance_sheet:
            result['balance_sheet'] = self.get_balance_sheet(symbol)
        if self.include_cash_flow:
            result['cash_flow'] = self.get_cash_flow(symbol)
        if self.include_key_metrics:
            result['key_metrics'] = self.get_key_metrics(symbol)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all_financial_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load financial data for all configured symbols."""
        results: Dict[str, Dict[str, pd.DataFrame]] = {}

        for symbol in self.symbols:
            try:
                logger.info(f"Loading financial data for {symbol}...")
                results[symbol] = self.get_all_financial_data(symbol)
                logger.info(f"Successfully loaded financial data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load financial data for {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}

        return results

    def _fetch_statement(self, symbol: str, statement_type: str) -> pd.DataFrame:
        """
        Generic helper to fetch one of the three financial statements.

        Args:
            symbol (str): Stock ticker symbol.
            statement_type (str): One of 'income_statement', 'balance_sheet',
                or 'cash_flow'.

        Returns:
            pd.DataFrame: Statement data transposed so each row is a reporting
                period.  Rows are filtered to [start_date, end_date] when those
                are set on the loader.
        """
        try:
            ticker = yf.Ticker(symbol)

            if self.frequency == 'quarterly':
                attr_map = {
                    'income_statement': 'quarterly_income_stmt',
                    'balance_sheet': 'quarterly_balance_sheet',
                    'cash_flow': 'quarterly_cashflow',
                }
            else:
                attr_map = {
                    'income_statement': 'income_stmt',
                    'balance_sheet': 'balance_sheet',
                    'cash_flow': 'cashflow',
                }

            attr = attr_map[statement_type]
            raw: pd.DataFrame = getattr(ticker, attr)

            if raw is None or raw.empty:
                logger.warning(f"No {statement_type} data found for {symbol}")
                return pd.DataFrame()

            # Transpose so rows = periods, columns = line items
            df = raw.T.copy()
            df.index.name = 'period'
            df = df.reset_index()
            df.insert(0, 'symbol', symbol)

            # Normalise column names
            df.columns = [
                str(c).strip().lower().replace(' ', '_')
                for c in df.columns
            ]

            # Apply temporal filtering on the 'period' column
            df = self._filter_by_date(df, 'period')

            logger.info(
                f"Successfully fetched {statement_type} ({self.frequency}) for {symbol} "
                f"({len(df)} periods)"
            )
            return df

        except Exception as e:
            logger.error(f"Error fetching {statement_type} for {symbol}: {str(e)}")
            return pd.DataFrame([{'symbol': symbol, 'error': str(e)}])

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        date_column: str,
    ) -> pd.DataFrame:
        """
        Filter *df* so only rows where *date_column* is within
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


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def load_financial_statements(
    symbol: str,
    frequency: str = 'annual',
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load all financial statements for a single symbol.

    Args:
        symbol (str): Stock ticker symbol (e.g. 'AAPL').
        frequency (str): 'annual' or 'quarterly'.
        start_date (str or datetime, optional): Earliest reporting period to include.
        end_date (str or datetime, optional): Latest reporting period to include.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys
        'income_statement', 'balance_sheet', 'cash_flow', 'key_metrics'.
    """
    loader = FinancialDataLoader(
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
    )
    return loader.get_all_financial_data(symbol)


def load_key_metrics(symbol: str) -> pd.DataFrame:
    """
    Convenience function to load key financial metrics for a single symbol.

    Args:
        symbol (str): Stock ticker symbol (e.g. 'AAPL').

    Returns:
        pd.DataFrame: Single-row DataFrame with key financial metrics.
    """
    loader = FinancialDataLoader()
    return loader.get_key_metrics(symbol)
