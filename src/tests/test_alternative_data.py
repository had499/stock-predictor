"""
Tests for the AlternativeDataLoader module.

These tests use mock data and unittest.mock to avoid live network calls,
consistent with the project's approach of keeping tests self-contained.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.alternative_data_loader import (
    AlternativeDataLoader,
    load_news,
    load_macro_indicators,
    load_insider_transactions,
    FRED_SERIES,
)


# ---------------------------------------------------------------------------
# Helpers – fake yfinance and datareader data
# ---------------------------------------------------------------------------

def _make_news_items():
    """Return a list of fake news items resembling yfinance output."""
    return [
        {
            'id': 'abc123',
            'content': {
                'id': 'abc123',
                'title': 'Apple reports record earnings',
                'provider': {'displayName': 'Reuters'},
                'canonicalUrl': {'url': 'https://example.com/article1'},
                'pubDate': '2024-01-15T10:00:00Z',
                'contentType': 'STORY',
                'finance': {'stockTickers': [{'symbol': 'AAPL'}]},
            },
        },
        {
            'id': 'def456',
            'content': {
                'id': 'def456',
                'title': 'Fed holds rates steady',
                'provider': {'displayName': 'Bloomberg'},
                'canonicalUrl': {'url': 'https://example.com/article2'},
                'pubDate': '2024-01-14T08:30:00Z',
                'contentType': 'STORY',
                'finance': {'stockTickers': [{'symbol': 'AAPL'}]},
            },
        },
    ]


def _make_insider_transactions() -> pd.DataFrame:
    """Return a fake insider transactions DataFrame."""
    return pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-10', '2024-01-05']),
        'Insider': ['Tim Cook', 'Luca Maestri'],
        'Transaction': ['Sale', 'Sale'],
        'Shares': [100_000, 50_000],
        'Value': [18_000_000, 9_000_000],
    })


def _make_institutional_holders() -> pd.DataFrame:
    """Return a fake institutional holders DataFrame."""
    return pd.DataFrame({
        'Holder': ['Vanguard Group', 'BlackRock Inc.'],
        'Shares': [1_200_000_000, 1_050_000_000],
        'Date Reported': pd.to_datetime(['2023-12-31', '2023-12-31']),
        '% Out': [7.9, 6.9],
        'Value': [220_000_000_000, 192_000_000_000],
    })


def _make_ticker_mock() -> MagicMock:
    """Build a MagicMock resembling a yfinance Ticker with stub alternative data."""
    mock = MagicMock()
    mock.news = _make_news_items()
    mock.insider_transactions = _make_insider_transactions()
    mock.institutional_holders = _make_institutional_holders()
    return mock


def _make_fred_series() -> pd.DataFrame:
    """Return a fake FRED time-series DataFrame."""
    idx = pd.date_range('2023-01-01', periods=12, freq='MS')
    return pd.DataFrame({'FEDFUNDS': np.linspace(4.5, 5.5, 12)}, index=idx)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_get_news():
    """AlternativeDataLoader should return a DataFrame with news articles."""
    print("Testing news fetching...")

    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = AlternativeDataLoader()
        df = loader.get_news('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "News DataFrame should not be empty"
    assert 'symbol' in df.columns, "Symbol column should be present"
    assert 'title' in df.columns, "Title column should be present"
    assert 'publisher' in df.columns, "Publisher column should be present"
    assert 'link' in df.columns, "Link column should be present"
    assert 'publish_time' in df.columns, "publish_time column should be present"
    assert len(df) == 2, "Should have 2 articles"
    print("✓ News fetching test passed")
    return True


def test_get_news_empty():
    """AlternativeDataLoader should return an empty DataFrame when no news found."""
    print("Testing news fetching with empty result...")

    mock = _make_ticker_mock()
    mock.news = []

    with patch('data.alternative_data_loader.yf.Ticker', return_value=mock):
        loader = AlternativeDataLoader()
        df = loader.get_news('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame even when empty"
    assert df.empty, "Should return empty DataFrame when no news"
    print("✓ Empty news test passed")
    return True


def test_get_insider_transactions():
    """AlternativeDataLoader should return a DataFrame of insider transactions."""
    print("Testing insider transactions fetching...")

    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = AlternativeDataLoader()
        df = loader.get_insider_transactions('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "Should not be empty"
    assert 'symbol' in df.columns, "Symbol column should be present"
    assert len(df) == 2, "Should have 2 transactions"
    print("✓ Insider transactions test passed")
    return True


def test_get_institutional_holders():
    """AlternativeDataLoader should return a DataFrame of institutional holders."""
    print("Testing institutional holders fetching...")

    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = AlternativeDataLoader()
        df = loader.get_institutional_holders('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "Should not be empty"
    assert 'symbol' in df.columns, "Symbol column should be present"
    assert len(df) == 2, "Should have 2 institutional holders"
    print("✓ Institutional holders test passed")
    return True


def test_get_macro_indicators_with_datareader():
    """get_macro_indicators should return a DataFrame when pandas_datareader is available."""
    print("Testing macro indicators fetching...")

    fake_fred = _make_fred_series()

    import data.alternative_data_loader as alt_mod
    import types

    # Inject a fake web module into the namespace so the patch target exists
    fake_web = types.SimpleNamespace(DataReader=lambda *a, **kw: fake_fred)

    original_available = alt_mod._DATAREADER_AVAILABLE
    original_web = getattr(alt_mod, 'web', None)
    try:
        alt_mod._DATAREADER_AVAILABLE = True
        alt_mod.web = fake_web
        loader = alt_mod.AlternativeDataLoader(
            start_date='2023-01-01',
            end_date='2023-12-31',
        )
        df = loader.get_macro_indicators(series={'fed_funds_rate': 'FEDFUNDS'})
    finally:
        alt_mod._DATAREADER_AVAILABLE = original_available
        if original_web is None:
            if hasattr(alt_mod, 'web'):
                del alt_mod.web
        else:
            alt_mod.web = original_web

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "Macro DataFrame should not be empty"
    assert 'date' in df.columns, "Date column should be present"
    assert 'fed_funds_rate' in df.columns, "fed_funds_rate should be a column"
    print("✓ Macro indicators test passed")
    return True


def test_get_macro_indicators_no_datareader():
    """get_macro_indicators should gracefully degrade when datareader is absent."""
    print("Testing macro indicators without pandas_datareader...")

    with patch('data.alternative_data_loader._DATAREADER_AVAILABLE', False):
        loader = AlternativeDataLoader()
        df = loader.get_macro_indicators()

    assert isinstance(df, pd.DataFrame), "Should still return a DataFrame"
    assert 'error' in df.columns, "Should have an error column"
    print("✓ Macro without datareader test passed")
    return True


def test_get_all_alternative_data():
    """get_all_alternative_data should return a dict with all expected keys."""
    print("Testing get_all_alternative_data...")

    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = AlternativeDataLoader()
        result = loader.get_all_alternative_data('AAPL')

    assert isinstance(result, dict), "Should return a dict"
    assert 'news' in result, "Should include news"
    assert 'insider_transactions' in result, "Should include insider_transactions"
    assert 'institutional_holders' in result, "Should include institutional_holders"
    for key, df in result.items():
        assert isinstance(df, pd.DataFrame), f"{key} should be a DataFrame"
    print("✓ get_all_alternative_data test passed")
    return True


def test_transform_interface():
    """transform() should load data for all configured symbols."""
    print("Testing sklearn transform interface...")

    fake_fred = _make_fred_series()

    import data.alternative_data_loader as alt_mod
    import types

    fake_web = types.SimpleNamespace(DataReader=lambda *a, **kw: fake_fred)
    original_available = alt_mod._DATAREADER_AVAILABLE
    original_web = getattr(alt_mod, 'web', None)
    try:
        alt_mod._DATAREADER_AVAILABLE = True
        alt_mod.web = fake_web
        with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
            loader = alt_mod.AlternativeDataLoader(
                symbols=['AAPL', 'MSFT'],
                start_date='2023-01-01',
                end_date='2023-12-31',
            )
            result = loader.transform(None)
    finally:
        alt_mod._DATAREADER_AVAILABLE = original_available
        if original_web is None:
            if hasattr(alt_mod, 'web'):
                del alt_mod.web
        else:
            alt_mod.web = original_web

    assert isinstance(result, dict), "Should return a dict"
    assert 'news' in result, "Should include news"
    assert 'macro' in result, "Should include macro"
    assert 'insider_transactions' in result, "Should include insider_transactions"
    assert 'institutional_holders' in result, "Should include institutional_holders"
    assert 'AAPL' in result['news'], "AAPL news should be present"
    assert 'MSFT' in result['news'], "MSFT news should be present"
    print("✓ Transform interface test passed")
    return True


def test_get_set_params():
    """get_params / set_params should work correctly."""
    print("Testing get_params / set_params...")

    loader = AlternativeDataLoader(include_news=True, include_macro=False)
    params = loader.get_params()

    assert 'include_news' in params
    assert params['include_news'] is True
    assert params['include_macro'] is False

    loader.set_params(include_macro=True)
    assert loader.include_macro is True, "set_params should update the attribute"
    print("✓ get_params / set_params test passed")
    return True


def test_selective_data_types():
    """AlternativeDataLoader should respect include_* flags in transform."""
    print("Testing selective data type loading...")

    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = AlternativeDataLoader(
            symbols=['AAPL'],
            include_news=True,
            include_macro=False,
            include_insider_transactions=False,
            include_institutional_holders=False,
        )
        result = loader.transform(None)

    assert 'news' in result, "news should be included"
    assert 'macro' not in result, "macro should be excluded"
    assert 'insider_transactions' not in result, "insider_transactions should be excluded"
    assert 'institutional_holders' not in result, "institutional_holders should be excluded"
    print("✓ Selective data type loading test passed")
    return True


def test_convenience_function_load_news():
    """load_news convenience function should return a DataFrame."""
    print("Testing load_news convenience function...")

    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        df = load_news('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty
    print("✓ load_news convenience function test passed")
    return True


def test_convenience_function_load_insider_transactions():
    """load_insider_transactions convenience function should return a DataFrame."""
    print("Testing load_insider_transactions convenience function...")

    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        df = load_insider_transactions('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty
    print("✓ load_insider_transactions convenience function test passed")
    return True


def test_fred_series_catalog():
    """The FRED_SERIES catalog should contain a meaningful set of indicators."""
    print("Testing FRED series catalog...")

    assert isinstance(FRED_SERIES, dict), "FRED_SERIES should be a dict"
    assert len(FRED_SERIES) >= 10, "Should include at least 10 macroeconomic series"
    assert 'fed_funds_rate' in FRED_SERIES, "fed_funds_rate should be in catalog"
    assert 'cpi' in FRED_SERIES, "cpi should be in catalog"
    assert 'unemployment_rate' in FRED_SERIES, "unemployment_rate should be in catalog"
    assert 'gdp_growth' in FRED_SERIES, "gdp_growth should be in catalog"
    assert 'vix' in FRED_SERIES, "vix should be in catalog"
    print("✓ FRED series catalog test passed")
    return True


# ---------------------------------------------------------------------------
# Temporal filtering tests
# ---------------------------------------------------------------------------

def test_news_temporal_filtering_start_date():
    """AlternativeDataLoader should exclude news published before start_date."""
    print("Testing news temporal filtering – start_date...")

    # Mock has articles published on 2024-01-15 and 2024-01-14
    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        # Only articles from 2024-01-15 onward
        loader = AlternativeDataLoader(start_date='2024-01-15')
        df = loader.get_news('AAPL')

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Should have articles on or after 2024-01-15"
    # The 2024-01-14 article must be excluded
    assert len(df) == 1, "Only one article falls on or after 2024-01-15"
    assert 'Apple reports record earnings' in df['title'].values
    print("✓ News temporal filtering – start_date test passed")
    return True


def test_news_temporal_filtering_end_date():
    """AlternativeDataLoader should exclude news published after end_date."""
    print("Testing news temporal filtering – end_date...")

    # Mock has articles on 2024-01-15 and 2024-01-14
    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        # Only articles up to 2024-01-14
        loader = AlternativeDataLoader(end_date='2024-01-14')
        df = loader.get_news('AAPL')

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Should have articles on or before 2024-01-14"
    assert len(df) == 1, "Only one article falls on or before 2024-01-14"
    assert 'Fed holds rates steady' in df['title'].values
    print("✓ News temporal filtering – end_date test passed")
    return True


def test_news_temporal_no_filter_returns_all():
    """AlternativeDataLoader without dates should return all news."""
    print("Testing news – no date filter returns all...")

    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = AlternativeDataLoader()  # no dates
        df = loader.get_news('AAPL')

    assert len(df) == 2, "Both news articles should be returned when no date filter"
    print("✓ News – no date filter returns all test passed")
    return True


def test_insider_transactions_temporal_filtering():
    """AlternativeDataLoader should filter insider transactions by date."""
    print("Testing insider transactions temporal filtering...")

    # Mock has transactions on 2024-01-10 and 2024-01-05
    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = AlternativeDataLoader(start_date='2024-01-08')
        df = loader.get_insider_transactions('AAPL')

    assert isinstance(df, pd.DataFrame)
    # Only the 2024-01-10 transaction should remain
    assert len(df) == 1, "Only one transaction falls on or after 2024-01-08"
    print("✓ Insider transactions temporal filtering test passed")
    return True


def test_institutional_holders_temporal_filtering():
    """AlternativeDataLoader should filter institutional holders by date_reported."""
    print("Testing institutional holders temporal filtering...")

    # Mock has two holders both with date_reported 2023-12-31
    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        # End date before the reporting date – nothing should match
        loader = AlternativeDataLoader(end_date='2023-06-30')
        df = loader.get_institutional_holders('AAPL')

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0, "No holders should be reported before 2023-06-30"
    print("✓ Institutional holders temporal filtering test passed")
    return True


def test_temporal_no_dates_returns_all_alternative():
    """AlternativeDataLoader without dates returns all rows for every data type."""
    print("Testing no-date filter returns all rows (alternative data)...")

    with patch('data.alternative_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = AlternativeDataLoader()
        news = loader.get_news('AAPL')
        transactions = loader.get_insider_transactions('AAPL')
        holders = loader.get_institutional_holders('AAPL')

    assert len(news) == 2, "All 2 news articles should be returned"
    assert len(transactions) == 2, "All 2 insider transactions should be returned"
    assert len(holders) == 2, "All 2 institutional holders should be returned"
    print("✓ No-date filter returns all rows (alternative data) test passed")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=== TESTING ALTERNATIVE DATA LOADER ===\n")

    tests = [
        test_get_news,
        test_get_news_empty,
        test_get_insider_transactions,
        test_get_institutional_holders,
        test_get_macro_indicators_with_datareader,
        test_get_macro_indicators_no_datareader,
        test_get_all_alternative_data,
        test_transform_interface,
        test_get_set_params,
        test_selective_data_types,
        test_convenience_function_load_news,
        test_convenience_function_load_insider_transactions,
        test_fred_series_catalog,
        # Temporal filtering
        test_news_temporal_filtering_start_date,
        test_news_temporal_filtering_end_date,
        test_news_temporal_no_filter_returns_all,
        test_insider_transactions_temporal_filtering,
        test_institutional_holders_temporal_filtering,
        test_temporal_no_dates_returns_all_alternative,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")

    print(f"\n=== TEST RESULTS ===")
    print(f"Passed: {passed}/{len(tests)}")
    if passed == len(tests):
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    return passed == len(tests)


if __name__ == "__main__":
    main()
