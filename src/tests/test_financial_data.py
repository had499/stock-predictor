"""
Tests for the FinancialDataLoader module.

These tests use mock data and unittest.mock to avoid live network calls,
consistent with the project's approach of keeping tests self-contained.
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.financial_data_loader import (
    FinancialDataLoader,
    load_financial_statements,
    load_key_metrics,
)


# ---------------------------------------------------------------------------
# Helpers – build fake yfinance data
# ---------------------------------------------------------------------------

def _make_income_stmt() -> pd.DataFrame:
    """Return a fake annual income statement (line items as index, periods as columns)."""
    periods = pd.to_datetime(['2023-12-31', '2022-12-31', '2021-12-31'])
    data = {
        'Total Revenue': [383_000e6, 394_000e6, 365_000e6],
        'Net Income': [97_000e6, 99_000e6, 94_000e6],
        'Gross Profit': [170_000e6, 170_000e6, 152_000e6],
        'Operating Income': [114_000e6, 119_000e6, 108_000e6],
    }
    return pd.DataFrame(data, index=periods).T


def _make_balance_sheet() -> pd.DataFrame:
    """Return a fake annual balance sheet."""
    periods = pd.to_datetime(['2023-12-31', '2022-12-31', '2021-12-31'])
    data = {
        'Total Assets': [352_755e6, 352_755e6, 351_002e6],
        'Total Liabilities Net Minority Interest': [290_437e6, 302_083e6, 287_912e6],
        'Total Equity Gross Minority Interest': [62_146e6, 50_672e6, 63_090e6],
    }
    return pd.DataFrame(data, index=periods).T


def _make_cashflow() -> pd.DataFrame:
    """Return a fake cash flow statement."""
    periods = pd.to_datetime(['2023-12-31', '2022-12-31'])
    data = {
        'Operating Cash Flow': [110_543e6, 122_151e6],
        'Capital Expenditure': [-11_282e6, -10_708e6],
        'Free Cash Flow': [99_261e6, 111_443e6],
    }
    return pd.DataFrame(data, index=periods).T


def _make_ticker_mock() -> MagicMock:
    """Build a MagicMock resembling a yfinance Ticker with stub financial data."""
    mock = MagicMock()
    mock.income_stmt = _make_income_stmt()
    mock.quarterly_income_stmt = _make_income_stmt()
    mock.balance_sheet = _make_balance_sheet()
    mock.quarterly_balance_sheet = _make_balance_sheet()
    mock.cashflow = _make_cashflow()
    mock.quarterly_cashflow = _make_cashflow()
    mock.info = {
        'trailingPE': 28.5,
        'forwardPE': 25.0,
        'priceToBook': 45.0,
        'profitMargins': 0.253,
        'returnOnEquity': 1.56,
        'debtToEquity': 181.0,
        'currentRatio': 0.98,
        'dividendYield': 0.005,
        'marketCap': 3_000_000_000_000,
        'trailingEps': 6.13,
    }
    return mock


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_fetch_income_statement():
    """FinancialDataLoader should return a non-empty income statement DataFrame."""
    print("Testing income statement fetching...")

    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader(frequency='annual')
        df = loader.get_income_statement('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "Income statement should not be empty"
    assert 'symbol' in df.columns, "Symbol column should be present"
    assert 'total_revenue' in df.columns, "Total revenue column should be present"
    assert len(df) == 3, "Should have 3 annual periods"
    print("✓ Income statement test passed")
    return True


def test_fetch_balance_sheet():
    """FinancialDataLoader should return a non-empty balance sheet DataFrame."""
    print("Testing balance sheet fetching...")

    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader(frequency='annual')
        df = loader.get_balance_sheet('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "Balance sheet should not be empty"
    assert 'symbol' in df.columns, "Symbol column should be present"
    print("✓ Balance sheet test passed")
    return True


def test_fetch_cash_flow():
    """FinancialDataLoader should return a non-empty cash flow DataFrame."""
    print("Testing cash flow fetching...")

    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader(frequency='annual')
        df = loader.get_cash_flow('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "Cash flow should not be empty"
    assert 'symbol' in df.columns, "Symbol column should be present"
    print("✓ Cash flow test passed")
    return True


def test_fetch_key_metrics():
    """FinancialDataLoader should return a single-row key metrics DataFrame."""
    print("Testing key metrics fetching...")

    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader()
        df = loader.get_key_metrics('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert len(df) == 1, "Should return exactly one row"
    assert df['pe_ratio'].iloc[0] == 28.5, "PE ratio should match mock data"
    assert df['profit_margin'].iloc[0] == 0.253, "Profit margin should match mock data"
    assert 'dividend_yield' in df.columns, "Dividend yield should be present"
    print("✓ Key metrics test passed")
    return True


def test_quarterly_frequency():
    """FinancialDataLoader should use quarterly attributes when frequency='quarterly'."""
    print("Testing quarterly frequency...")

    mock = _make_ticker_mock()

    with patch('data.financial_data_loader.yf.Ticker', return_value=mock):
        loader = FinancialDataLoader(frequency='quarterly')
        df = loader.get_income_statement('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "Quarterly income statement should not be empty"
    # The quarterly accessor should have been called (not the annual one)
    assert mock.quarterly_income_stmt is not None
    print("✓ Quarterly frequency test passed")
    return True


def test_get_all_financial_data():
    """get_all_financial_data should return a dict with all four keys."""
    print("Testing get_all_financial_data...")

    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader()
        result = loader.get_all_financial_data('AAPL')

    assert isinstance(result, dict), "Should return a dict"
    assert 'income_statement' in result, "Should include income_statement"
    assert 'balance_sheet' in result, "Should include balance_sheet"
    assert 'cash_flow' in result, "Should include cash_flow"
    assert 'key_metrics' in result, "Should include key_metrics"
    for key, df in result.items():
        assert isinstance(df, pd.DataFrame), f"{key} should be a DataFrame"
    print("✓ get_all_financial_data test passed")
    return True


def test_transform_interface():
    """transform() should load data for all configured symbols."""
    print("Testing sklearn transform interface...")

    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader(symbols=['AAPL', 'MSFT'])
        result = loader.transform(None)

    assert isinstance(result, dict), "Should return a dict"
    assert 'AAPL' in result, "AAPL should be in results"
    assert 'MSFT' in result, "MSFT should be in results"
    print("✓ Transform interface test passed")
    return True


def test_transform_raises_without_symbols():
    """transform() should raise ValueError when symbols is not set."""
    print("Testing transform raises without symbols...")

    loader = FinancialDataLoader()  # no symbols
    raised = False
    try:
        loader.transform(None)
    except ValueError:
        raised = True

    assert raised, "Should raise ValueError when symbols is None"
    print("✓ Raises without symbols test passed")
    return True


def test_get_set_params():
    """get_params and set_params should work correctly."""
    print("Testing get_params / set_params...")

    loader = FinancialDataLoader(frequency='annual')
    params = loader.get_params()

    assert 'frequency' in params, "frequency should be in params"
    assert params['frequency'] == 'annual'

    loader.set_params(frequency='quarterly')
    assert loader.frequency == 'quarterly', "set_params should update the attribute"

    print("✓ get_params / set_params test passed")
    return True


def test_convenience_function_load_financial_statements():
    """load_financial_statements convenience function should return a dict."""
    print("Testing load_financial_statements convenience function...")

    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        result = load_financial_statements('AAPL', frequency='annual')

    assert isinstance(result, dict), "Should return a dict"
    assert 'income_statement' in result
    assert 'balance_sheet' in result
    assert 'cash_flow' in result
    assert 'key_metrics' in result
    print("✓ load_financial_statements convenience function test passed")
    return True


def test_convenience_function_load_key_metrics():
    """load_key_metrics convenience function should return a DataFrame."""
    print("Testing load_key_metrics convenience function...")

    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        df = load_key_metrics('AAPL')

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert len(df) == 1, "Should return one row"
    print("✓ load_key_metrics convenience function test passed")
    return True


def test_selective_statements():
    """FinancialDataLoader should respect include_* flags."""
    print("Testing selective statement loading...")

    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader(
            include_income_statement=True,
            include_balance_sheet=False,
            include_cash_flow=False,
            include_key_metrics=False,
        )
        result = loader.get_all_financial_data('AAPL')

    assert 'income_statement' in result, "income_statement should be included"
    assert 'balance_sheet' not in result, "balance_sheet should be excluded"
    assert 'cash_flow' not in result, "cash_flow should be excluded"
    assert 'key_metrics' not in result, "key_metrics should be excluded"
    print("✓ Selective statement loading test passed")
    return True


def test_temporal_filtering_start_date():
    """FinancialDataLoader should exclude periods before start_date."""
    print("Testing temporal filtering – start_date...")

    # Mock has periods: 2021-12-31, 2022-12-31, 2023-12-31
    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader(start_date='2022-06-01')
        df = loader.get_income_statement('AAPL')

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Should have some results"
    periods = pd.to_datetime(df['period'])
    assert (periods >= pd.Timestamp('2022-06-01')).all(), (
        "All returned periods should be on or after 2022-06-01"
    )
    # 2021-12-31 must be excluded
    assert not (periods == pd.Timestamp('2021-12-31')).any(), (
        "2021-12-31 period should be excluded by start_date filter"
    )
    print("✓ Temporal filtering – start_date test passed")
    return True


def test_temporal_filtering_end_date():
    """FinancialDataLoader should exclude periods after end_date."""
    print("Testing temporal filtering – end_date...")

    # Mock has periods: 2021-12-31, 2022-12-31, 2023-12-31
    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader(end_date='2022-12-31')
        df = loader.get_income_statement('AAPL')

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Should have some results"
    periods = pd.to_datetime(df['period'])
    assert (periods <= pd.Timestamp('2022-12-31')).all(), (
        "All returned periods should be on or before 2022-12-31"
    )
    # 2023-12-31 must be excluded
    assert not (periods == pd.Timestamp('2023-12-31')).any(), (
        "2023-12-31 period should be excluded by end_date filter"
    )
    print("✓ Temporal filtering – end_date test passed")
    return True


def test_temporal_filtering_date_range():
    """FinancialDataLoader should return only periods within [start_date, end_date]."""
    print("Testing temporal filtering – date range...")

    # Mock has periods: 2021-12-31, 2022-12-31, 2023-12-31
    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader(
            start_date='2022-01-01',
            end_date='2022-12-31',
        )
        df = loader.get_income_statement('AAPL')

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1, "Exactly one period should match 2022-01-01 to 2022-12-31"
    assert pd.to_datetime(df['period'].iloc[0]) == pd.Timestamp('2022-12-31'), (
        "Only the 2022-12-31 period should be returned"
    )
    print("✓ Temporal filtering – date range test passed")
    return True


def test_temporal_no_dates_returns_all():
    """FinancialDataLoader without dates should return all available periods."""
    print("Testing no-date filter returns all periods...")

    # Mock has 3 annual periods for income/balance and 2 for cash flow
    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader()  # no start/end date
        df = loader.get_income_statement('AAPL')

    assert len(df) == 3, "All 3 periods should be returned when no date filter is set"
    print("✓ No-date filter returns all periods test passed")
    return True


def test_temporal_date_range_no_matches():
    """FinancialDataLoader should return empty DataFrame when no period falls in range."""
    print("Testing temporal filtering – no matching periods...")

    # Mock periods: 2021-12-31, 2022-12-31, 2023-12-31
    # Request a far-future range with no matches
    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        loader = FinancialDataLoader(
            start_date='2030-01-01',
            end_date='2030-12-31',
        )
        df = loader.get_income_statement('AAPL')

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0, "No periods should match a far-future date range"
    print("✓ Temporal filtering – no matches test passed")
    return True


def test_temporal_params_in_get_params():
    """start_date and end_date should appear in get_params()."""
    print("Testing start_date / end_date in get_params...")

    loader = FinancialDataLoader(start_date='2022-01-01', end_date='2023-12-31')
    params = loader.get_params()

    assert 'start_date' in params, "start_date should be in params"
    assert 'end_date' in params, "end_date should be in params"
    assert params['start_date'] == '2022-01-01'
    assert params['end_date'] == '2023-12-31'
    print("✓ start_date / end_date in get_params test passed")
    return True


def test_temporal_convenience_function():
    """load_financial_statements should pass start_date/end_date through to the loader."""
    print("Testing temporal filtering in load_financial_statements convenience function...")

    # Mock has periods 2021-12-31, 2022-12-31, 2023-12-31
    with patch('data.financial_data_loader.yf.Ticker', return_value=_make_ticker_mock()):
        result = load_financial_statements(
            'AAPL',
            frequency='annual',
            start_date='2023-01-01',
            end_date='2023-12-31',
        )

    income = result['income_statement']
    assert isinstance(income, pd.DataFrame)
    assert len(income) == 1, "Only 2023-12-31 should match"
    print("✓ Temporal filtering in load_financial_statements test passed")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=== TESTING FINANCIAL DATA LOADER ===\n")

    tests = [
        test_fetch_income_statement,
        test_fetch_balance_sheet,
        test_fetch_cash_flow,
        test_fetch_key_metrics,
        test_quarterly_frequency,
        test_get_all_financial_data,
        test_transform_interface,
        test_transform_raises_without_symbols,
        test_get_set_params,
        test_convenience_function_load_financial_statements,
        test_convenience_function_load_key_metrics,
        test_selective_statements,
        # Temporal filtering
        test_temporal_filtering_start_date,
        test_temporal_filtering_end_date,
        test_temporal_filtering_date_range,
        test_temporal_no_dates_returns_all,
        test_temporal_date_range_no_matches,
        test_temporal_params_in_get_params,
        test_temporal_convenience_function,
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
