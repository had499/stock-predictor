"""
Tests for src/data/alternative/scrapers/insider.py (scrape_insider_trades).

Strategy: mock the `edgar` library at the point it is imported inside the
function so we never make real HTTP calls.  The _cache helpers are also
patched so tests are hermetic and don't touch disk.

Real edgartools API (Form4 object):
  - ownership.to_dataframe() → pd.DataFrame
      columns: Code, Date, Shares, Value, Position, ...
"""

import sys
import types
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# All output columns (in order)
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = {
    "date", "ticker",
    "insider_buys", "insider_sells", "insider_net_transactions",
    "insider_buy_value", "insider_sell_value", "insider_net_value",
    "insider_transaction_count", "insider_buy_ratio",
    "officer_buys", "officer_sells",
    "grant_count", "grant_shares",
    "tax_withheld_count", "tax_withheld_shares",
    "option_exercises", "gift_count",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_txn_df(rows):
    """
    Build a DataFrame matching edgartools Form4.to_dataframe() schema.
    Each row: Code, Date, Shares, Value, Position
    """
    if not rows:
        return pd.DataFrame(columns=["Code", "Date", "Shares", "Value", "Position"])
    return pd.DataFrame(rows)


def _make_ownership(rows):
    o = MagicMock()
    o.to_dataframe.return_value = _make_txn_df(rows)
    return o


def _make_filing(ownership_obj, filing_date="2023-06-15"):
    f = MagicMock()
    f.filing_date = filing_date
    f.obj.return_value = ownership_obj
    return f


def _patch_edgar(filings):
    fake_edgar = types.ModuleType("edgar")
    mock_company_cls = MagicMock()
    mock_company_cls.return_value.get_filings.return_value = filings
    fake_edgar.Company = mock_company_cls
    fake_edgar.set_identity = MagicMock()
    return patch.dict(sys.modules, {"edgar": fake_edgar})


def _run(filings, **kwargs):
    from src.data.alternative.scrapers.insider import scrape_insider_trades
    with _patch_edgar(filings), \
         patch("src.data.alternative.scrapers.insider._load_cache", return_value=None), \
         patch("src.data.alternative.scrapers.insider._save_cache"):
        return scrape_insider_trades(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicAggregation:
    """One buy + one sell on the same day → verify all columns."""

    def setup_method(self):
        rows = [
            {"Code": "P", "Date": "2023-06-01", "Shares": 100.0, "Value": 5000.0,  "Position": "President and CEO"},
            {"Code": "S", "Date": "2023-06-01", "Shares": 200.0, "Value": 12000.0, "Position": "EVP & CFO"},
        ]
        ownership = _make_ownership(rows)
        filing = _make_filing(ownership, filing_date="2023-06-01")
        self.df = _run([filing], ticker="AAPL", start_date="2023-01-01",
                       end_date="2023-12-31", use_cache=False)

    def test_columns_present(self):
        assert EXPECTED_COLUMNS.issubset(set(self.df.columns))

    def test_one_row_per_day(self):
        assert len(self.df) == 1

    def test_ticker_uppercased(self):
        assert self.df["ticker"].iloc[0] == "AAPL"

    def test_buy_sell_counts(self):
        row = self.df.iloc[0]
        assert row["insider_buys"] == 1
        assert row["insider_sells"] == 1
        assert row["insider_transaction_count"] == 2

    def test_net_transactions(self):
        assert self.df["insider_net_transactions"].iloc[0] == 0

    def test_buy_ratio(self):
        assert self.df["insider_buy_ratio"].iloc[0] == pytest.approx(0.5)

    def test_values(self):
        row = self.df.iloc[0]
        assert row["insider_buy_value"] == pytest.approx(5000.0)
        assert row["insider_sell_value"] == pytest.approx(12000.0)
        assert row["insider_net_value"] == pytest.approx(5000.0 - 12000.0)

    def test_officer_flags(self):
        row = self.df.iloc[0]
        assert row["officer_buys"] == 1
        assert row["officer_sells"] == 1


class TestNewTransactionTypes:
    """A, F, M, G codes are captured in new columns."""

    def setup_method(self):
        rows = [
            {"Code": "A", "Date": "2023-06-01", "Shares": 1000.0, "Value": 0.0,   "Position": "Director"},
            {"Code": "A", "Date": "2023-06-01", "Shares": 500.0,  "Value": 0.0,   "Position": "EVP"},
            {"Code": "F", "Date": "2023-06-01", "Shares": 300.0,  "Value": 0.0,   "Position": "EVP"},
            {"Code": "M", "Date": "2023-06-01", "Shares": 200.0,  "Value": 8000.0,"Position": "President and CEO"},
            {"Code": "G", "Date": "2023-06-01", "Shares": 50.0,   "Value": 0.0,   "Position": "Director"},
        ]
        ownership = _make_ownership(rows)
        filing = _make_filing(ownership)
        self.df = _run([filing], ticker="NVDA", start_date="2023-01-01",
                       end_date="2023-12-31", use_cache=False)

    def test_grant_count(self):
        assert self.df["grant_count"].iloc[0] == 2

    def test_grant_shares(self):
        assert self.df["grant_shares"].iloc[0] == pytest.approx(1500.0)

    def test_tax_withheld_count(self):
        assert self.df["tax_withheld_count"].iloc[0] == 1

    def test_tax_withheld_shares(self):
        assert self.df["tax_withheld_shares"].iloc[0] == pytest.approx(300.0)

    def test_option_exercises(self):
        assert self.df["option_exercises"].iloc[0] == 1

    def test_gift_count(self):
        assert self.df["gift_count"].iloc[0] == 1

    def test_grants_not_counted_as_buys(self):
        assert self.df["insider_buys"].iloc[0] == 0

    def test_tax_not_counted_as_sells(self):
        assert self.df["insider_sells"].iloc[0] == 0


class TestOfficerDetection:
    """Officer detection uses Position keywords, not just 'officer' string."""

    def _run_single(self, position):
        rows = [{"Code": "S", "Date": "2023-05-01", "Shares": 10.0,
                 "Value": 100.0, "Position": position}]
        ownership = _make_ownership(rows)
        filing = _make_filing(ownership, filing_date="2023-05-01")
        df = _run([filing], ticker="X", start_date="2023-01-01",
                  end_date="2023-12-31", use_cache=False)
        return df["officer_sells"].iloc[0]

    def test_ceo_is_officer(self):
        assert self._run_single("President and CEO") == 1

    def test_evp_is_officer(self):
        assert self._run_single("EVP & Chief Financial Officer") == 1

    def test_pure_director_is_not_officer(self):
        assert self._run_single("Director") == 0

    def test_vp_is_officer(self):
        assert self._run_single("VP of Engineering") == 1


class TestEmptyResults:

    def test_empty_filings(self):
        df = _run([], ticker="MSFT", start_date="2023-01-01",
                  end_date="2023-12-31", use_cache=False)
        assert df.empty
        assert EXPECTED_COLUMNS.issubset(set(df.columns))

    def test_filing_with_no_transactions(self):
        ownership = _make_ownership([])
        filing = _make_filing(ownership)
        df = _run([filing], ticker="TSLA", start_date="2023-01-01",
                  end_date="2023-12-31", use_cache=False)
        assert df.empty


class TestDateFiltering:

    def test_out_of_range_excluded(self):
        rows = [
            {"Code": "P", "Date": "2022-12-31", "Shares": 100.0, "Value": 1000.0, "Position": "CEO"},
            {"Code": "P", "Date": "2023-03-01", "Shares": 200.0, "Value": 2000.0, "Position": "CEO"},
        ]
        ownership = _make_ownership(rows)
        filing = _make_filing(ownership, filing_date="2023-03-01")
        df = _run([filing], ticker="GOOG", start_date="2023-01-01",
                  end_date="2023-12-31", use_cache=False)
        assert len(df) == 1
        assert df["date"].iloc[0] == "2023-03-01"

    def test_none_date_falls_back_to_filing_date(self):
        rows = [{"Code": "P", "Date": None, "Shares": 50.0, "Value": 500.0, "Position": "CFO"}]
        ownership = _make_ownership(rows)
        filing = _make_filing(ownership, filing_date="2023-04-10")
        df = _run([filing], ticker="META", start_date="2023-01-01",
                  end_date="2023-12-31", use_cache=False)
        assert df["date"].iloc[0] == "2023-04-10"


class TestMaxFilings:

    def test_cap_respected(self):
        def _make(date):
            rows = [{"Code": "S", "Date": date, "Shares": 10.0, "Value": 50.0, "Position": "CFO"}]
            return _make_filing(_make_ownership(rows), filing_date=date)

        filings = [_make(f"2023-0{i}-01") for i in range(1, 6)]
        df = _run(filings, ticker="NFLX", start_date="2023-01-01",
                  end_date="2023-12-31", use_cache=False, max_filings=3)
        assert len(df) == 3


class TestCacheBehavior:

    def test_cache_hit_skips_edgar(self):
        cached_df = pd.DataFrame([{
            "date": "2023-01-01", "ticker": "AMZN",
            "insider_buys": 0, "insider_sells": 1, "insider_net_transactions": -1,
            "insider_buy_value": 0.0, "insider_sell_value": 1000.0, "insider_net_value": -1000.0,
            "insider_transaction_count": 1, "insider_buy_ratio": 0.0,
            "officer_buys": 0, "officer_sells": 1,
            "grant_count": 0, "grant_shares": 0.0,
            "tax_withheld_count": 0, "tax_withheld_shares": 0.0,
            "option_exercises": 0, "gift_count": 0,
        }])
        fake_edgar = types.ModuleType("edgar")
        mock_company = MagicMock()
        fake_edgar.Company = mock_company
        fake_edgar.set_identity = MagicMock()

        from src.data.alternative.scrapers.insider import scrape_insider_trades
        with patch.dict(sys.modules, {"edgar": fake_edgar}), \
             patch("src.data.alternative.scrapers.insider._load_cache", return_value=cached_df) as mock_load, \
             patch("src.data.alternative.scrapers.insider._save_cache") as mock_save:
            result = scrape_insider_trades("AMZN", "2023-01-01", "2023-12-31", use_cache=True)

        mock_load.assert_called_once()
        mock_company.assert_not_called()
        mock_save.assert_not_called()
        pd.testing.assert_frame_equal(result, cached_df)

    def test_cache_miss_saves_result(self):
        rows = [{"Code": "S", "Date": "2023-05-01", "Shares": 10.0, "Value": 50.0, "Position": "EVP"}]
        ownership = _make_ownership(rows)
        filing = _make_filing(ownership, filing_date="2023-05-01")

        from src.data.alternative.scrapers.insider import scrape_insider_trades
        with _patch_edgar([filing]), \
             patch("src.data.alternative.scrapers.insider._load_cache", return_value=None), \
             patch("src.data.alternative.scrapers.insider._save_cache") as mock_save:
            scrape_insider_trades("AAPL", "2023-01-01", "2023-12-31", use_cache=True)

        mock_save.assert_called_once()
        assert not mock_save.call_args[0][0].empty


class TestEdgartoolsImportError:

    def test_import_error_propagated(self):
        from src.data.alternative.scrapers.insider import scrape_insider_trades
        with patch.dict(sys.modules, {"edgar": None}):
            with pytest.raises(ImportError, match="edgartools"):
                scrape_insider_trades("AAPL", use_cache=False)


class TestEdgarFetchFailure:

    def test_fetch_exception_returns_empty(self):
        fake_edgar = types.ModuleType("edgar")
        mock_company = MagicMock()
        mock_company.return_value.get_filings.side_effect = RuntimeError("network error")
        fake_edgar.Company = mock_company
        fake_edgar.set_identity = MagicMock()

        from src.data.alternative.scrapers.insider import scrape_insider_trades
        with patch.dict(sys.modules, {"edgar": fake_edgar}), \
             patch("src.data.alternative.scrapers.insider._load_cache", return_value=None), \
             patch("src.data.alternative.scrapers.insider._save_cache"):
            df = scrape_insider_trades("AAPL", "2023-01-01", "2023-12-31", use_cache=False)

        assert df.empty


class TestMultipleDays:

    def test_different_codes_on_different_days(self):
        rows = [
            {"Code": "P", "Date": "2023-01-10", "Shares": 10.0,   "Value": 500.0,  "Position": "CEO"},
            {"Code": "S", "Date": "2023-02-20", "Shares": 20.0,   "Value": 800.0,  "Position": "CFO"},
            {"Code": "A", "Date": "2023-03-15", "Shares": 1000.0, "Value": 0.0,    "Position": "Director"},
            {"Code": "F", "Date": "2023-03-15", "Shares": 250.0,  "Value": 0.0,    "Position": "Director"},
        ]
        ownership = _make_ownership(rows)
        filing = _make_filing(ownership, filing_date="2023-03-15")
        df = _run([filing], ticker="NVDA", start_date="2023-01-01",
                  end_date="2023-12-31", use_cache=False)

        assert len(df) == 3
        mar = df[df["date"] == "2023-03-15"].iloc[0]
        assert mar["grant_count"] == 1
        assert mar["grant_shares"] == pytest.approx(1000.0)
        assert mar["tax_withheld_count"] == 1
        assert mar["insider_buys"] == 0
        assert mar["insider_sells"] == 0
