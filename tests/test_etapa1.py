"""
Tests for Etapa 1 — Schema Adaptation (mocked DB, no live connections needed).

Covers:
  - _flatten_market(): resolution fields are present and correctly mapped
  - insert_price_history(): bid/ask propagated to INSERT records
  - update_market_resolution(): correct SQL executed with right parameters
"""

import json
import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy C-extensions so tests run without a live DB install
# ---------------------------------------------------------------------------
def _stub_module(name: str) -> ModuleType:
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod

if "psycopg2" not in sys.modules:
    pg = _stub_module("psycopg2")
    pg.connect = MagicMock()
    pg.OperationalError = Exception
    extras = _stub_module("psycopg2.extras")
    extras.Json = lambda x: x          # return the dict as-is for tests
    extras.execute_values = MagicMock()

if "dotenv" not in sys.modules:
    dotenv = _stub_module("dotenv")
    dotenv.load_dotenv = MagicMock()

# Allow imports from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from db import _flatten_market, insert_price_history, update_market_resolution


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_market(**overrides) -> dict:
    """Returns a minimal market dict as returned by the Gamma API."""
    base = {
        "id": "market-001",
        "conditionId": "0xABC",
        "questionID": "0xDEF",
        "question": "Will X happen?",
        "description": "Test market",
        "createdAt": "2025-01-01T00:00:00Z",
        "startDate": "2025-01-01T00:00:00Z",
        "endDate": "2025-06-01T00:00:00Z",
        "acceptingOrdersTimestamp": "2025-01-01T00:00:00Z",
        "competitive": 0.9,
        "active": False,
        "closed": True,
        "archived": False,
        "acceptingOrders": False,
        "umaBond": "1500000000000000000",
        "orderMinSize": "5",
        "orderPriceMinTickSize": "0.01",
        "closedTime": "2025-06-01T20:56:36+00:00",
        "umaResolutionStatus": "resolved",
        "outcomePrices": '["1", "0"]',
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["TOKEN_YES", "TOKEN_NO"]',
    }
    base.update(overrides)
    return base


def _make_conn_mock() -> MagicMock:
    """Returns a mock psycopg2 connection with a cursor context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


# ---------------------------------------------------------------------------
# _flatten_market tests
# ---------------------------------------------------------------------------

class TestFlattenMarket:
    def test_resolution_fields_present(self):
        market = _make_market()
        flat = _flatten_market(market, event_id="event-001")

        assert "resolved_at" in flat
        assert "winner_token_id" in flat
        assert "uma_resolution_status" in flat

    def test_resolved_at_maps_closed_time(self):
        market = _make_market(closedTime="2025-06-01T20:56:36+00:00")
        flat = _flatten_market(market, event_id="event-001")

        assert flat["resolved_at"] == "2025-06-01T20:56:36+00:00"

    def test_winner_token_id_is_none_at_flatten_time(self):
        """winner_token_id is always None from _flatten_market; set later via update_market_resolution."""
        flat = _flatten_market(_make_market(), event_id="event-001")
        assert flat["winner_token_id"] is None

    def test_uma_resolution_status_mapped(self):
        market = _make_market(umaResolutionStatus="resolved")
        flat = _flatten_market(market, event_id="event-001")
        assert flat["uma_resolution_status"] == "resolved"

    def test_missing_resolution_fields_default_none(self):
        """Markets without closedTime/umaResolutionStatus (active markets) should not crash."""
        market = _make_market()
        del market["closedTime"]
        del market["umaResolutionStatus"]
        flat = _flatten_market(market, event_id="event-001")

        assert flat["resolved_at"] is None
        assert flat["uma_resolution_status"] is None

    def test_legacy_fields_still_present(self):
        """Ensure existing fields were not accidentally removed."""
        flat = _flatten_market(_make_market(), event_id="event-001")
        for field in ("id", "event_id", "condition_id", "question", "active", "closed"):
            assert field in flat, f"Legacy field '{field}' missing from _flatten_market output"


# ---------------------------------------------------------------------------
# insert_price_history tests
# ---------------------------------------------------------------------------

class TestInsertPriceHistory:
    def _price_data(self):
        return [
            {"t": 1700000000, "p": "0.85"},
            {"t": 1700003600, "p": "0.87"},
        ]

    def test_mid_only_inserts_null_bid_ask(self):
        conn, cursor = _make_conn_mock()
        insert_price_history(conn, "TOKEN_YES", self._price_data())

        # execute_values is called once
        assert cursor.called or conn.cursor.called

        # Inspect the records that would be inserted via execute_values
        # We need to patch execute_values to capture args
        from unittest.mock import patch as _patch
        import db as db_module

        with _patch("db.execute_values") as mock_ev:
            conn2, _ = _make_conn_mock()
            insert_price_history(conn2, "TOKEN_YES", self._price_data())
            args, kwargs = mock_ev.call_args
            records = args[2]  # third positional arg to execute_values

        assert len(records) == 2
        # bid and ask are the 6th and 7th elements (index 5, 6)
        for rec in records:
            assert rec[5] is None, "bid should be None when not provided"
            assert rec[6] is None, "ask should be None when not provided"

    def test_bid_ask_propagated_to_records(self):
        import db as db_module
        from unittest.mock import patch as _patch

        with _patch("db.execute_values") as mock_ev:
            conn, _ = _make_conn_mock()
            insert_price_history(conn, "TOKEN_YES", self._price_data(), bid=0.84, ask=0.86)
            args, _ = mock_ev.call_args
            records = args[2]

        assert len(records) == 2
        for rec in records:
            assert rec[5] == 0.84, "bid should be 0.84"
            assert rec[6] == 0.86, "ask should be 0.86"

    def test_empty_price_data_does_nothing(self):
        conn, cursor = _make_conn_mock()
        insert_price_history(conn, "TOKEN_YES", [])
        conn.commit.assert_not_called()

    def test_side_column_is_mid(self):
        import db as db_module
        from unittest.mock import patch as _patch

        with _patch("db.execute_values") as mock_ev:
            conn, _ = _make_conn_mock()
            insert_price_history(conn, "TOKEN_YES", self._price_data())
            args, _ = mock_ev.call_args
            records = args[2]

        for rec in records:
            assert rec[4] == "MID", "side should always be 'MID'"

    def test_sql_includes_bid_ask_columns(self):
        """The INSERT SQL must reference the bid and ask columns."""
        import db as db_module
        from unittest.mock import patch as _patch

        with _patch("db.execute_values") as mock_ev:
            conn, _ = _make_conn_mock()
            insert_price_history(conn, "TOKEN_YES", self._price_data())
            args, _ = mock_ev.call_args
            sql: str = args[1]

        assert "bid" in sql
        assert "ask" in sql


# ---------------------------------------------------------------------------
# update_market_resolution tests
# ---------------------------------------------------------------------------

class TestUpdateMarketResolution:
    def test_executes_update_with_correct_params(self):
        conn, cursor = _make_conn_mock()
        update_market_resolution(
            conn,
            market_id="market-001",
            winner_token_id="TOKEN_YES",
            resolved_at="2025-06-01T20:56:36+00:00",
            uma_resolution_status="resolved",
        )

        cursor.execute.assert_called_once()
        # call_args → (positional_args_tuple, kwargs_dict)
        # cursor.execute(sql, params_tuple) → positional_args = (sql, params_tuple)
        pos_args, _ = cursor.execute.call_args
        params = pos_args[1]  # the params tuple passed to execute
        assert params[0] == "TOKEN_YES"
        assert params[1] == "2025-06-01T20:56:36+00:00"
        assert params[2] == "resolved"
        assert params[3] == "market-001"

    def test_commits_after_update(self):
        conn, _ = _make_conn_mock()
        update_market_resolution(conn, "market-001", "TOKEN_YES", "2025-06-01", "resolved")
        conn.commit.assert_called_once()

    def test_allows_null_winner_token_id(self):
        """Ambiguous/N/A resolutions must set winner_token_id to None without error."""
        conn, cursor = _make_conn_mock()
        update_market_resolution(
            conn,
            market_id="market-002",
            winner_token_id=None,
            resolved_at="2025-06-01T20:56:36+00:00",
            uma_resolution_status="N/A",
        )
        pos_args, _ = cursor.execute.call_args
        params = pos_args[1]
        assert params[0] is None

    def test_sql_updates_all_three_fields(self):
        conn, cursor = _make_conn_mock()
        update_market_resolution(conn, "market-001", "TOKEN_YES", "2025-06-01", "resolved")
        sql, _ = cursor.execute.call_args
        sql_str: str = sql[0]

        for col in ("winner_token_id", "resolved_at", "uma_resolution_status"):
            assert col in sql_str, f"UPDATE SQL must set '{col}'"
