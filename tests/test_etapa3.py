"""
Tests for Etapa 3 — Bid/Ask Capture.

All tests are fully mocked — no live DB, Redis, or HTTP connections required.

Covers:
  - get_bid_ask()            : URL construction, float conversion, error handling
  - get_active_token_ids()   : SQL filtering, return format
  - insert_price_history()   : side='SNAPSHOT' parameter propagation
  - _process_batch()         : mid-price calc, None skip, per-token fault isolation
  - run()                    : connection lifecycle, empty-token skip, finally block
"""

import sys
import os
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Stub C-extensions and network libraries before any project import
# ---------------------------------------------------------------------------


def _stub_module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


if "psycopg2" not in sys.modules:
    _stub_module("psycopg2", connect=MagicMock(), OperationalError=Exception)
    _stub_module("psycopg2.extras", Json=lambda x: x, execute_values=MagicMock())

if "dotenv" not in sys.modules:
    _stub_module("dotenv", load_dotenv=MagicMock())

# Stub py_clob_client so the module-level `client = ClobClient(CLOB_BASE_URL)` works
if "py_clob_client" not in sys.modules:
    _stub_module("py_clob_client")
    _mock_client_instance = MagicMock()
    _mock_client_instance.host = "https://clob.polymarket.com/"
    _stub_module("py_clob_client.client", ClobClient=MagicMock(return_value=_mock_client_instance))
    _stub_module("py_clob_client.http_helpers")
    _stub_module("py_clob_client.http_helpers.helpers", get=MagicMock(), post=MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clob_api import get_bid_ask  # noqa: E402
from db import get_active_token_ids, insert_price_history  # noqa: E402
from orderbook_snapshot_worker import _process_batch, run  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


_NOW = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# get_bid_ask
# ---------------------------------------------------------------------------


class TestGetBidAsk:
    def test_returns_bid_and_ask_as_floats(self):
        with patch("clob_api.get") as mock_get:
            mock_get.side_effect = [{"price": "0.491"}, {"price": "0.509"}]
            bid, ask = get_bid_ask("TOKEN_X")

        assert bid == pytest.approx(0.491)
        assert ask == pytest.approx(0.509)

    def test_sell_side_used_for_bid(self):
        """First GET request must use side=SELL."""
        with patch("clob_api.get") as mock_get:
            mock_get.side_effect = [{"price": "0.49"}, {"price": "0.51"}]
            get_bid_ask("MY_TOKEN")

        first_url: str = mock_get.call_args_list[0][0][0]
        assert "side=SELL" in first_url
        assert "MY_TOKEN" in first_url

    def test_buy_side_used_for_ask(self):
        """Second GET request must use side=BUY."""
        with patch("clob_api.get") as mock_get:
            mock_get.side_effect = [{"price": "0.49"}, {"price": "0.51"}]
            get_bid_ask("MY_TOKEN")

        second_url: str = mock_get.call_args_list[1][0][0]
        assert "side=BUY" in second_url
        assert "MY_TOKEN" in second_url

    def test_price_endpoint_in_url(self):
        with patch("clob_api.get") as mock_get:
            mock_get.side_effect = [{"price": "0.49"}, {"price": "0.51"}]
            get_bid_ask("TOKEN_X")

        for c in mock_get.call_args_list:
            assert "/price" in c[0][0]

    def test_returns_none_none_on_exception(self):
        with patch("clob_api.get", side_effect=Exception("timeout")):
            bid, ask = get_bid_ask("TOKEN_X")

        assert bid is None
        assert ask is None

    def test_returns_none_none_when_bid_price_key_missing(self):
        with patch("clob_api.get") as mock_get:
            mock_get.side_effect = [{"error": "no liquidity"}, {"price": "0.51"}]
            bid, ask = get_bid_ask("TOKEN_X")

        assert bid is None
        assert ask is None

    def test_returns_none_none_when_ask_price_key_missing(self):
        with patch("clob_api.get") as mock_get:
            mock_get.side_effect = [{"price": "0.49"}, {}]
            bid, ask = get_bid_ask("TOKEN_X")

        assert bid is None
        assert ask is None

    def test_returns_none_none_when_response_is_not_dict(self):
        with patch("clob_api.get", return_value=None):
            bid, ask = get_bid_ask("TOKEN_X")

        assert bid is None
        assert ask is None

    def test_bid_less_than_ask(self):
        """Sanity: with valid prices, bid should be <= ask."""
        with patch("clob_api.get") as mock_get:
            mock_get.side_effect = [{"price": "0.49"}, {"price": "0.51"}]
            bid, ask = get_bid_ask("TOKEN_X")

        assert bid <= ask


# ---------------------------------------------------------------------------
# get_active_token_ids
# ---------------------------------------------------------------------------


class TestGetActiveTokenIds:
    def test_returns_list_of_token_ids(self):
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = [("TOKEN_A",), ("TOKEN_B",), ("TOKEN_C",)]
        result = get_active_token_ids(conn)
        assert result == ["TOKEN_A", "TOKEN_B", "TOKEN_C"]

    def test_returns_empty_list_when_no_active_markets(self):
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = []
        assert get_active_token_ids(conn) == []

    def test_sql_filters_active_and_not_closed(self):
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = []
        get_active_token_ids(conn)

        sql: str = cursor.execute.call_args[0][0].lower()
        assert "active" in sql
        assert "closed" in sql

    def test_sql_joins_market_outcomes_and_markets(self):
        conn, cursor = _make_conn()
        cursor.fetchall.return_value = []
        get_active_token_ids(conn)

        sql: str = cursor.execute.call_args[0][0].lower()
        assert "market_outcomes" in sql
        assert "markets" in sql


# ---------------------------------------------------------------------------
# insert_price_history — side parameter
# ---------------------------------------------------------------------------


class TestInsertPriceHistoryWithSide:
    _data = [{"t": 1700000000, "p": "0.85"}]

    def test_side_snapshot_stored_in_record(self):
        with patch("db.execute_values") as mock_ev:
            conn, _ = _make_conn()
            insert_price_history(conn, "TOKEN_X", self._data, side="SNAPSHOT")
            records = mock_ev.call_args[0][2]

        assert records[0][4] == "SNAPSHOT"

    def test_side_mid_is_default(self):
        with patch("db.execute_values") as mock_ev:
            conn, _ = _make_conn()
            insert_price_history(conn, "TOKEN_X", self._data)
            records = mock_ev.call_args[0][2]

        assert records[0][4] == "MID"

    def test_snapshot_carries_bid_and_ask(self):
        with patch("db.execute_values") as mock_ev:
            conn, _ = _make_conn()
            insert_price_history(conn, "TOKEN_X", self._data, bid=0.48, ask=0.52, side="SNAPSHOT")
            records = mock_ev.call_args[0][2]

        rec = records[0]
        assert rec[4] == "SNAPSHOT"
        assert rec[5] == pytest.approx(0.48)   # bid
        assert rec[6] == pytest.approx(0.52)   # ask


# ---------------------------------------------------------------------------
# _process_batch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_mid_price_is_average_of_bid_and_ask(self):
        conn, _ = _make_conn()
        with patch("orderbook_snapshot_worker.get_bid_ask", return_value=(0.48, 0.52)), \
             patch("orderbook_snapshot_worker.insert_price_history") as mock_ins:
            _process_batch(conn, ["TOKEN_X"], _NOW)

        price_data = mock_ins.call_args.kwargs["price_data"]
        assert price_data[0]["p"] == pytest.approx(0.50)

    def test_bid_and_ask_forwarded_to_insert(self):
        conn, _ = _make_conn()
        with patch("orderbook_snapshot_worker.get_bid_ask", return_value=(0.48, 0.52)), \
             patch("orderbook_snapshot_worker.insert_price_history") as mock_ins:
            _process_batch(conn, ["TOKEN_X"], _NOW)

        kw = mock_ins.call_args.kwargs
        assert kw["bid"] == pytest.approx(0.48)
        assert kw["ask"] == pytest.approx(0.52)

    def test_side_is_snapshot(self):
        conn, _ = _make_conn()
        with patch("orderbook_snapshot_worker.get_bid_ask", return_value=(0.48, 0.52)), \
             patch("orderbook_snapshot_worker.insert_price_history") as mock_ins:
            _process_batch(conn, ["TOKEN_X"], _NOW)

        assert mock_ins.call_args.kwargs["side"] == "SNAPSHOT"

    def test_skips_token_when_bid_ask_is_none(self):
        conn, _ = _make_conn()
        with patch("orderbook_snapshot_worker.get_bid_ask", return_value=(None, None)), \
             patch("orderbook_snapshot_worker.insert_price_history") as mock_ins:
            count = _process_batch(conn, ["TOKEN_X"], _NOW)

        mock_ins.assert_not_called()
        assert count == 0

    def test_returns_count_of_inserted_tokens(self):
        conn, _ = _make_conn()
        with patch("orderbook_snapshot_worker.get_bid_ask", return_value=(0.48, 0.52)), \
             patch("orderbook_snapshot_worker.insert_price_history"):
            count = _process_batch(conn, ["T1", "T2", "T3"], _NOW)

        assert count == 3

    def test_continues_after_per_token_exception(self):
        """A failure on one token must not abort the remaining tokens in the batch."""
        conn, _ = _make_conn()

        def flaky_bid_ask(token_id: str):
            if token_id == "T2":
                raise RuntimeError("CLOB timeout")
            return 0.48, 0.52

        with patch("orderbook_snapshot_worker.get_bid_ask", side_effect=flaky_bid_ask), \
             patch("orderbook_snapshot_worker.insert_price_history") as mock_ins:
            count = _process_batch(conn, ["T1", "T2", "T3"], _NOW)

        assert count == 2
        assert mock_ins.call_count == 2

    def test_returns_zero_for_empty_batch(self):
        conn, _ = _make_conn()
        with patch("orderbook_snapshot_worker.get_bid_ask"), \
             patch("orderbook_snapshot_worker.insert_price_history") as mock_ins:
            count = _process_batch(conn, [], _NOW)

        mock_ins.assert_not_called()
        assert count == 0

    def test_correct_token_id_passed_to_insert(self):
        conn, _ = _make_conn()
        with patch("orderbook_snapshot_worker.get_bid_ask", return_value=(0.49, 0.51)), \
             patch("orderbook_snapshot_worker.insert_price_history") as mock_ins:
            _process_batch(conn, ["EXPECTED_TOKEN"], _NOW)

        assert mock_ins.call_args.kwargs["token_id"] == "EXPECTED_TOKEN"


# ---------------------------------------------------------------------------
# run() — one full cycle via StopIteration on time.sleep
# ---------------------------------------------------------------------------


class TestRunCycle:
    def _one_cycle(self, token_ids, side_effect_on_active=None):
        """
        Runs the worker loop exactly one cycle.
        time.sleep raises StopIteration on its first call, breaking the while loop.
        Returns (conn_mock, mock_process_batch, mock_time).
        """
        conn = MagicMock()
        mock_pb = MagicMock(return_value=len(token_ids))

        get_active = MagicMock(side_effect=side_effect_on_active) \
            if side_effect_on_active else MagicMock(return_value=token_ids)

        with patch("orderbook_snapshot_worker.get_db_connection", return_value=conn), \
             patch("orderbook_snapshot_worker.get_active_token_ids", get_active), \
             patch("orderbook_snapshot_worker._process_batch", mock_pb), \
             patch("orderbook_snapshot_worker.time") as mock_time:
            mock_time.sleep.side_effect = StopIteration
            mock_time.monotonic.return_value = 0.0

            with pytest.raises(StopIteration):
                run()

        return conn, mock_pb, mock_time

    def test_connection_closed_after_successful_cycle(self):
        conn, _, _ = self._one_cycle(["TOKEN_A"])
        conn.close.assert_called_once()

    def test_connection_closed_when_no_active_tokens(self):
        conn, _, _ = self._one_cycle([])
        conn.close.assert_called_once()

    def test_connection_closed_even_when_get_active_token_ids_raises(self):
        """finally block must run regardless of exceptions inside the cycle."""
        conn, _, _ = self._one_cycle([], side_effect_on_active=RuntimeError("DB down"))
        conn.close.assert_called_once()

    def test_process_batch_not_called_when_no_tokens(self):
        _, mock_pb, _ = self._one_cycle([])
        mock_pb.assert_not_called()

    def test_process_batch_called_with_tokens(self):
        _, mock_pb, _ = self._one_cycle(["T1", "T2"])
        mock_pb.assert_called_once()
        # Verify the batch contains all tokens (within one BATCH_SIZE)
        batch_arg = mock_pb.call_args[0][1]  # second positional arg: token_ids batch
        assert set(batch_arg) == {"T1", "T2"}

    def test_sleep_called_at_end_of_cycle(self):
        _, _, mock_time = self._one_cycle(["TOKEN_A"])
        mock_time.sleep.assert_called()
