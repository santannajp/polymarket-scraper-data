"""
Tests for Etapa 2 — historical_worker.py

All tests use mocked I/O — no live DB, Redis, or HTTP connections required.

Covers:
  - extract_winner_token_id(): winner detection, N/A, edge cases
  - _is_within_lookback(): cutoff date filtering
  - _process_event(): correct DB + Redis calls per event
  - run_full(): pagination, checkpoint read/write, early termination
  - run_incremental(): lookback window, stops on stale page, no checkpoint used
"""

import json
import sys
import os
from datetime import datetime, timedelta, timezone
from types import ModuleType
from unittest.mock import DEFAULT, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Stub C-extensions and network libraries before any project module is imported
# ---------------------------------------------------------------------------


def _stub_module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


if "psycopg2" not in sys.modules:
    pg = _stub_module("psycopg2", connect=MagicMock(), OperationalError=Exception)
    _stub_module("psycopg2.extras", Json=lambda x: x, execute_values=MagicMock())

if "dotenv" not in sys.modules:
    _stub_module("dotenv", load_dotenv=MagicMock())

if "redis" not in sys.modules:
    r_mod = _stub_module("redis", Redis=MagicMock())
    _stub_module("redis.exceptions", ConnectionError=Exception)

if "requests" not in sys.modules:
    req = _stub_module("requests", Session=MagicMock())
    _stub_module(
        "requests.adapters", HTTPAdapter=MagicMock(), Retry=MagicMock()
    )
    _stub_module(
        "requests.exceptions", HTTPError=Exception, RequestException=Exception
    )

# Allow imports from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from historical_worker import (  # noqa: E402
    CHECKPOINT_KEY,
    PRICE_QUEUE,
    _is_within_lookback,
    _process_event,
    extract_winner_token_id,
    run_full,
    run_incremental,
)


# ---------------------------------------------------------------------------
# Test fixtures / factory helpers
# ---------------------------------------------------------------------------


def _make_market(**overrides) -> dict:
    """Minimal closed market dict as returned by the Gamma API."""
    base = {
        "id": "market-001",
        "conditionId": "0xABC",
        "outcomePrices": '["1", "0"]',
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["TOKEN_YES", "TOKEN_NO"]',
        "closedTime": "2026-03-01T12:00:00+00:00",
        "umaResolutionStatus": "resolved",
    }
    base.update(overrides)
    return base


def _make_event(**overrides) -> dict:
    """Minimal closed event dict wrapping one default market."""
    base = {
        "id": "event-001",
        "title": "Test Event",
        "tags": [{"id": 1, "slug": "politics", "label": "Politics"}],
        "markets": [_make_market()],
    }
    base.update(overrides)
    return base


def _make_gamma_client(*pages) -> MagicMock:
    """
    Returns a mock GammaClient whose get_events() yields each page in turn,
    then returns [] to signal end-of-data.
    """
    client = MagicMock()
    client.get_events.side_effect = list(pages) + [[]]
    return client


def _make_redis() -> MagicMock:
    r = MagicMock()
    r.get.return_value = None  # no checkpoint by default
    return r


def _make_conn() -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


def _patch_db_functions():
    """
    Returns a context-manager that patches all DB functions used by
    _process_event.  Using DEFAULT ensures patch.multiple returns a dict of
    auto-created MagicMocks accessible by name (e.g. mocks["upsert_events"]).
    """
    return patch.multiple(
        "historical_worker",
        upsert_events=DEFAULT,
        upsert_tags=DEFAULT,
        link_tags_to_events=DEFAULT,
        upsert_markets=DEFAULT,
        upsert_market_outcomes=DEFAULT,
        update_market_resolution=DEFAULT,
    )


# ---------------------------------------------------------------------------
# extract_winner_token_id
# ---------------------------------------------------------------------------


class TestExtractWinnerTokenId:
    def test_first_outcome_wins(self):
        market = _make_market(
            outcomePrices='["1", "0"]', clobTokenIds='["TOKEN_A", "TOKEN_B"]'
        )
        assert extract_winner_token_id(market) == "TOKEN_A"

    def test_second_outcome_wins(self):
        market = _make_market(
            outcomePrices='["0", "1"]', clobTokenIds='["TOKEN_A", "TOKEN_B"]'
        )
        assert extract_winner_token_id(market) == "TOKEN_B"

    def test_na_resolution_returns_none(self):
        """outcomePrices of 0.5/0.5 indicates an N/A resolution."""
        market = _make_market(
            outcomePrices='["0.5", "0.5"]',
            clobTokenIds='["TOKEN_A", "TOKEN_B"]',
        )
        assert extract_winner_token_id(market) is None

    def test_all_zeros_returns_none(self):
        market = _make_market(
            outcomePrices='["0", "0"]', clobTokenIds='["TOKEN_A", "TOKEN_B"]'
        )
        assert extract_winner_token_id(market) is None

    def test_malformed_outcome_prices_json_returns_none(self):
        market = _make_market(outcomePrices="NOT_JSON", clobTokenIds='["TOKEN_A"]')
        assert extract_winner_token_id(market) is None

    def test_malformed_token_ids_json_returns_none(self):
        market = _make_market(outcomePrices='["1"]', clobTokenIds="NOT_JSON")
        assert extract_winner_token_id(market) is None

    def test_empty_lists_return_none(self):
        market = _make_market(outcomePrices="[]", clobTokenIds="[]")
        assert extract_winner_token_id(market) is None

    def test_missing_keys_return_none(self):
        assert extract_winner_token_id({}) is None

    def test_single_outcome_winner(self):
        market = _make_market(
            outcomePrices='["1"]', clobTokenIds='["TOKEN_ONLY"]'
        )
        assert extract_winner_token_id(market) == "TOKEN_ONLY"

    def test_winner_index_beyond_token_ids_skipped(self):
        """If the winning price index has no matching token_id, return None."""
        market = _make_market(
            outcomePrices='["0", "0", "1"]',  # winner at index 2
            clobTokenIds='["TOKEN_A", "TOKEN_B"]',  # only 2 entries
        )
        assert extract_winner_token_id(market) is None

    def test_mismatched_lengths_uses_valid_indices(self):
        """More prices than token_ids — winner at a valid index is still found."""
        market = _make_market(
            outcomePrices='["0", "1", "0"]',  # winner at index 1
            clobTokenIds='["TOKEN_A", "TOKEN_B"]',
        )
        assert extract_winner_token_id(market) == "TOKEN_B"


# ---------------------------------------------------------------------------
# _is_within_lookback
# ---------------------------------------------------------------------------


class TestIsWithinLookback:
    _now = datetime.now(tz=timezone.utc)

    def test_event_closed_yesterday_is_within_7_day_window(self):
        event = _make_event(
            markets=[
                _make_market(
                    closedTime=(self._now - timedelta(days=1)).isoformat()
                )
            ]
        )
        cutoff = self._now - timedelta(days=7)
        assert _is_within_lookback(event, cutoff) is True

    def test_event_closed_30_days_ago_is_outside_7_day_window(self):
        event = _make_event(
            markets=[
                _make_market(
                    closedTime=(self._now - timedelta(days=30)).isoformat()
                )
            ]
        )
        cutoff = self._now - timedelta(days=7)
        assert _is_within_lookback(event, cutoff) is False

    def test_event_closed_exactly_at_cutoff_is_within(self):
        cutoff = self._now - timedelta(days=7)
        event = _make_event(markets=[_make_market(closedTime=cutoff.isoformat())])
        assert _is_within_lookback(event, cutoff) is True

    def test_event_without_closed_time_is_outside(self):
        market = _make_market()
        del market["closedTime"]
        event = _make_event(markets=[market])
        cutoff = self._now - timedelta(days=7)
        assert _is_within_lookback(event, cutoff) is False

    def test_event_with_no_markets_is_outside(self):
        event = _make_event(markets=[])
        cutoff = self._now - timedelta(days=7)
        assert _is_within_lookback(event, cutoff) is False

    def test_any_recent_market_makes_event_within(self):
        """One recent market among several old ones is enough."""
        old_market = _make_market(
            id="m-old",
            closedTime=(self._now - timedelta(days=30)).isoformat(),
        )
        new_market = _make_market(
            id="m-new",
            closedTime=(self._now - timedelta(days=1)).isoformat(),
        )
        event = _make_event(markets=[old_market, new_market])
        cutoff = self._now - timedelta(days=7)
        assert _is_within_lookback(event, cutoff) is True


# ---------------------------------------------------------------------------
# _process_event
# ---------------------------------------------------------------------------


class TestProcessEvent:
    def test_calls_all_upsert_functions(self):
        event = _make_event()
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions() as mocks:
            _process_event(conn, redis_conn, event)

        mocks["upsert_events"].assert_called_once_with(conn, [event])
        mocks["upsert_tags"].assert_called_once_with(conn, [event])
        mocks["link_tags_to_events"].assert_called_once_with(conn, [event])
        mocks["upsert_markets"].assert_called_once()
        mocks["upsert_market_outcomes"].assert_called_once()

    def test_upsert_markets_receives_event_id(self):
        event = _make_event(id="event-XYZ")
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions() as mocks:
            _process_event(conn, redis_conn, event)

        _, call_args, _ = mocks["upsert_markets"].mock_calls[0]
        assert call_args[2] == "event-XYZ"  # third positional arg is event_id

    def test_calls_update_market_resolution_with_winner(self):
        event = _make_event()  # market has outcomePrices ["1","0"] → TOKEN_YES wins
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions() as mocks:
            _process_event(conn, redis_conn, event)

        mocks["update_market_resolution"].assert_called_once_with(
            conn,
            market_id="market-001",
            winner_token_id="TOKEN_YES",
            resolved_at="2026-03-01T12:00:00+00:00",
            uma_resolution_status="resolved",
        )

    def test_calls_update_market_resolution_with_none_winner_for_na(self):
        market = _make_market(outcomePrices='["0.5", "0.5"]')
        event = _make_event(markets=[market])
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions() as mocks:
            _process_event(conn, redis_conn, event)

        _, _, kwargs = mocks["update_market_resolution"].mock_calls[0]
        assert kwargs["winner_token_id"] is None

    def test_enqueues_each_market_to_redis(self):
        event = _make_event()
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions():
            _process_event(conn, redis_conn, event)

        redis_conn.lpush.assert_called_once_with(PRICE_QUEUE, "market-001")

    def test_enqueues_multiple_markets(self):
        markets = [_make_market(id=f"market-{i}") for i in range(3)]
        event = _make_event(markets=markets)
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions():
            _process_event(conn, redis_conn, event)

        assert redis_conn.lpush.call_count == 3

    def test_returns_enqueued_count(self):
        markets = [_make_market(id=f"m-{i}") for i in range(4)]
        event = _make_event(markets=markets)
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions():
            count = _process_event(conn, redis_conn, event)

        assert count == 4

    def test_skips_event_without_id(self):
        event = _make_event(id=None)
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions() as mocks:
            count = _process_event(conn, redis_conn, event)

        assert count == 0
        mocks["upsert_events"].assert_not_called()
        redis_conn.lpush.assert_not_called()

    def test_skips_markets_without_id(self):
        market_no_id = _make_market(id=None)
        event = _make_event(markets=[market_no_id])
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions() as mocks:
            count = _process_event(conn, redis_conn, event)

        assert count == 0
        mocks["update_market_resolution"].assert_not_called()
        redis_conn.lpush.assert_not_called()

    def test_handles_event_with_no_markets(self):
        event = _make_event(markets=[])
        conn = _make_conn()
        redis_conn = _make_redis()

        with _patch_db_functions() as mocks:
            count = _process_event(conn, redis_conn, event)

        assert count == 0
        mocks["upsert_markets"].assert_not_called()
        mocks["upsert_market_outcomes"].assert_not_called()


# ---------------------------------------------------------------------------
# run_full
# ---------------------------------------------------------------------------


class TestRunFull:
    def test_stops_on_empty_page(self):
        """get_events is called until it returns an empty list."""
        client = _make_gamma_client(
            [_make_event(id="e-1")],
            [_make_event(id="e-2")],
        )
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1), \
                patch("historical_worker.time.sleep"):
            run_full(client, conn, redis_conn)

        # 2 data pages + 1 empty terminator = 3 calls
        assert client.get_events.call_count == 3

    def test_saves_checkpoint_after_each_page(self):
        """Checkpoint is updated with the new offset after every successful page."""
        client = _make_gamma_client([_make_event()])
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1), \
                patch("historical_worker.time.sleep"):
            run_full(client, conn, redis_conn)

        redis_conn.set.assert_called_with(CHECKPOINT_KEY, 100)

    def test_checkpoint_increments_by_page_size_per_page(self):
        """Two pages → checkpoint advances by 2 × PAGE_SIZE."""
        client = _make_gamma_client(
            [_make_event(id="e-1")],
            [_make_event(id="e-2")],
        )
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1), \
                patch("historical_worker.time.sleep"):
            run_full(client, conn, redis_conn)

        assert redis_conn.set.call_count == 2
        calls = redis_conn.set.call_args_list
        assert calls[0] == call(CHECKPOINT_KEY, 100)
        assert calls[1] == call(CHECKPOINT_KEY, 200)

    def test_resumes_from_existing_checkpoint(self):
        """If Redis has a checkpoint, the first API call starts from that offset."""
        redis_conn = _make_redis()
        redis_conn.get.return_value = "500"

        client = _make_gamma_client()  # immediately returns []
        conn = _make_conn()

        with patch("historical_worker._process_event", return_value=0), \
                patch("historical_worker.time.sleep"):
            run_full(client, conn, redis_conn)

        first_call_kwargs = client.get_events.call_args_list[0][1]
        assert first_call_kwargs["offset"] == 500

    def test_starts_from_zero_with_no_checkpoint(self):
        """With no stored checkpoint, iteration begins at offset=0."""
        redis_conn = _make_redis()
        redis_conn.get.return_value = None

        client = _make_gamma_client()
        conn = _make_conn()

        with patch("historical_worker._process_event", return_value=0), \
                patch("historical_worker.time.sleep"):
            run_full(client, conn, redis_conn)

        first_call_kwargs = client.get_events.call_args_list[0][1]
        assert first_call_kwargs["offset"] == 0

    def test_process_event_called_for_every_event(self):
        """Every event in every page must be processed."""
        page1 = [_make_event(id=f"e-{i}") for i in range(3)]
        page2 = [_make_event(id=f"e-{i}") for i in range(3, 5)]
        client = _make_gamma_client(page1, page2)
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1) as mock_pe, \
                patch("historical_worker.time.sleep"):
            run_full(client, conn, redis_conn)

        assert mock_pe.call_count == 5

    def test_requests_closed_not_active_events(self):
        """API must be queried with closed=True, active=False."""
        client = _make_gamma_client()
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=0), \
                patch("historical_worker.time.sleep"):
            run_full(client, conn, redis_conn)

        kwargs = client.get_events.call_args_list[0][1]
        assert kwargs["closed"] is True
        assert kwargs["active"] is False


# ---------------------------------------------------------------------------
# run_incremental
# ---------------------------------------------------------------------------


class TestRunIncremental:
    _now = datetime.now(tz=timezone.utc)

    def _recent_event(self, event_id: str = "event-001") -> dict:
        market = _make_market(
            closedTime=(self._now - timedelta(days=1)).isoformat()
        )
        return _make_event(id=event_id, markets=[market])

    def _old_event(self, event_id: str = "event-old") -> dict:
        market = _make_market(
            id="market-old",
            closedTime=(self._now - timedelta(days=30)).isoformat(),
        )
        return _make_event(id=event_id, markets=[market])

    def test_processes_events_within_lookback_window(self):
        client = _make_gamma_client([self._recent_event()])
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1) as mock_pe, \
                patch("historical_worker.time.sleep"):
            run_incremental(client, conn, redis_conn, lookback_days=7)

        mock_pe.assert_called_once()

    def test_stops_when_entire_page_is_outside_window(self):
        """Once a full page has no recent events, stop — do not fetch another page."""
        client = _make_gamma_client([self._old_event()])
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1) as mock_pe, \
                patch("historical_worker.time.sleep"):
            run_incremental(client, conn, redis_conn, lookback_days=7)

        mock_pe.assert_not_called()
        # Only 1 API call — stopped immediately, did not fetch a second page
        assert client.get_events.call_count == 1

    def test_skips_old_events_within_mixed_page(self):
        """On a mixed page, only recent events are processed."""
        client = _make_gamma_client(
            [self._recent_event("e-new"), self._old_event("e-old")]
        )
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1) as mock_pe, \
                patch("historical_worker.time.sleep"):
            run_incremental(client, conn, redis_conn, lookback_days=7)

        # Only the recent event was passed to _process_event
        assert mock_pe.call_count == 1
        processed_event = mock_pe.call_args[0][2]  # third positional: the event
        assert processed_event["id"] == "e-new"

    def test_continues_paging_while_recent_events_exist(self):
        """As long as pages contain recent events, keep fetching the next page."""
        client = _make_gamma_client(
            [self._recent_event("e-1")],
            [self._recent_event("e-2")],
        )
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1) as mock_pe, \
                patch("historical_worker.time.sleep"):
            run_incremental(client, conn, redis_conn, lookback_days=7)

        assert mock_pe.call_count == 2
        assert client.get_events.call_count == 3  # 2 data + 1 empty

    def test_always_starts_from_offset_zero(self):
        """Incremental mode does not use a checkpoint — offset always starts at 0."""
        client = _make_gamma_client([self._recent_event()])
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1), \
                patch("historical_worker.time.sleep"):
            run_incremental(client, conn, redis_conn, lookback_days=7)

        first_call_kwargs = client.get_events.call_args_list[0][1]
        assert first_call_kwargs["offset"] == 0

    def test_does_not_read_or_write_checkpoint_in_redis(self):
        """Incremental mode must not touch the checkpoint key."""
        client = _make_gamma_client([self._recent_event()])
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1), \
                patch("historical_worker.time.sleep"):
            run_incremental(client, conn, redis_conn, lookback_days=7)

        redis_conn.get.assert_not_called()
        # set() should not be called with the checkpoint key
        for c in redis_conn.set.call_args_list:
            assert c[0][0] != CHECKPOINT_KEY

    def test_handles_empty_first_page(self):
        """If the API returns nothing at all, function exits cleanly."""
        client = _make_gamma_client()  # immediately []
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=0) as mock_pe, \
                patch("historical_worker.time.sleep"):
            run_incremental(client, conn, redis_conn, lookback_days=7)

        mock_pe.assert_not_called()

    def test_requests_closed_not_active_events(self):
        client = _make_gamma_client([self._recent_event()])
        conn = _make_conn()
        redis_conn = _make_redis()

        with patch("historical_worker._process_event", return_value=1), \
                patch("historical_worker.time.sleep"):
            run_incremental(client, conn, redis_conn, lookback_days=7)

        kwargs = client.get_events.call_args_list[0][1]
        assert kwargs["closed"] is True
        assert kwargs["active"] is False
