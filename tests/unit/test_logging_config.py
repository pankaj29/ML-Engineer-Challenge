"""Unit tests for structured logging and correlation IDs.

Part 2 of the brief requires "structured logging with correlation IDs" and
"request/response logging for debugging". The audit found `api/logging_config.py`
at 43% coverage - the lowest module in `api/` - which meant the formatters that
produce every log line in production were largely unexercised.

The correlation ID is the thread that ties a user's complaint to the exact
request in the logs, so these tests care most about it surviving every path.
"""

from __future__ import annotations

import json
import logging

import pytest

from api.logging_config import (
    ConsoleFormatter,
    CorrelationIdFilter,
    JsonFormatter,
    bind_correlation_id,
    configure_logging,
    get_correlation_id,
    get_logger,
    new_correlation_id,
)


def _record(
    msg: str = "something_happened", level: int = logging.INFO, **extra
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="api.test", level=level, pathname=__file__, lineno=42, msg=msg, args=(), exc_info=None
    )
    record.module, record.funcName = "test_mod", "test_fn"
    for k, v in extra.items():
        setattr(record, k, v)
    return record


class TestCorrelationId:
    def test_generated_ids_are_unique(self) -> None:
        assert new_correlation_id() != new_correlation_id()

    def test_bind_then_read_back(self) -> None:
        cid = bind_correlation_id("trace-abc")
        assert cid == "trace-abc"
        assert get_correlation_id() == "trace-abc"

    def test_bind_without_a_value_generates_one(self) -> None:
        cid = bind_correlation_id()
        assert cid
        assert get_correlation_id() == cid

    def test_rebinding_replaces_the_previous_id(self) -> None:
        bind_correlation_id("first")
        bind_correlation_id("second")
        assert get_correlation_id() == "second"

    @pytest.mark.parametrize("bad", ["x" * 65, "has space", "line\nbreak", 'quote"d'])
    def test_malformed_inbound_id_is_replaced(self, bad: str) -> None:
        cid = bind_correlation_id(bad)
        assert cid != bad
        assert len(cid) == 32

    def test_64_char_id_is_kept(self) -> None:
        assert bind_correlation_id("a" * 64) == "a" * 64


class TestCorrelationIdFilter:
    def test_stamps_the_current_id_onto_a_record(self) -> None:
        bind_correlation_id("trace-filter")
        record = _record()
        assert CorrelationIdFilter().filter(record) is True
        assert record.correlation_id == "trace-filter"

    def test_does_not_overwrite_an_id_already_on_the_record(self) -> None:
        """A worker replaying a job carries the original request's id."""
        bind_correlation_id("ambient")
        record = _record(correlation_id="explicit")
        CorrelationIdFilter().filter(record)
        assert record.correlation_id == "explicit"


class TestJsonFormatter:
    def test_emits_one_parseable_json_object(self) -> None:
        bind_correlation_id("trace-json")
        out = JsonFormatter().format(_record("user_login"))
        parsed = json.loads(out)
        assert parsed["event"] == "user_login"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "api.test"
        assert parsed["correlation_id"] == "trace-json"

    def test_includes_a_source_location(self) -> None:
        parsed = json.loads(JsonFormatter().format(_record()))
        assert parsed["source"] == "test_mod:test_fn:42"

    def test_timestamp_is_iso_8601_utc(self) -> None:
        parsed = json.loads(JsonFormatter().format(_record()))
        assert "T" in parsed["timestamp"]
        assert parsed["timestamp"].endswith("+00:00")

    def test_null_and_empty_keys_are_dropped(self) -> None:
        """Log stores should not fill up with nulls - `taskName` especially."""
        parsed = json.loads(JsonFormatter().format(_record(taskName=None, blank="")))
        assert "taskName" not in parsed
        assert "blank" not in parsed

    def test_extra_fields_survive_into_the_output(self) -> None:
        parsed = json.loads(JsonFormatter().format(_record(model="resnet50", latency_ms=12.5)))
        assert parsed["model"] == "resnet50"
        assert parsed["latency_ms"] == 12.5


class TestConsoleFormatter:
    def test_renders_a_human_readable_line(self) -> None:
        out = ConsoleFormatter().format(_record("cache_hit"))
        assert "cache_hit" in out
        assert "INFO" in out

    def test_includes_the_logger_name(self) -> None:
        assert "api.test" in ConsoleFormatter().format(_record())

    def test_renders_every_level(self) -> None:
        for level in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ):
            assert ConsoleFormatter().format(_record(level=level))


class TestConfigureLogging:
    def test_json_format_attaches_a_json_formatter(self) -> None:
        configure_logging("INFO", "json")
        handlers = logging.getLogger().handlers
        assert handlers
        assert any(isinstance(h.formatter, JsonFormatter) for h in handlers)

    def test_console_format_attaches_a_console_formatter(self) -> None:
        configure_logging("DEBUG", "console")
        handlers = logging.getLogger().handlers
        assert any(isinstance(h.formatter, ConsoleFormatter) for h in handlers)

    def test_level_is_applied(self) -> None:
        configure_logging("WARNING", "json")
        assert logging.getLogger().level == logging.WARNING

    def test_reconfiguring_does_not_stack_handlers(self) -> None:
        """Called once per process - but a double call must not double-log."""
        configure_logging("INFO", "json")
        first = len(logging.getLogger().handlers)
        configure_logging("INFO", "json")
        assert len(logging.getLogger().handlers) == first

    def test_teardown_restores_a_quiet_root(self) -> None:
        configure_logging("CRITICAL", "json")
        assert logging.getLogger().level == logging.CRITICAL


class TestGetLogger:
    def test_returns_a_logger_with_the_requested_name(self) -> None:
        assert get_logger("api.services.thing").name == "api.services.thing"

    def test_same_name_returns_the_same_logger(self) -> None:
        assert get_logger("api.same") is get_logger("api.same")
