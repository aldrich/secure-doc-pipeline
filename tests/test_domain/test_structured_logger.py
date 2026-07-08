import json
import logging

from domain.structured_logger import StructuredFormatter


class TestStructuredFormatter:
    def setup_method(self):
        self.formatter = StructuredFormatter()

    def test_format_basic_record(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
        assert "extra" not in parsed

    def test_format_record_with_extras(self):
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        record.session_id = "abc123"
        record.latency = 0.45
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert parsed["extra"]["session_id"] == "abc123"
        assert parsed["extra"]["latency"] == 0.45

    def test_private_attrs_excluded(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        record._internal = "should not appear"
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert "extra" not in parsed

    def test_standard_attrs_excluded_from_extra(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert "extra" not in parsed