# Configure the application's root logger to emit INFO-level
# messages to standard output using structured JSON format.
import json
import logging
from typing import Any, Dict

class StructuredFormatter(logging.Formatter):
    _standard_attrs = frozenset({
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    })

    def format(self, record):
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in self._standard_attrs and not k.startswith("_")
        }
        payload: dict[str, Any] = {"level": record.levelname, "message": record.getMessage()}
        if extra:
            payload["extra"] = extra
        return json.dumps(payload)