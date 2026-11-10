import json
import logging
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """
    Formats log records as structured JSON lines.
    Enables instant filtering and indexing in cloud log management platforms (Render, GCP, Datadog).
    """
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        # Capture any extra attributes passed via logging.info("...", extra={...})
        for key, val in record.__dict__.items():
            if key not in ("args", "asctime", "created", "exc_info", "exc_text", "filename",
                           "funcName", "id", "levelname", "levelno", "lineno", "module",
                           "msecs", "message", "msg", "name", "pathname", "process",
                           "processName", "relativeCreated", "stack_info", "thread", "threadName"):
                log_obj[key] = val
        return json.dumps(log_obj)

def setup_structured_logging(level=logging.INFO):
    """Configure root logger to emit structured JSON to stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove existing handlers to avoid duplicate log lines
    root_logger.handlers = [handler]
    
    # Silence overly verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

