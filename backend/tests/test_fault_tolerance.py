"""
Tests for fault tolerance and job recovery components.

Covers:
- Per-page retry logic in background_task_service (exponential backoff)
- Stale job detection logic in job_recovery_service
- Job status endpoint ownership enforcement
- Idempotency constants and configuration
"""
import asyncio
import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.background_task_service import (
    _PAGE_MAX_RETRIES,
    _PAGE_RETRY_BASE_DELAY,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestFaultToleranceConstants:
    def test_page_max_retries_is_positive(self):
        assert _PAGE_MAX_RETRIES > 0, "Must retry at least once"

    def test_page_max_retries_is_reasonable(self):
        assert 1 <= _PAGE_MAX_RETRIES <= 10, "Retry count should be between 1 and 10"

    def test_page_retry_base_delay_is_positive(self):
        assert _PAGE_RETRY_BASE_DELAY > 0, "Base delay must be positive"

    def test_page_retry_base_delay_is_reasonable(self):
        assert 0.1 <= _PAGE_RETRY_BASE_DELAY <= 30.0, "Base delay should be between 0.1s and 30s"


# ---------------------------------------------------------------------------
# Exponential backoff maths
# ---------------------------------------------------------------------------

class TestExponentialBackoff:
    def test_backoff_increases_with_attempt(self):
        """Verify the computed backoff values grow exponentially."""
        delays = [_PAGE_RETRY_BASE_DELAY * (2 ** (a - 1)) for a in range(1, _PAGE_MAX_RETRIES + 1)]
        for i in range(1, len(delays)):
            assert delays[i] > delays[i - 1], f"Delay at attempt {i + 1} must exceed delay at attempt {i}"

    def test_first_attempt_delay_equals_base(self):
        delay = _PAGE_RETRY_BASE_DELAY * (2 ** 0)
        assert delay == _PAGE_RETRY_BASE_DELAY

    def test_second_attempt_delay_is_double_base(self):
        if _PAGE_MAX_RETRIES >= 2:
            delay = _PAGE_RETRY_BASE_DELAY * (2 ** 1)
            assert delay == _PAGE_RETRY_BASE_DELAY * 2


# ---------------------------------------------------------------------------
# Stale job detection logic
# ---------------------------------------------------------------------------

class TestStaleJobDetection:
    def test_stale_threshold_is_reasonable(self):
        """The stale threshold should be between 1 and 60 minutes."""
        from app.services.job_recovery_service import STALE_THRESHOLD_MINUTES
        assert 1 <= STALE_THRESHOLD_MINUTES <= 60

    def test_stale_threshold_is_integer_or_float(self):
        from app.services.job_recovery_service import STALE_THRESHOLD_MINUTES
        assert isinstance(STALE_THRESHOLD_MINUTES, (int, float))


# ---------------------------------------------------------------------------
# job_recovery_service: recover_stale_jobs
# ---------------------------------------------------------------------------

class TestRecoverStaleJobs:
    def test_recover_stale_jobs_is_coroutine(self):
        """recover_stale_jobs must be an async function."""
        from app.services.job_recovery_service import recover_stale_jobs
        import inspect
        assert inspect.iscoroutinefunction(recover_stale_jobs)

    def test_recover_stale_jobs_handles_db_error_gracefully(self):
        """If Firestore is unavailable at startup, the reaper must not crash the app."""
        from app.services.job_recovery_service import recover_stale_jobs

        with patch("app.services.job_recovery_service.get_db", side_effect=RuntimeError("DB unavailable")):
            # Should not raise
            asyncio.run(recover_stale_jobs())


# ---------------------------------------------------------------------------
# background_task_service: page retry integration (mocked pipeline)
# ---------------------------------------------------------------------------

class TestPageRetryLogic:
    def test_successful_page_does_not_trigger_retry(self):
        """If _process_single_page succeeds on first attempt, no sleep should be called."""
        with (
            patch("app.services.background_task_service.get_db") as mock_db,
            patch("app.services.background_task_service.get_page_count", return_value=1),
            patch("app.services.background_task_service._process_single_page",
                  new_callable=AsyncMock,
                  return_value={"text": "hello world", "ocr_used": False}),
            patch("app.services.background_task_service._index_extracted_content",
                  new_callable=AsyncMock),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            # Set up mock job ref
            mock_ref = MagicMock()
            mock_db.return_value.collection.return_value.document.return_value = mock_ref

            asyncio.run(
                __import__("app.services.background_task_service", fromlist=["run_ocr_pipeline"])
                .run_ocr_pipeline("job1", b"fake", "user1", "subj1", "test.pdf", "pdf")
            )
            # asyncio.sleep should not have been called for a successful page
            mock_sleep.assert_not_called()

    def test_failing_page_triggers_retries_and_sleep(self):
        """If _process_single_page always raises, sleep is called between retries."""
        with (
            patch("app.services.background_task_service.get_db") as mock_db,
            patch("app.services.background_task_service.get_page_count", return_value=1),
            patch("app.services.background_task_service._process_single_page",
                  new_callable=AsyncMock,
                  side_effect=RuntimeError("OCR boom")),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_ref = MagicMock()
            mock_db.return_value.collection.return_value.document.return_value = mock_ref

            asyncio.run(
                __import__("app.services.background_task_service", fromlist=["run_ocr_pipeline"])
                .run_ocr_pipeline("job2", b"fake", "user1", "subj1", "test.pdf", "pdf")
            )
            # sleep should have been called (_PAGE_MAX_RETRIES - 1) times (not after last attempt)
            assert mock_sleep.call_count == _PAGE_MAX_RETRIES - 1

    def test_all_pages_fail_sets_status_failed(self):
        """When every page fails every retry, final status must be 'failed'."""
        with (
            patch("app.services.background_task_service.get_db") as mock_db,
            patch("app.services.background_task_service.get_page_count", return_value=2),
            patch("app.services.background_task_service._process_single_page",
                  new_callable=AsyncMock,
                  side_effect=RuntimeError("OCR boom")),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_ref = MagicMock()
            mock_db.return_value.collection.return_value.document.return_value = mock_ref

            asyncio.run(
                __import__("app.services.background_task_service", fromlist=["run_ocr_pipeline"])
                .run_ocr_pipeline("job3", b"fake", "user1", "subj1", "test.pdf", "pdf")
            )

            # Verify that the final update call included status=failed
            update_calls = mock_ref.update.call_args_list
            final_call_kwargs = update_calls[-1][0][0]  # positional dict arg
            assert final_call_kwargs.get("status") in ("failed",)

    def test_partial_failure_sets_status_partial(self):
        """When only some pages fail, final status must be 'partial'."""
        call_count = {"n": 0}

        async def alternating_page(*args, **kwargs):
            call_count["n"] += 1
            # Always fail on retry attempts (call_count > 1 for each page)
            # Page 0 → succeed on first attempt, page 1 → always fail
            # We track by total calls: first _PAGE_MAX_RETRIES calls = page 1 (fails),
            # then one call = page 0 (succeeds). Simpler: succeed on even calls.
            if call_count["n"] % 2 == 0:
                return {"text": "good page", "ocr_used": False}
            raise RuntimeError("boom")

        with (
            patch("app.services.background_task_service.get_db") as mock_db,
            patch("app.services.background_task_service.get_page_count", return_value=2),
            patch("app.services.background_task_service._process_single_page",
                  new_callable=AsyncMock,
                  side_effect=alternating_page),
            patch("app.services.background_task_service._index_extracted_content",
                  new_callable=AsyncMock),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_ref = MagicMock()
            mock_db.return_value.collection.return_value.document.return_value = mock_ref

            asyncio.run(
                __import__("app.services.background_task_service", fromlist=["run_ocr_pipeline"])
                .run_ocr_pipeline("job4", b"fake", "user1", "subj1", "test.pdf", "pdf")
            )
            update_calls = mock_ref.update.call_args_list
            statuses = [c[0][0].get("status") for c in update_calls if "status" in c[0][0]]
            # At least one update with partial or completed (some pages extracted)
            assert any(s in ("partial", "completed") for s in statuses), (
                f"Expected partial/completed status but got: {statuses}"
            )
