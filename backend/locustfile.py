"""
AskMyNotes - Cloud Performance & Scalability Load Test Suite (Locust)
Simulates student traffic across different workloads to measure:
- RPS (Requests Per Second)
- Median & P95 latency
- Failure rate under concurrency spikes

Usage:
  1. Install locust: pip install locust
  2. Run locally against backend:
       locust -f locustfile.py --host=http://127.0.0.1:8000
  3. Open web dashboard: http://localhost:8089
"""

from locust import HttpUser, task, between
import random

class StudentUser(HttpUser):
    # Wait between 1 and 3 seconds between actions to simulate human reading time
    wait_time = between(1.0, 3.0)

    @task(5)
    def check_health(self):
        """High-frequency probe (e.g. heartbeat / health checks)."""
        self.client.get("/health", name="[Probe] GET /health")

    @task(3)
    def get_service_root(self):
        """Simulate pinging the root API metadata endpoint."""
        self.client.get("/", name="[API] GET /")

    @task(2)
    def fetch_api_schema(self):
        """Simulate initial client loading Swagger/OpenAPI spec."""
        self.client.get("/api/v1/openapi.json", name="[Docs] GET /api/v1/openapi.json")

    @task(1)
    def test_mock_upload_validation(self):
        """Simulate a client submitting a bad upload request to verify 422 error rate & latency."""
        self.client.post(
            "/api/v1/upload/",
            files={"file": ("dummy.txt", b"Invalid file payload test for rate checking")},
            name="[Upload] POST /api/v1/upload/ (Validation Check)",
        )
