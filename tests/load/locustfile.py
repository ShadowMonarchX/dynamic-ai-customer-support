from __future__ import annotations

from locust import HttpUser, between, task


class CustomerSupportUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        response = self.client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"},
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        payload = response.json()
        self.token = payload["access_token"]
        self.headers = {"authorization": f"Bearer {self.token}"}

    @task(2)
    def query_contact(self):
        self.client.post(
            "/api/v1/query",
            json={"user_query": "How can I contact support?"},
            headers=self.headers,
        )

    @task(1)
    def query_identity(self):
        self.client.post(
            "/api/v1/query",
            json={"user_query": "Who is Nayan Raval?"},
            headers=self.headers,
        )
