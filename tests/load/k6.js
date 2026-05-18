import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 20,
  duration: '60s',
};

export default function () {
  const loginRes = http.post('http://localhost:8000/api/v1/auth/token', {
    username: 'admin',
    password: 'admin123',
  }, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });

  check(loginRes, { 'login status is 200': (r) => r.status === 200 });
  const token = loginRes.json('access_token');

  const queryRes = http.post(
    'http://localhost:8000/api/v1/query',
    JSON.stringify({ user_query: 'How can I contact support?' }),
    {
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    }
  );

  check(queryRes, { 'query status is 200': (r) => r.status === 200 });
  sleep(1);
}
