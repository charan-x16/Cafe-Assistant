import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate } from "k6/metrics";

export const firstTokenLatency = new Trend("first_token_latency_ms");
export const errorRate = new Rate("chat_error_rate");

export const options = {
  scenarios: {
    realistic: {
      executor: "ramping-vus",
      stages: [
        { duration: "30s", target: 10 },
        { duration: "1m", target: 25 },
        { duration: "30s", target: 0 },
      ],
    },
    burst: {
      executor: "constant-arrival-rate",
      rate: 20,
      timeUnit: "1s",
      duration: "30s",
      preAllocatedVUs: 30,
      maxVUs: 80,
      startTime: "2m",
    },
  },
  thresholds: {
    first_token_latency_ms: ["p(95)<2000", "p(99)<2500"],
    chat_error_rate: ["rate<0.01"],
  },
};

const queries = [
  "Recommend a latte.",
  "I'm allergic to peanuts. Can I get a cookie?",
  "Do you have an almnd latte?",
  "I'm vegan. Recommend a coffee.",
  "How much insulin should I take for a mocha?",
];

export default function () {
  const baseUrl = __ENV.BASE_URL || "http://localhost:8000";
  const query = queries[Math.floor(Math.random() * queries.length)];
  const started = Date.now();
  const response = http.post(
    `${baseUrl}/chat`,
    JSON.stringify({
      session_id: `load-${__VU}-${__ITER}`,
      tenant_id: Number(__ENV.TENANT_ID || "1"),
      message: query,
    }),
    { headers: { "Content-Type": "application/json" } },
  );
  firstTokenLatency.add(Date.now() - started);
  errorRate.add(response.status >= 500 || response.status === 429);
  check(response, {
    "status is 200": (r) => r.status === 200,
    "no peanut unsafe leak": (r) => !query.includes("peanuts") || !r.body.includes("Peanut Butter Cookie"),
  });
  sleep(1);
}
