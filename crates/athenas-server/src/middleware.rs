use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{ConnectInfo, Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use tokio::sync::Mutex;

pub struct RateLimiter {
    buckets: Mutex<HashMap<IpAddr, Bucket>>,
    max_tokens: u32,
    refill_rate: Duration,
}

struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

impl RateLimiter {
    pub fn new(max_tokens: u32, refill_per_second: u32) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            max_tokens,
            refill_rate: Duration::from_secs_f64(1.0 / refill_per_second as f64),
        }
    }

    async fn check(&self, ip: IpAddr) -> bool {
        let mut buckets = self.buckets.lock().await;
        let now = Instant::now();
        let bucket = buckets.entry(ip).or_insert(Bucket {
            tokens: self.max_tokens as f64,
            last_refill: now,
        });

        let elapsed = now.duration_since(bucket.last_refill);
        let refilled = elapsed.as_secs_f64() / self.refill_rate.as_secs_f64();
        bucket.tokens = (bucket.tokens + refilled).min(self.max_tokens as f64);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

pub type SharedRateLimiter = Arc<RateLimiter>;

pub async fn rate_limit_middleware(
    State(limiter): State<SharedRateLimiter>,
    req: Request,
    next: Next,
) -> Response {
    let ip = req
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip())
        .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)));

    if !limiter.check(ip).await {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded. Slow down.",
        )
            .into_response();
    }

    next.run(req).await
}
