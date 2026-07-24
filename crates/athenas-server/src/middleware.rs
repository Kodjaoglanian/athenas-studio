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

// ─── IP Allowlist / Denylist Middleware ───

/// IP filter configuration for allowlist/denylist.
#[derive(Clone, Debug)]
pub struct IpFilterConfig {
    /// List of allowed IPs/CIDRs. Empty = allow all (subject to denylist).
    pub allowlist: Vec<String>,
    /// List of denied IPs/CIDRs. These are always blocked.
    pub denylist: Vec<String>,
}

impl IpFilterConfig {
    pub fn new(allowlist: Vec<String>, denylist: Vec<String>) -> Self {
        Self { allowlist, denylist }
    }

    /// Check if an IP address is allowed.
    pub fn is_allowed(&self, ip: &IpAddr) -> bool {
        // Check denylist first
        for entry in &self.denylist {
            if ip_matches(ip, entry) {
                return false;
            }
        }

        // If allowlist is empty, allow all (not in denylist)
        if self.allowlist.is_empty() {
            return true;
        }

        // Check allowlist
        for entry in &self.allowlist {
            if ip_matches(ip, entry) {
                return true;
            }
        }

        false
    }
}

/// Check if an IP address matches an entry (IP or CIDR).
fn ip_matches(ip: &IpAddr, entry: &str) -> bool {
    // Try CIDR notation first
    if let Some((range_ip, prefix_len)) = entry.split_once('/') {
        if let Ok(prefix_len) = prefix_len.parse::<u8>() {
            return cidr_match(ip, range_ip, prefix_len);
        }
    }

    // Try exact IP match
    if let Ok(entry_ip) = entry.parse::<IpAddr>() {
        return ip == &entry_ip;
    }

    false
}

/// Check if an IP is within a CIDR range.
fn cidr_match(ip: &IpAddr, range_ip: &str, prefix_len: u8) -> bool {
    match (ip, range_ip.parse()) {
        (IpAddr::V4(addr), Ok(IpAddr::V4(range))) => {
            if prefix_len > 32 {
                return false;
            }
            let mask = if prefix_len == 0 {
                0u32
            } else {
                u32::MAX << (32 - prefix_len)
            };
            (u32::from(*addr) & mask) == (u32::from(range) & mask)
        }
        (IpAddr::V6(addr), Ok(IpAddr::V6(range))) => {
            if prefix_len > 128 {
                return false;
            }
            let addr_bytes = addr.octets();
            let range_bytes = range.octets();
            let full_bytes = (prefix_len / 8) as usize;
            let remainder_bits = prefix_len % 8;

            // Compare full bytes
            if addr_bytes[..full_bytes] != range_bytes[..full_bytes] {
                return false;
            }

            // Compare remaining bits
            if remainder_bits > 0 && full_bytes < 16 {
                let mask = 0xFFu8 << (8 - remainder_bits);
                if (addr_bytes[full_bytes] & mask) != (range_bytes[full_bytes] & mask) {
                    return false;
                }
            }

            true
        }
        _ => false,
    }
}

/// IP filter middleware.
pub async fn ip_filter_middleware(
    State(filter): State<IpFilterConfig>,
    req: Request,
    next: Next,
) -> Response {
    let ip = req
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip())
        .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)));

    if !filter.is_allowed(&ip) {
        tracing::warn!("IP {} blocked by filter", ip);
        return (
            StatusCode::FORBIDDEN,
            "Access denied: IP not allowed.",
        )
            .into_response();
    }

    next.run(req).await
}
