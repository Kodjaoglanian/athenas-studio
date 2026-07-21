use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use prometheus::{
    register_histogram_vec, register_int_counter_vec, register_int_gauge, HistogramVec,
    IntCounterVec, IntGauge, TextEncoder,
};

pub struct Metrics {
    pub requests_total: IntCounterVec,
    pub requests_active: IntGauge,
    pub request_duration: HistogramVec,
    pub tokens_prompt_total: IntCounterVec,
    pub tokens_generated_total: IntCounterVec,
    pub errors_total: IntCounterVec,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_total: register_int_counter_vec!(
                "athenas_requests_total",
                "Total number of requests",
                &["endpoint", "method"]
            )
            .unwrap(),
            requests_active: register_int_gauge!(
                "athenas_requests_active",
                "Number of active in-flight requests"
            )
            .unwrap(),
            request_duration: register_histogram_vec!(
                "athenas_request_duration_seconds",
                "Request duration in seconds",
                &["endpoint"],
                vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,]
            )
            .unwrap(),
            tokens_prompt_total: register_int_counter_vec!(
                "athenas_tokens_prompt_total",
                "Total prompt tokens processed",
                &["model"]
            )
            .unwrap(),
            tokens_generated_total: register_int_counter_vec!(
                "athenas_tokens_generated_total",
                "Total tokens generated",
                &["model"]
            )
            .unwrap(),
            errors_total: register_int_counter_vec!(
                "athenas_errors_total",
                "Total number of errors",
                &["endpoint", "type"]
            )
            .unwrap(),
        }
    }

    pub fn record_tokens(&self, model: &str, prompt: u32, generated: u32) {
        self.tokens_prompt_total
            .with_label_values(&[model])
            .inc_by(prompt as u64);
        self.tokens_generated_total
            .with_label_values(&[model])
            .inc_by(generated as u64);
    }

    pub fn render() -> String {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = String::new();
        encoder.encode_utf8(&metric_families, &mut buffer).unwrap();
        buffer
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

pub type SharedMetrics = Arc<Metrics>;

pub async fn metrics_middleware(
    State(metrics): State<SharedMetrics>,
    req: Request,
    next: Next,
) -> Response {
    let endpoint = req.uri().path().to_string();
    let method = req.method().to_string();

    metrics
        .requests_total
        .with_label_values(&[&endpoint, &method])
        .inc();
    metrics.requests_active.inc();

    let start = Instant::now();
    let response = next.run(req).await;
    let duration = start.elapsed().as_secs_f64();

    metrics.requests_active.dec();
    metrics
        .request_duration
        .with_label_values(&[&endpoint])
        .observe(duration);

    if !response.status().is_success() {
        metrics
            .errors_total
            .with_label_values(&[&endpoint, "http_error"])
            .inc();
    }

    response
}
