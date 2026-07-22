use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts, TextEncoder};

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
        // Use the global registry to avoid "AlreadyReg" panics when
        // Metrics::new() is called more than once (e.g. server restart in TUI).
        let registry = prometheus::default_registry();

        let requests_total = IntCounterVec::new(
            Opts::new("athenas_requests_total", "Total number of requests")
                .variable_labels(vec!["endpoint".into(), "method".into()]),
            &["endpoint", "method"],
        )
        .expect("create counter");
        registry.register(Box::new(requests_total.clone())).ok(); // ignore AlreadyReg

        let requests_active = IntGauge::new(
            "athenas_requests_active",
            "Number of active in-flight requests",
        )
        .expect("create gauge");
        registry.register(Box::new(requests_active.clone())).ok();

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                "athenas_request_duration_seconds",
                "Request duration in seconds",
            )
            .variable_labels(vec!["endpoint".into()])
            .buckets(vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
            &["endpoint"],
        )
        .expect("create histogram");
        registry.register(Box::new(request_duration.clone())).ok();

        let tokens_prompt_total = IntCounterVec::new(
            Opts::new(
                "athenas_tokens_prompt_total",
                "Total prompt tokens processed",
            )
            .variable_labels(vec!["model".into()]),
            &["model"],
        )
        .expect("create counter");
        registry
            .register(Box::new(tokens_prompt_total.clone()))
            .ok();

        let tokens_generated_total = IntCounterVec::new(
            Opts::new("athenas_tokens_generated_total", "Total tokens generated")
                .variable_labels(vec!["model".into()]),
            &["model"],
        )
        .expect("create counter");
        registry
            .register(Box::new(tokens_generated_total.clone()))
            .ok();

        let errors_total = IntCounterVec::new(
            Opts::new("athenas_errors_total", "Total number of errors")
                .variable_labels(vec!["endpoint".into(), "type".into()]),
            &["endpoint", "type"],
        )
        .expect("create counter");
        registry.register(Box::new(errors_total.clone())).ok();

        Self {
            requests_total,
            requests_active,
            request_duration,
            tokens_prompt_total,
            tokens_generated_total,
            errors_total,
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
