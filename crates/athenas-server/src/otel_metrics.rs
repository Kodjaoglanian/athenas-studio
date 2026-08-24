//! OpenTelemetry metrics instruments — mirrors the Prometheus metrics
//! so that OTLP collectors (like Autopsy) receive the same data.
//!
//! Instruments are created lazily via a global `OnceLock` and use the
//! global meter provider registered by `tracing_setup::init_tracing()`.
//! When OTEL is disabled, the instruments are no-ops (the global meter
//! provider returns no-op instruments).

use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::{global, KeyValue};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::OnceLock;

/// Atomic counter for active requests — shared with the observable gauge
/// so the OTLP gauge reports the same value the Prometheus gauge does.
static ACTIVE_REQUESTS: AtomicI64 = AtomicI64::new(0);

static OTEL_METRICS: OnceLock<OtelMetrics> = OnceLock::new();

pub struct OtelMetrics {
    #[allow(dead_code)]
    meter: Meter,
    requests_total: Counter<u64>,
    request_duration: Histogram<f64>,
    tokens_prompt_total: Counter<u64>,
    tokens_generated_total: Counter<u64>,
    errors_total: Counter<u64>,
}

impl OtelMetrics {
    fn new() -> Self {
        let meter = global::meter("athenas-server");

        let requests_total = meter
            .u64_counter("athenas_requests_total")
            .with_description("Total number of requests")
            .with_unit("{request}")
            .build();

        let request_duration = meter
            .f64_histogram("athenas_request_duration_seconds")
            .with_description("Request duration in seconds")
            .with_unit("s")
            .build();

        let tokens_prompt_total = meter
            .u64_counter("athenas_tokens_prompt_total")
            .with_description("Total prompt tokens processed")
            .with_unit("{token}")
            .build();

        let tokens_generated_total = meter
            .u64_counter("athenas_tokens_generated_total")
            .with_description("Total tokens generated")
            .with_unit("{token}")
            .build();

        let errors_total = meter
            .u64_counter("athenas_errors_total")
            .with_description("Total number of errors")
            .with_unit("{error}")
            .build();

        // Observable gauge for active requests — reads from the static atomic
        let _active_gauge = meter
            .i64_observable_gauge("athenas_requests_active")
            .with_description("Number of active in-flight requests")
            .with_unit("{request}")
            .with_callback(|m| {
                m.observe(ACTIVE_REQUESTS.load(Ordering::Relaxed), &[]);
            })
            .build();

        Self {
            meter,
            requests_total,
            request_duration,
            tokens_prompt_total,
            tokens_generated_total,
            errors_total,
        }
    }
}

/// Get the global OTEL metrics instance, initializing lazily on first call.
/// Safe to call from multiple threads — `OnceLock` guarantees single init.
pub fn otel_metrics() -> &'static OtelMetrics {
    OTEL_METRICS.get_or_init(OtelMetrics::new)
}

// --- Convenience functions for the middleware ---

pub fn record_request(endpoint: &str, method: &str) {
    otel_metrics().requests_total.add(
        1,
        &[
            KeyValue::new("endpoint", endpoint.to_string()),
            KeyValue::new("method", method.to_string()),
        ],
    );
}

pub fn record_duration(endpoint: &str, duration_secs: f64) {
    otel_metrics().request_duration.record(
        duration_secs,
        &[KeyValue::new("endpoint", endpoint.to_string())],
    );
}

pub fn record_tokens(model: &str, prompt: u64, generated: u64) {
    let m = otel_metrics();
    m.tokens_prompt_total
        .add(prompt, &[KeyValue::new("model", model.to_string())]);
    m.tokens_generated_total
        .add(generated, &[KeyValue::new("model", model.to_string())]);
}

pub fn record_error(endpoint: &str, error_type: &str) {
    otel_metrics().errors_total.add(
        1,
        &[
            KeyValue::new("endpoint", endpoint.to_string()),
            KeyValue::new("type", error_type.to_string()),
        ],
    );
}

pub fn inc_active() {
    ACTIVE_REQUESTS.fetch_add(1, Ordering::Relaxed);
}

pub fn dec_active() {
    ACTIVE_REQUESTS.fetch_sub(1, Ordering::Relaxed);
}
