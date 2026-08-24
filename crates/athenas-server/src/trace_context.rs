//! W3C Trace Context extraction middleware.
//!
//! Extracts `traceparent` and `tracestate` headers from incoming HTTP
//! requests and creates a server span linked to the caller's trace.
//! This enables Autopsy's dependency topology to infer service-to-service
//! edges when an instrumented client calls the Athenas API.
//!
//! Uses the OpenTelemetry API directly (not tracing-opentelemetry) to
//! create the span, because `tracing-opentelemetry`'s `set_parent()`
//! does not update the `trace_id` that was already assigned in
//! `on_new_span`. By calling `tracer.start_with_context()` directly,
//! we guarantee the span inherits the remote trace_id and parent_span_id.

use axum::{extract::Request, http::HeaderMap, middleware::Next, response::Response};
use opentelemetry::{
    propagation::Extractor,
    trace::{Span, Tracer},
    KeyValue,
};

/// Adapter so opentelemetry's propagator can read from axum HeaderMap.
struct HeaderExtractor<'a>(&'a HeaderMap);

impl<'a> Extractor for HeaderExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}

/// Middleware that extracts W3C trace context from incoming request headers
/// and creates a server span as a child of the caller's trace.
///
/// When a client instrumented with OpenTelemetry sends a request with a
/// `traceparent` header, this middleware:
/// 1. Extracts the remote span context from the headers
/// 2. Creates a server span (`http.server.handle`) with the remote context
///    as parent — inheriting the caller's trace_id and parent_span_id
/// 3. Exports the span when the request completes
///
/// This enables Autopsy's dependency topology to infer the edge
/// `client-service → athenas-studio`.
pub async fn trace_context_middleware(req: Request, next: Next) -> Response {
    // Extract the remote context from W3C traceparent/tracestate headers
    let remote_context = opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(req.headers()))
    });

    let method = req.method().to_string();
    let route = req.uri().path().to_string();

    // Create a server span directly via the OTEL API.
    // start_with_context uses the remote parent's trace_id and span_id,
    // which is exactly what Autopsy needs to infer the dependency edge.
    let tracer = opentelemetry::global::tracer("athenas-server");
    let mut span = tracer.start_with_context("http.server.handle", &remote_context);
    span.set_attribute(KeyValue::new("http.method", method));
    span.set_attribute(KeyValue::new("http.route", route));
    span.set_attribute(KeyValue::new("span.kind", "server"));

    let response = next.run(req).await;

    span.set_attribute(KeyValue::new(
        "http.status_code",
        response.status().as_u16() as i64,
    ));
    if !response.status().is_success() {
        span.set_attribute(KeyValue::new("error", true));
    }

    // End the span — this queues it for batch export via OTLP
    span.end();

    response
}
