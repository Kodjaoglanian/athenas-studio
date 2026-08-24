//! W3C Trace Context extraction middleware.
//!
//! Extracts `traceparent` and `tracestate` headers from incoming HTTP
//! requests and links the Athenas server spans to the caller's trace.
//! This enables Autopsy's dependency topology to infer service-to-service
//! edges when an instrumented client calls the Athenas API.

use axum::{extract::Request, http::HeaderMap, middleware::Next, response::Response};
use opentelemetry::propagation::Extractor;
use tracing_opentelemetry::OpenTelemetrySpanExt;

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
/// and creates a parent span linked to the caller's trace.
///
/// When a client instrumented with OpenTelemetry sends a request with a
/// `traceparent` header, this middleware extracts the remote span context
/// and sets it as the parent of the server-side trace. All downstream spans
/// (from `#[tracing::instrument]` in route handlers) become children of
/// this parent, allowing Autopsy to infer the dependency edge.
pub async fn trace_context_middleware(req: Request, next: Next) -> Response {
    // Extract the remote context from W3C traceparent/tracestate headers
    let remote_context = opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(req.headers()))
    });

    // Create a server span and link it to the remote parent context
    let span = tracing::info_span!(
        "http.server.handle",
        http.method = %req.method(),
        http.route = %req.uri().path(),
        http.status_code = tracing::field::Empty,
    );
    span.set_parent(remote_context);

    // Enter the span so all child spans (from #[tracing::instrument])
    // are linked to this parent
    let _guard = span.enter();
    let response = next.run(req).await;

    // Record the response status code
    if let Some(status) = response.status().as_u16().into() {
        span.record("http.status_code", status);
    }

    response
}
