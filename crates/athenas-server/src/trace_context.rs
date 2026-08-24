//! W3C Trace Context extraction middleware.
//!
//! Extracts `traceparent` and `tracestate` headers from incoming HTTP
//! requests and creates a server span linked to the caller's trace.
//! This enables Autopsy's dependency topology to infer service-to-service
//! edges when an instrumented client calls the Athenas API.

use axum::{extract::Request, http::HeaderMap, middleware::Next, response::Response};
use opentelemetry::{
    propagation::Extractor,
    trace::{Span, SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState, Tracer},
    Context, KeyValue,
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

/// Manually parse a W3C traceparent header into a SpanContext.
///
/// The opentelemetry_sdk 0.27 TraceContextPropagator rejects trace_flags > 2
/// for version 0, but the Python OTEL SDK sends flags=03. We parse manually
/// to be permissive.
fn parse_traceparent(header: &str) -> Option<SpanContext> {
    let parts: Vec<&str> = header.trim().split('-').collect();
    if parts.len() < 4 {
        return None;
    }

    let version = u8::from_str_radix(parts[0], 16).ok()?;
    if version > 2 {
        return None;
    }

    let trace_id = TraceId::from_hex(parts[1]).ok()?;
    let span_id = SpanId::from_hex(parts[2]).ok()?;

    // Parse trace flags — accept any value, mask to SAMPLED bit
    let opts = u8::from_str_radix(parts[3], 16).ok()?;
    let trace_flags = TraceFlags::new(opts) & TraceFlags::SAMPLED;

    let span_context =
        SpanContext::new(trace_id, span_id, trace_flags, true, TraceState::default());
    if span_context.is_valid() {
        Some(span_context)
    } else {
        None
    }
}

/// Middleware that extracts W3C trace context from incoming request headers
/// and creates a server span as a child of the caller's trace.
pub async fn trace_context_middleware(req: Request, next: Next) -> Response {
    // Try manual parse of traceparent first (permissive), then fall back
    // to the global propagator (for tracestate, baggage, etc.)
    let remote_context = if let Some(tp) = req
        .headers()
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
    {
        if let Some(span_context) = parse_traceparent(tp) {
            eprintln!(
                "trace_context: extracted parent trace_id={}, span_id={}",
                span_context.trace_id(),
                span_context.span_id(),
            );
            Context::current().with_remote_span_context(span_context)
        } else {
            eprintln!("trace_context: failed to parse traceparent={:?}", tp);
            Context::current()
        }
    } else {
        // Fall back to global propagator (handles tracestate, baggage)
        opentelemetry::global::get_text_map_propagator(|propagator| {
            propagator.extract(&HeaderExtractor(req.headers()))
        })
    };

    let method = req.method().to_string();
    let route = req.uri().path().to_string();

    // Create a server span directly via the OTEL API.
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

    span.end();

    response
}
