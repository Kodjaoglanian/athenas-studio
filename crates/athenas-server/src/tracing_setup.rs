use athenas_core::OtelConfig;
use opentelemetry::trace::TracerProvider as TracerProviderTrait;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::TracerProvider as SdkTracerProvider;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::Sampler;
use opentelemetry_semantic_conventions::resource::SERVICE_NAME;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Initialize OpenTelemetry tracing and return a guard for cleanup.
pub fn init_tracing(config: &OtelConfig) -> Option<SdkTracerProvider> {
    if !config.enabled {
        // Just initialize basic tracing
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();
        return None;
    }

    let resource = Resource::new(vec![opentelemetry::KeyValue::new(
        SERVICE_NAME,
        config.service_name.clone(),
    )]);

    let sampler = if config.sample_ratio >= 1.0 {
        Sampler::AlwaysOn
    } else {
        Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(config.sample_ratio)))
    };

    let provider = if let Some(endpoint) = &config.endpoint {
        // OTLP exporter
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .expect("Failed to create OTLP exporter");

        SdkTracerProvider::builder()
            .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
            .with_resource(resource)
            .with_sampler(sampler)
            .build()
    } else {
        // No exporter - just use a no-op provider with logging
        SdkTracerProvider::builder()
            .with_resource(resource)
            .with_sampler(sampler)
            .build()
    };

    let tracer = TracerProviderTrait::tracer(&provider, config.service_name.clone());

    let otel_layer = OpenTelemetryLayer::new(tracer);

    let _ = tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .try_init();

    Some(provider)
}

/// Shut down the tracing provider, flushing any pending spans.
pub fn shutdown_tracing(provider: Option<SdkTracerProvider>) {
    if let Some(provider) = provider {
        if let Err(e) = provider.shutdown() {
            eprintln!("Failed to shutdown OpenTelemetry provider: {:?}", e);
        }
    }
}
