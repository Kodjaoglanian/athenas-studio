use athenas_core::OtelConfig;
use opentelemetry::trace::TracerProvider as TracerProviderTrait;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::logs::LoggerProvider;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::Sampler;
use opentelemetry_sdk::trace::TracerProvider as SdkTracerProvider;
use opentelemetry_sdk::Resource;
// Semantic convention attribute keys — using string literals for compatibility
// across opentelemetry-semantic-conventions versions.
const SERVICE_NAME_KEY: &str = "service.name";
const SERVICE_VERSION_KEY: &str = "service.version";
const HOST_NAME_KEY: &str = "host.name";
const SERVICE_INSTANCE_ID_KEY: &str = "service.instance.id";
const SERVICE_NAMESPACE_KEY: &str = "service.namespace";
const DEPLOYMENT_ENVIRONMENT_NAME_KEY: &str = "deployment.environment.name";
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Holds the providers that need to be shut down on exit.
pub struct OtelGuards {
    tracer_provider: Option<SdkTracerProvider>,
    logger_provider: Option<LoggerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl OtelGuards {
    pub fn empty() -> Self {
        Self {
            tracer_provider: None,
            logger_provider: None,
            meter_provider: None,
        }
    }
}

/// Build the OTLP resource with rich attributes for Autopsy service catalog.
fn build_resource(config: &OtelConfig) -> Resource {
    let mut kvs = vec![opentelemetry::KeyValue::new(
        SERVICE_NAME_KEY,
        config.service_name.clone(),
    )];

    // Service version from crate CARGO_PKG_VERSION
    kvs.push(opentelemetry::KeyValue::new(
        SERVICE_VERSION_KEY,
        env!("CARGO_PKG_VERSION").to_string(),
    ));

    // Host name
    if let Ok(hostname) = hostname() {
        kvs.push(opentelemetry::KeyValue::new(HOST_NAME_KEY, hostname));
    }

    // Service instance ID — auto-generate from hostname+pid if not configured
    let instance_id = config.service_instance_id.clone().unwrap_or_else(|| {
        let host = hostname().unwrap_or_else(|_| "unknown".to_string());
        format!("{}-{}", host, std::process::id())
    });
    kvs.push(opentelemetry::KeyValue::new(
        SERVICE_INSTANCE_ID_KEY,
        instance_id,
    ));

    if let Some(ns) = &config.service_namespace {
        kvs.push(opentelemetry::KeyValue::new(
            SERVICE_NAMESPACE_KEY,
            ns.clone(),
        ));
    }

    if let Some(env) = &config.environment {
        kvs.push(opentelemetry::KeyValue::new(
            DEPLOYMENT_ENVIRONMENT_NAME_KEY,
            env.clone(),
        ));
    }

    Resource::new(kvs)
}

fn hostname() -> std::io::Result<String> {
    // Try /etc/hostname first, then HOSTNAME env var, then uname
    if let Ok(name) = std::fs::read_to_string("/etc/hostname") {
        let name = name.trim().to_string();
        if !name.is_empty() {
            return Ok(name);
        }
    }
    if let Ok(name) = std::env::var("HOSTNAME") {
        if !name.is_empty() {
            return Ok(name);
        }
    }
    // Fallback: use uname via /proc/sys/kernel/hostname
    if let Ok(name) = std::fs::read_to_string("/proc/sys/kernel/hostname") {
        let name = name.trim().to_string();
        if !name.is_empty() {
            return Ok(name);
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "hostname not available",
    ))
}

/// Initialize OpenTelemetry tracing, logs, and metrics.
/// Returns guards that must be held for the lifetime of the server.
pub fn init_tracing(config: &OtelConfig) -> OtelGuards {
    if !config.enabled {
        // Just initialize basic tracing.
        // Disable ANSI colors so the log file output is clean and parseable
        // by the TUI log tailer.
        let _ = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();
        return OtelGuards::empty();
    }

    eprintln!(
        "OpenTelemetry: enabled, endpoint={:?}, service={}, sample_ratio={}, logs={}, metrics={}",
        config.endpoint,
        config.service_name,
        config.sample_ratio,
        config.export_logs,
        config.export_metrics
    );

    let resource = build_resource(config);

    let sampler = if config.sample_ratio >= 1.0 {
        Sampler::AlwaysOn
    } else {
        Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(config.sample_ratio)))
    };

    // --- Traces ---
    let tracer_provider = if let Some(endpoint) = &config.endpoint {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .expect("Failed to create OTLP trace exporter");

        eprintln!("OpenTelemetry: trace exporter created for {}", endpoint);

        SdkTracerProvider::builder()
            .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
            .with_resource(resource.clone())
            .with_sampler(sampler)
            .build()
    } else {
        eprintln!("OpenTelemetry: no endpoint configured — traces will not be exported");
        SdkTracerProvider::builder()
            .with_resource(resource.clone())
            .with_sampler(sampler)
            .build()
    };

    let tracer = TracerProviderTrait::tracer(&tracer_provider, config.service_name.clone());
    let otel_layer = OpenTelemetryLayer::new(tracer);

    // Register the global tracer provider so that
    // opentelemetry::global::tracer() returns a real tracer (not noop).
    // This is used by trace_context_middleware to create server spans
    // that inherit the remote W3C trace context.
    opentelemetry::global::set_tracer_provider(tracer_provider.clone());

    // --- Logs ---
    let logger_provider = if config.export_logs {
        if let Some(endpoint) = &config.endpoint {
            let log_exporter = opentelemetry_otlp::LogExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint)
                .build()
                .expect("Failed to create OTLP log exporter");

            eprintln!("OpenTelemetry: log exporter created for {}", endpoint);

            let provider = LoggerProvider::builder()
                .with_batch_exporter(log_exporter, opentelemetry_sdk::runtime::Tokio)
                .with_resource(resource.clone())
                .build();

            Some(provider)
        } else {
            None
        }
    } else {
        eprintln!("OpenTelemetry: log export disabled by config");
        None
    };

    // --- Metrics ---
    let meter_provider = if config.export_metrics {
        if let Some(endpoint) = &config.endpoint {
            let metric_exporter = opentelemetry_otlp::MetricExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint)
                .build()
                .expect("Failed to create OTLP metric exporter");

            eprintln!("OpenTelemetry: metric exporter created for {}", endpoint);

            let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(
                metric_exporter,
                opentelemetry_sdk::runtime::Tokio,
            )
            .with_interval(std::time::Duration::from_secs(10))
            .build();

            let provider = SdkMeterProvider::builder()
                .with_reader(reader)
                .with_resource(resource.clone())
                .build();

            // Register a global meter provider so all code can access it
            opentelemetry::global::set_meter_provider(provider.clone());

            Some(provider)
        } else {
            None
        }
    } else {
        eprintln!("OpenTelemetry: metric export disabled by config");
        None
    };

    // --- Assemble subscriber ---
    // The OpenTelemetry tracing layer bridges spans.
    // The OpenTelemetry log appender bridges tracing events (info!, warn!, etc.)
    // to OTEL logs.
    let log_appender = logger_provider
        .as_ref()
        .map(opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new);

    // Register the W3C TraceContext propagator globally so that
    // trace_context_middleware can extract traceparent headers.
    opentelemetry::global::set_text_map_propagator(
        opentelemetry_sdk::propagation::TraceContextPropagator::new(),
    );

    let init_result = tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer().with_ansi(false))
        .with(otel_layer)
        .with(log_appender)
        .try_init();

    if let Err(e) = &init_result {
        eprintln!("OpenTelemetry: tracing subscriber init failed: {}", e);
    } else {
        eprintln!("OpenTelemetry: tracing subscriber initialized successfully");
    }

    OtelGuards {
        tracer_provider: Some(tracer_provider),
        logger_provider,
        meter_provider,
    }
}

/// Shut down all providers, flushing any pending data.
pub fn shutdown_tracing(guards: OtelGuards) {
    if let Some(provider) = guards.tracer_provider {
        if let Err(e) = provider.shutdown() {
            eprintln!("Failed to shutdown OpenTelemetry tracer provider: {:?}", e);
        }
    }
    if let Some(provider) = guards.logger_provider {
        if let Err(e) = provider.shutdown() {
            eprintln!("Failed to shutdown OpenTelemetry logger provider: {:?}", e);
        }
    }
    if let Some(provider) = guards.meter_provider {
        if let Err(e) = provider.shutdown() {
            eprintln!("Failed to shutdown OpenTelemetry meter provider: {:?}", e);
        }
    }
}
