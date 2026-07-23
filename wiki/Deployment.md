# Deployment

## Production Server Setup

### Basic Deployment

```bash
athenas serve model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --max-concurrent 20 \
  --rate-limit 50 \
  --timeout 300 \
  --max-body-size 50
```

### Systemd Service

Create `/etc/systemd/system/athenas.service`:

```ini
[Unit]
Description=Athenas Studio LLM Server
After=network.target

[Service]
Type=simple
User=athenas
Group=athenas
ExecStart=/home/athenas/.athenas/bin/athenas serve /path/to/model.gguf --host 0.0.0.0 --port 8080 --max-concurrent 20 --rate-limit 50
Restart=always
RestartSec=5
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable athenas
sudo systemctl start athenas
```

### Docker Deployment

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    curl libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install athenas
RUN curl -fsSL https://github.com/Kodjaoglanian/athenas-studio/releases/latest/download/install.sh | bash

# Copy model
COPY model.gguf /models/model.gguf

EXPOSE 8080

CMD ["athenas", "serve", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
docker build -t athenas-studio .
docker run -d --gpus all -p 8080:8080 athenas-studio
```

### Docker Compose

```yaml
version: '3.8'

services:
  athenas:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
      - ./config:/root/.athenas
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: athenas serve /models/model.gguf --host 0.0.0.0 --port 8080 --max-concurrent 20
    restart: unless-stopped
```

## Kubernetes Deployment

### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: athenas-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: athenas
  template:
    metadata:
      labels:
        app: athenas
    spec:
      containers:
      - name: athenas
        image: athenas-studio:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
        readinessProbe:
          httpGet:
            path: /v1/ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /v1/health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        volumeMounts:
        - name: models
          mountPath: /models
        - name: config
          mountPath: /root/.athenas
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: athenas-models
      - name: config
        persistentVolumeClaim:
          claimName: athenas-config
---
apiVersion: v1
kind: Service
metadata:
  name: athenas-service
spec:
  selector:
    app: athenas
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

### Health Checks

- **Readiness**: `GET /v1/ready` — Returns 200 when a model is loaded, 503 otherwise
- **Liveness**: `GET /v1/health` — Returns 200 when the server is running

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: athenas-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: athenas-server
  minReplicas: 1
  maxReplicas: 4
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Reverse Proxy (Nginx)

```nginx
upstream athenas {
    server 127.0.0.1:8080;
}

server {
    listen 443 ssl;
    server_name inference.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://athenas;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # Metrics (restrict access)
    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://athenas;
    }
}
```

## Monitoring

### Prometheus + Grafana

1. Athenas Studio exposes Prometheus metrics at `/metrics`
2. Configure Prometheus to scrape:

```yaml
scrape_configs:
  - job_name: 'athenas'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

3. Available metrics:
   - `athenas_requests_total` — Request counter by method/path/status
   - `athenas_requests_active` — Active request gauge
   - `athenas_request_duration_seconds` — Latency histogram
   - `athenas_tokens_prompt_total` — Prompt token counter
   - `athenas_tokens_generated_total` — Generated token counter
   - `athenas_errors_total` — Error counter by type

### Audit Log Monitoring

See [Audit Logging](Audit-Logging) for SIEM integration details.

## Security Checklist

- [ ] Set `api_key` in config or use [multi-tenant API keys](Multi-tenant-API-Keys)
- [ ] Bind to `0.0.0.0` only behind a reverse proxy or firewall
- [ ] Enable HTTPS via reverse proxy (Nginx, Caddy, Traefik)
- [ ] Set appropriate `rate_limit_per_second`
- [ ] Set `max_body_size_mb` to prevent DoS
- [ ] Set `request_timeout_secs` to prevent stuck requests
- [ ] Enable [audit logging](Audit-Logging)
- [ ] Restrict `/metrics` endpoint to internal networks
- [ ] Regularly rotate API keys
- [ ] Monitor token usage for anomalous patterns
