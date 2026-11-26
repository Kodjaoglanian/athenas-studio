use reqwest::Client;
use serde::{Deserialize, Deserializer, Serialize};
use std::time::Duration;

use athenas_core::{AthenasError, Result};

const HF_API_BASE: &str = "https://huggingface.co/api";

#[derive(Clone)]
pub struct HuggingFaceClient {
    client: Client,
    download_client: Client,
    token: Option<String>,
    base_url: String,
}

fn deserialize_gated<'de, D>(deserializer: D) -> std::result::Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    struct GatedVisitor;

    impl<'de> serde::de::Visitor<'de> for GatedVisitor {
        type Value = Option<bool>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a boolean, string (\"auto\"/\"manual\"), or null")
        }

        fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_bool<E>(self, v: bool) -> std::result::Result<Self::Value, E> {
            Ok(Some(v))
        }

        fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E> {
            let trimmed = v.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("false") {
                Ok(Some(false))
            } else {
                Ok(Some(true))
            }
        }
    }

    deserializer.deserialize_any(GatedVisitor)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelInfo {
    #[serde(rename = "id")]
    pub id: String,
    pub author: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Option<Vec<String>>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub last_modified: Option<String>,
    #[serde(default, deserialize_with = "deserialize_gated")]
    pub gated: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelFile {
    pub r#type: String,
    pub path: String,
    pub size: Option<u64>,
    pub lfs: Option<HfLfsInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfLfsInfo {
    pub size: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfTreeResponse {
    pub r#type: String,
    pub path: String,
    pub size: Option<u64>,
    pub children: Option<Vec<HfTreeResponse>>,
}

fn build_url(base: &str, segments: &[&str]) -> String {
    if let Ok(mut url) = url::Url::parse(base) {
        if let Ok(mut path_segs) = url.path_segments_mut() {
            path_segs.pop_if_empty();
            for seg in segments {
                for part in seg.split('/') {
                    if !part.is_empty() {
                        path_segs.push(part);
                    }
                }
            }
        }
        return url.to_string();
    }
    let mut out = base.trim_end_matches('/').to_string();
    for seg in segments {
        let trimmed = seg.trim_matches('/');
        if !trimmed.is_empty() {
            out.push('/');
            out.push_str(trimmed);
        }
    }
    out
}

impl HuggingFaceClient {
    pub fn new(token: Option<String>) -> Self {
        let token = token
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .or_else(|| {
                std::env::var("HF_TOKEN")
                    .ok()
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
            });

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(16)
            .pool_idle_timeout(Duration::from_secs(60))
            .tcp_nodelay(true)
            .user_agent("athenas-studio/0.1")
            .build()
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to build HF API client with custom settings: {}", e);
                Client::new()
            });

        // Dedicated download client: NO global total timeout so slow-internet
        // multi-GB model downloads don't get killed mid-transfer. Instead use
        // connect_timeout + per-read stall timeout + TCP keepalive.
        let download_client = Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(60))
            .tcp_keepalive(Duration::from_secs(30))
            .pool_idle_timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(16)
            .tcp_nodelay(true)
            .user_agent("athenas-studio/0.1")
            .build()
            .unwrap_or_else(|e| {
                tracing::warn!(
                    "Failed to build HF download client with custom settings: {}",
                    e
                );
                Client::new()
            });

        Self {
            client,
            download_client,
            token,
            base_url: HF_API_BASE.to_string(),
        }
    }

    pub fn with_mirror(mut self, mirror_url: String) -> Self {
        let trimmed = mirror_url.trim().trim_end_matches('/');
        if !trimmed.is_empty() {
            if trimmed.ends_with("/api") {
                self.base_url = trimmed.to_string();
            } else {
                self.base_url = format!("{}/api", trimmed);
            }
        }
        self
    }

    fn add_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref token) = self.token {
            req.bearer_auth(token)
        } else {
            req
        }
    }

    pub async fn search_models(
        &self,
        query: &str,
        filters: &super::search::ModelSearchFilters,
    ) -> Result<Vec<super::search::ModelSearchResult>> {
        let mut params = vec![("search", query.to_string())];

        if let Some(ref pipeline) = filters.pipeline_tag {
            params.push(("filter", pipeline.clone()));
        }
        if let Some(ref library) = filters.library_name {
            params.push(("filter", library.clone()));
        }
        if filters.gguf_only {
            params.push(("filter", "gguf".to_string()));
        }
        if filters.safetensors_only {
            params.push(("filter", "safetensors".to_string()));
        }

        params.push(("sort", "downloads".to_string()));
        params.push(("direction", "-1".to_string()));
        params.push(("limit", "50".to_string()));

        let url = build_url(&self.base_url, &["models"]);
        let req = self.client.get(&url).query(&params);
        let req = self.add_auth(req);

        let resp = req
            .send()
            .await
            .map_err(|e| AthenasError::HfApi(format!("Search request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::HfApi(format!(
                "HuggingFace API returned {}: {}",
                status, text
            )));
        }

        let models: Vec<HfModelInfo> = resp
            .json()
            .await
            .map_err(|e| AthenasError::HfApi(format!("Failed to parse search results: {}", e)))?;

        let results: Vec<super::search::ModelSearchResult> = models
            .iter()
            .map(|m| super::search::ModelSearchResult {
                id: m.id.clone(),
                author: m.author.clone().unwrap_or_default(),
                downloads: m.downloads.unwrap_or(0),
                likes: m.likes.unwrap_or(0),
                tags: m.tags.clone().unwrap_or_default(),
                pipeline_tag: m.pipeline_tag.clone().unwrap_or_default(),
                library_name: m.library_name.clone().unwrap_or_default(),
                last_modified: m.last_modified.clone().unwrap_or_default(),
                gated: m.gated.unwrap_or(false),
            })
            .collect();

        Ok(results)
    }

    pub async fn get_model_files(&self, repo_id: &str, revision: &str) -> Result<Vec<HfModelFile>> {
        let url = build_url(&self.base_url, &["models", repo_id, "tree", revision]);
        self.get_tree(&url).await
    }

    /// List ALL files in the repo recursively (nested dirs flattened into
    /// `path`). Needed for ONNX models whose weights/tokenizer/configs live
    /// in variant subdirectories (e.g. `onnx/`, `cpu-int4-rtn-block-32/`).
    pub async fn get_model_files_recursive(
        &self,
        repo_id: &str,
        revision: &str,
    ) -> Result<Vec<HfModelFile>> {
        let base = build_url(&self.base_url, &["models", repo_id, "tree", revision]);
        let url = format!("{}?recursive=true&expand=false", base);
        self.get_tree(&url).await
    }

    async fn get_tree(&self, url: &str) -> Result<Vec<HfModelFile>> {
        let req = self.client.get(url);
        let req = self.add_auth(req);

        let resp = req
            .send()
            .await
            .map_err(|e| AthenasError::HfApi(format!("Failed to get model files: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::HfApi(format!(
                "HuggingFace API returned {}: {}",
                status, text
            )));
        }

        let files: Vec<HfModelFile> = resp
            .json()
            .await
            .map_err(|e| AthenasError::HfApi(format!("Failed to parse model files: {}", e)))?;

        Ok(files)
    }

    pub async fn get_model_info(&self, repo_id: &str) -> Result<HfModelInfo> {
        let url = build_url(&self.base_url, &["models", repo_id]);
        let req = self.client.get(&url);
        let req = self.add_auth(req);

        let resp = req
            .send()
            .await
            .map_err(|e| AthenasError::HfApi(format!("Failed to get model info: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::HfApi(format!(
                "HuggingFace API returned {}: {}",
                status, text
            )));
        }

        let info: HfModelInfo = resp
            .json()
            .await
            .map_err(|e| AthenasError::HfApi(format!("Failed to parse model info: {}", e)))?;

        Ok(info)
    }

    pub fn download_url(&self, repo_id: &str, filename: &str, revision: &str) -> String {
        // Derive the resolve host from base_url so mirrors (e.g.
        // hf-mirror.com) apply to downloads too, not just the API.
        let base = self.base_url.trim_end_matches('/');
        let host = base.strip_suffix("/api").unwrap_or(base);
        build_url(host, &[repo_id, "resolve", revision, filename])
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn download_client(&self) -> &Client {
        &self.download_client
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_model_info_gated_deserializes_bool_and_strings() {
        let cases = [
            (r#"{"id":"org/a","gated":false}"#, Some(false)),
            (r#"{"id":"org/b","gated":true}"#, Some(true)),
            (r#"{"id":"org/c","gated":"auto"}"#, Some(true)),
            (r#"{"id":"org/d","gated":"manual"}"#, Some(true)),
            (r#"{"id":"org/e","gated":null}"#, None),
            (r#"{"id":"org/f"}"#, None),
        ];

        for (json, expected) in cases {
            let info: HfModelInfo = serde_json::from_str(json).expect(json);
            assert_eq!(info.gated, expected, "failed for {json}");
        }
    }

    #[test]
    fn with_mirror_normalizes_api_suffix_and_download_url() {
        let c1 = HuggingFaceClient::new(None).with_mirror("https://hf-mirror.com".to_string());
        assert_eq!(c1.base_url, "https://hf-mirror.com/api");
        assert_eq!(
            c1.download_url("org/my repo", "onnx/model #1.onnx", "main"),
            "https://hf-mirror.com/org/my%20repo/resolve/main/onnx/model%20%231.onnx"
        );

        let c2 = HuggingFaceClient::new(None).with_mirror("https://hf-mirror.com/api/".to_string());
        assert_eq!(c2.base_url, "https://hf-mirror.com/api");
        assert_eq!(
            c2.download_url("org/repo", "file.gguf", "main"),
            "https://hf-mirror.com/org/repo/resolve/main/file.gguf"
        );
    }

    #[test]
    fn empty_token_is_ignored() {
        let c = HuggingFaceClient::new(Some("   ".to_string()));
        assert_ne!(c.token(), Some(""));
        assert_ne!(c.token(), Some("   "));
    }
}
