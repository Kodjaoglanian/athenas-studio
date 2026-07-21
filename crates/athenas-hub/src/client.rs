use reqwest::Client;
use serde::{Deserialize, Serialize};

use athenas_core::{AthenasError, Result};

const HF_API_BASE: &str = "https://huggingface.co/api";

#[derive(Clone)]
pub struct HuggingFaceClient {
    client: Client,
    token: Option<String>,
    base_url: String,
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

impl HuggingFaceClient {
    pub fn new(token: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("athenas-studio/0.1")
            .build()
            .unwrap();

        Self {
            client,
            token,
            base_url: HF_API_BASE.to_string(),
        }
    }

    pub fn with_mirror(mut self, mirror_url: String) -> Self {
        self.base_url = mirror_url;
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

        let url = format!("{}/models", self.base_url);
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
        let url = format!("{}/models/{}/tree/{}", self.base_url, repo_id, revision);
        let req = self.client.get(&url);
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
        let url = format!("{}/models/{}", self.base_url, repo_id);
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
            )))?;
        }

        let info: HfModelInfo = resp
            .json()
            .await
            .map_err(|e| AthenasError::HfApi(format!("Failed to parse model info: {}", e)))?;

        Ok(info)
    }

    pub fn download_url(&self, repo_id: &str, filename: &str, revision: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            repo_id, revision, filename
        )
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }
}
