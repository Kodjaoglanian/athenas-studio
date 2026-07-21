use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::errors::{AthenasError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    PyTorch,
    Mlx,
    Onnx,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelFormat::Gguf => write!(f, "GGUF"),
            ModelFormat::Safetensors => write!(f, "safetensors"),
            ModelFormat::PyTorch => write!(f, "PyTorch"),
            ModelFormat::Mlx => write!(f, "MLX"),
            ModelFormat::Onnx => write!(f, "ONNX"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub repo_id: String,
    pub name: String,
    pub format: ModelFormat,
    pub file_path: PathBuf,
    pub file_size_bytes: u64,
    pub quantization: Option<String>,
    pub context_length: Option<u32>,
    pub architecture: Option<String>,
    pub huggingface_url: Option<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
    pub downloaded_at: chrono::DateTime<chrono::Utc>,
    pub last_used_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl ModelInfo {
    pub fn file_size_gb(&self) -> f64 {
        self.file_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn file_size_mb(&self) -> f64 {
        self.file_size_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn format_size(&self) -> String {
        let gb = self.file_size_gb();
        if gb >= 1.0 {
            format!("{:.2} GB", gb)
        } else {
            format!("{:.0} MB", self.file_size_mb())
        }
    }
}

impl std::fmt::Display for ModelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Name:        {}", self.name)?;
        writeln!(f, "  Repo:        {}", self.repo_id)?;
        writeln!(f, "  Format:      {}", self.format)?;
        writeln!(f, "  Size:        {}", self.format_size())?;
        if let Some(q) = &self.quantization {
            writeln!(f, "  Quantization: {}", q)?;
        }
        if let Some(ctx) = self.context_length {
            writeln!(f, "  Context:     {}", ctx)?;
        }
        if let Some(arch) = &self.architecture {
            writeln!(f, "  Architecture: {}", arch)?;
        }
        if let Some(lic) = &self.license {
            writeln!(f, "  License:     {}", lic)?;
        }
        if !self.tags.is_empty() {
            writeln!(f, "  Tags:        {}", self.tags.join(", "))?;
        }
        writeln!(
            f,
            "  Downloaded:  {}",
            self.downloaded_at.format("%Y-%m-%d %H:%M")
        )?;
        if let Some(last) = self.last_used_at {
            writeln!(f, "  Last used:   {}", last.format("%Y-%m-%d %H:%M"))?;
        }
        Ok(())
    }
}

pub struct ModelRegistry {
    models_dir: PathBuf,
}

impl ModelRegistry {
    pub fn new(models_dir: PathBuf) -> Self {
        Self { models_dir }
    }

    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
    }

    pub fn list_local_models(&self) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();

        if !self.models_dir.exists() {
            return Ok(models);
        }

        scan_dir_for_models(&self.models_dir, &mut models)?;
        models.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(models)
    }

    pub fn find_model(&self, id_or_name: &str) -> Result<ModelInfo> {
        let models = self.list_local_models()?;
        models
            .iter()
            .find(|m| m.id == id_or_name || m.name == id_or_name || m.repo_id == id_or_name)
            .cloned()
            .ok_or_else(|| AthenasError::ModelNotFound(id_or_name.to_string()))
    }

    pub fn model_path(&self, repo_id: &str, filename: &str) -> PathBuf {
        let safe_repo = repo_id.replace('/', "__");
        self.models_dir.join(safe_repo).join(filename)
    }

    pub fn model_dir(&self, repo_id: &str) -> PathBuf {
        let safe_repo = repo_id.replace('/', "__");
        self.models_dir.join(safe_repo)
    }

    pub fn remove_model(&self, id_or_name: &str) -> Result<()> {
        let model = self.find_model(id_or_name)?;
        let dir = model.file_path.parent().unwrap_or(&self.models_dir);
        if dir != self.models_dir {
            std::fs::remove_dir_all(dir)?;
        } else {
            std::fs::remove_file(&model.file_path)?;
        }
        Ok(())
    }

    pub fn disk_usage(&self) -> Result<u64> {
        let mut total = 0u64;
        if self.models_dir.exists() {
            total = dir_size(&self.models_dir);
        }
        Ok(total)
    }
}

fn scan_dir_for_models(dir: &PathBuf, models: &mut Vec<ModelInfo>) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(AthenasError::Io)?;

    for entry in entries {
        let entry = entry.map_err(AthenasError::Io)?;
        let path = entry.path();

        if path.is_dir() {
            scan_dir_for_models(&path, models)?;
        } else if let Some(ext) = path.extension() {
            if ext == "gguf" {
                let model = create_model_info_from_file(&path, ModelFormat::Gguf)?;
                models.push(model);
            } else if ext == "safetensors" {
                let model = create_model_info_from_file(&path, ModelFormat::Safetensors)?;
                models.push(model);
            }
        }
    }

    Ok(())
}

fn create_model_info_from_file(path: &PathBuf, format: ModelFormat) -> Result<ModelInfo> {
    let metadata = std::fs::metadata(path)?;
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let repo_id = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(|s| s.replace("__", "/"))
        .unwrap_or_default();

    let quantization = detect_quantization(&filename);
    let id = format!("{}/{}", repo_id, filename);
    let hf_url = format!("https://huggingface.co/{}", repo_id);

    Ok(ModelInfo {
        id,
        repo_id,
        name: filename,
        format,
        file_path: path.clone(),
        file_size_bytes: metadata.len(),
        quantization,
        context_length: None,
        architecture: None,
        huggingface_url: Some(hf_url),
        license: None,
        tags: Vec::new(),
        downloaded_at: chrono::Utc::now(),
        last_used_at: None,
    })
}

fn detect_quantization(filename: &str) -> Option<String> {
    let lower = filename.to_lowercase();
    let quants = [
        "q8_0", "q7_0", "q6_0", "q5_1", "q5_0", "q4_1", "q4_0", "q4_k_m", "q4_k_s", "q3_k_m",
        "q3_k_s", "q3_k_l", "q2_k", "q1_0", "f16", "f32", "iq4_xs", "iq3_xs", "q8_0_k", "q6_k",
        "q5_k_m", "q5_k_s",
    ];

    for q in &quants {
        if lower.contains(q) {
            return Some(q.to_uppercase());
        }
    }
    None
}

fn dir_size(path: &PathBuf) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += dir_size(&path);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}
