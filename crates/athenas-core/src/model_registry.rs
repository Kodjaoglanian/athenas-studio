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
    /// Model category: "llm" (text generation), "whisper" (audio transcription),
    /// "clip" (vision), or None (unknown).
    #[serde(default)]
    pub category: Option<String>,
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
            let filename_lower = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_lowercase();

            // Skip multimodal projector files — they are not standalone
            // models and will cause "unsupported model architecture: 'clip'"
            // errors if loaded directly. They should be loaded via --mmproj.
            let is_mmproj = filename_lower.contains("mmproj")
                || filename_lower.contains("mmproj-")
                || filename_lower.contains("-mmproj");

            if ext == "gguf" && !is_mmproj {
                // Read GGUF metadata (architecture, context length, license)
                let meta = read_gguf_metadata(&path);

                // Categorize model based on architecture
                let category = meta
                    .as_ref()
                    .and_then(|m| m.architecture.as_ref())
                    .map(|a| categorize_model(a));

                let mut model = create_model_info_from_file(&path, ModelFormat::Gguf)?;
                if let Some(meta) = meta {
                    model.architecture = meta.architecture;
                    model.context_length = meta.context_length;
                    model.license = meta.license;
                }
                model.category = category;
                models.push(model);
            } else if ext == "safetensors" && !is_mmproj {
                let model = create_model_info_from_file(&path, ModelFormat::Safetensors)?;
                models.push(model);
            }
        }
    }

    Ok(())
}

/// Categorize a model based on its architecture string.
/// Returns "llm" for text generation models, "whisper" for audio transcription,
/// "clip" for vision models, etc.
fn categorize_model(arch: &str) -> String {
    let arch_lower = arch.to_lowercase();
    match arch_lower.as_str() {
        "whisper" => "whisper".to_string(),
        "t5" | "speech-t5" => "tts".to_string(),
        "tts" | "vits" | "bark" | "clvp" => "tts".to_string(),
        "diffusion" => "diffusion".to_string(),
        // All other architectures are text LLMs (llama, mistral, qwen, etc.)
        _ => "llm".to_string(),
    }
}

/// Metadata extracted from a GGUF file header.
#[derive(Debug, Default)]
struct GgufMetadata {
    architecture: Option<String>,
    context_length: Option<u32>,
    license: Option<String>,
    name: Option<String>,
}

/// Read metadata fields from a GGUF file header.
/// Returns None if the file can't be parsed.
fn read_gguf_metadata(path: &PathBuf) -> Option<GgufMetadata> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    // Read first 64KB — metadata is at the start of the file
    // and 64KB is more than enough for the KV pairs.
    file.by_ref().take(65536).read_to_end(&mut buf).ok()?;

    let mut cursor = std::io::Cursor::new(&buf);

    // GGUF magic: "GGUF" = 0x46554747 (little-endian)
    let magic: u32 = read_u32(&mut cursor)?;
    if magic != 0x46554747 {
        return None;
    }

    let version: u32 = read_u32(&mut cursor)?;

    // In v1/v2, tensor_count and metadata_kv_count are u32.
    // In v3+, they are u64.
    let kv_count = if version >= 3 {
        let _tensor_count: u64 = read_u64(&mut cursor)?;
        read_u64(&mut cursor)? as usize
    } else {
        let _tensor_count: u32 = read_u32(&mut cursor)?;
        read_u32(&mut cursor)? as usize
    };
    parse_gguf_metadata(&mut cursor, kv_count)
}

fn parse_gguf_metadata(
    cursor: &mut std::io::Cursor<&Vec<u8>>,
    kv_count: usize,
) -> Option<GgufMetadata> {
    let mut meta = GgufMetadata::default();
    // Context length keys are per-architecture (e.g. "llama.context_length")
    // and may appear before general.architecture — collect them all.
    let mut ctx_lengths: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

    for _ in 0..kv_count {
        // Read key (gguf string: u64 length + bytes)
        let key = read_gguf_string(cursor)?;

        // Read value type (u32)
        let value_type: u32 = read_u32(cursor)?;

        match key.as_str() {
            "general.architecture" if value_type == 8 => {
                meta.architecture = read_gguf_string(cursor);
            }
            "general.license" if value_type == 8 => {
                meta.license = read_gguf_string(cursor);
            }
            "general.name" if value_type == 8 => {
                meta.name = read_gguf_string(cursor);
            }
            "general.context_length" if value_type == 4 => {
                meta.context_length = Some(read_u32(cursor)?);
            }
            _ if key.ends_with(".context_length") && value_type == 4 => {
                ctx_lengths.insert(key, read_u32(cursor)?);
            }
            _ => {
                // Skip the value based on its type
                skip_gguf_value(cursor, value_type)?;
            }
        }
    }

    // Prefer the architecture-specific context length
    if meta.context_length.is_none() {
        if let Some(ref arch) = meta.architecture {
            meta.context_length = ctx_lengths
                .get(&format!("{}.context_length", arch))
                .copied();
        }
        if meta.context_length.is_none() {
            meta.context_length = ctx_lengths.values().next().copied();
        }
    }

    Some(meta)
}

fn read_u32(cursor: &mut std::io::Cursor<&Vec<u8>>) -> Option<u32> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf).ok()?;
    Some(u32::from_le_bytes(buf))
}

fn read_u64(cursor: &mut std::io::Cursor<&Vec<u8>>) -> Option<u64> {
    use std::io::Read;
    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf).ok()?;
    Some(u64::from_le_bytes(buf))
}

fn read_gguf_string(cursor: &mut std::io::Cursor<&Vec<u8>>) -> Option<String> {
    // GGUF string: u64 length (in v3+, u32 in v1/v2) + UTF-8 bytes
    // We'll try u64 first (v3+), which is the common case
    let len: u64 = read_u64(cursor)?;
    if len > 1_000_000 {
        // Probably a v1/v2 file — retry with u32
        cursor.set_position(cursor.position() - 4);
        let len32: u32 = read_u32(cursor)?;
        if len32 > 1_000_000 {
            return None;
        }
        let mut buf = vec![0u8; len32 as usize];
        use std::io::Read;
        cursor.read_exact(&mut buf).ok()?;
        return String::from_utf8(buf).ok();
    }
    let mut buf = vec![0u8; len as usize];
    use std::io::Read;
    cursor.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn skip_gguf_value(cursor: &mut std::io::Cursor<&Vec<u8>>, value_type: u32) -> Option<()> {
    use std::io::Read;

    match value_type {
        0 => {
            cursor.read_exact(&mut [0u8; 1]).ok()?;
        } // UINT8
        1 => {
            cursor.read_exact(&mut [0u8; 1]).ok()?;
        } // INT8
        2 => {
            cursor.read_exact(&mut [0u8; 2]).ok()?;
        } // UINT16
        3 => {
            cursor.read_exact(&mut [0u8; 2]).ok()?;
        } // INT16
        4 => {
            cursor.read_exact(&mut [0u8; 4]).ok()?;
        } // UINT32
        5 => {
            cursor.read_exact(&mut [0u8; 4]).ok()?;
        } // INT32
        6 => {
            cursor.read_exact(&mut [0u8; 4]).ok()?;
        } // FLOAT32
        7 => {
            cursor.read_exact(&mut [0u8; 1]).ok()?;
        } // BOOL
        8 => {
            // STRING
            let _ = read_gguf_string(cursor)?;
        }
        9 => {
            // ARRAY
            let elem_type: u32 = read_u32(cursor)?;
            // In v3+, array length is u64; in v1/v2 it's u32
            // We'll try u64 and fall back
            let len: u64 = read_u64(cursor)?;
            if len > 10_000_000 {
                cursor.set_position(cursor.position() - 4);
                let len32: u32 = read_u32(cursor)?;
                if len32 > 10_000_000 {
                    return None;
                }
                for _ in 0..len32 {
                    skip_gguf_value(cursor, elem_type)?;
                }
            } else {
                for _ in 0..len {
                    skip_gguf_value(cursor, elem_type)?;
                }
            }
        }
        10 => {
            cursor.read_exact(&mut [0u8; 8]).ok()?;
        } // UINT64
        11 => {
            cursor.read_exact(&mut [0u8; 8]).ok()?;
        } // INT64
        12 => {
            cursor.read_exact(&mut [0u8; 8]).ok()?;
        } // FLOAT64
        _ => {
            // Unknown type — can't skip reliably
            return None;
        }
    }
    Some(())
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
        category: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_detection() {
        assert_eq!(
            detect_quantization("model-Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
        assert_eq!(
            detect_quantization("llama-3.2-3b-instruct-q8_0.gguf"),
            Some("Q8_0".to_string())
        );
        assert_eq!(
            detect_quantization("model-f16.gguf"),
            Some("F16".to_string())
        );
        assert_eq!(detect_quantization("model.gguf"), None);
    }

    #[test]
    fn categorize_whisper_vs_llm() {
        assert_eq!(categorize_model("whisper"), "whisper");
        assert_eq!(categorize_model("llama"), "llm");
        assert_eq!(categorize_model("qwen2"), "llm");
        assert_eq!(categorize_model("t5"), "tts");
    }

    /// Build a minimal GGUF v3 header with metadata KV pairs for testing.
    fn write_test_gguf(dir: &std::path::Path, name: &str, kvs: &[(&str, u32, &[u8])]) -> PathBuf {
        use std::io::Write;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF"); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes()); // kv_count
        for (key, vtype, val) in kvs {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&vtype.to_le_bytes());
            buf.extend_from_slice(val);
        }
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        path
    }

    fn gguf_string(s: &str) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
        v
    }

    #[test]
    fn gguf_metadata_parsing() {
        let dir = std::env::temp_dir().join(format!("athenas-gguf-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let kvs: Vec<(String, u32, Vec<u8>)> = vec![
            ("general.architecture".into(), 8, gguf_string("llama")),
            ("general.license".into(), 8, gguf_string("apache-2.0")),
            ("general.name".into(), 8, gguf_string("Test Model")),
            (
                "llama.context_length".into(),
                4,
                8192u32.to_le_bytes().to_vec(),
            ),
        ];
        let kv_refs: Vec<(&str, u32, &[u8])> = kvs
            .iter()
            .map(|(k, t, v)| (k.as_str(), *t, v.as_slice()))
            .collect();

        let path = write_test_gguf(&dir, "test.gguf", &kv_refs);
        let meta = read_gguf_metadata(&path).unwrap();
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.license.as_deref(), Some("apache-2.0"));
        assert_eq!(meta.name.as_deref(), Some("Test Model"));
        assert_eq!(meta.context_length, Some(8192));
    }

    #[test]
    fn gguf_bad_magic() {
        let dir = std::env::temp_dir().join(format!("athenas-gguf-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.gguf");
        std::fs::write(&path, b"NOTGGUF").unwrap();
        assert!(read_gguf_metadata(&path).is_none());
    }

    #[test]
    fn gguf_context_length_fallback_any_arch() {
        let dir = std::env::temp_dir().join(format!("athenas-gguf-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        // Only a ctx key whose arch doesn't match "general.architecture"
        let kvs: Vec<(String, u32, Vec<u8>)> = vec![
            ("general.architecture".into(), 8, gguf_string("mistral")),
            (
                "qwen2.context_length".into(),
                4,
                32768u32.to_le_bytes().to_vec(),
            ),
        ];
        let kv_refs: Vec<(&str, u32, &[u8])> = kvs
            .iter()
            .map(|(k, t, v)| (k.as_str(), *t, v.as_slice()))
            .collect();
        let path = write_test_gguf(&dir, "m.gguf", &kv_refs);
        let meta = read_gguf_metadata(&path).unwrap();
        // Falls back to any *.context_length when arch-specific key missing
        assert_eq!(meta.context_length, Some(32768));
    }
}
