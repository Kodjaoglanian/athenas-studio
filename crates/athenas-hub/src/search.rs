use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSearchResult {
    pub id: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub tags: Vec<String>,
    pub pipeline_tag: String,
    pub library_name: String,
    pub last_modified: String,
    pub gated: bool,
}

impl std::fmt::Display for ModelSearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  {} ⭐ {}  ⬇ {}", self.id, self.likes, format_downloads(self.downloads))?;
        if !self.pipeline_tag.is_empty() {
            writeln!(f, "    Pipeline: {}", self.pipeline_tag)?;
        }
        if !self.library_name.is_empty() {
            writeln!(f, "    Library:  {}", self.library_name)?;
        }
        if !self.tags.is_empty() {
            let tags: Vec<&str> = self.tags.iter().take(8).map(|s| s.as_str()).collect();
            writeln!(f, "    Tags:     {}", tags.join(", "))?;
        }
        if self.gated {
            writeln!(f, "    ⚠ Gated model (requires HF token)")?;
        }
        Ok(())
    }
}

fn format_downloads(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelSearchFilters {
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub gguf_only: bool,
    pub safetensors_only: bool,
}
