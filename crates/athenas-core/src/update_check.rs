use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const UPDATE_CACHE_TTL_SECS: i64 = 24 * 60 * 60; // 24 hours
pub const GITHUB_REPO: &str = "Kodjaoglanian/athenas-studio";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    pub latest_version: String,
    pub release_url: String,
    #[serde(default)]
    pub release_notes: Option<String>,
    /// Unix timestamp (seconds) when the check was performed.
    pub checked_at: i64,
}

impl UpdateInfo {
    pub fn cache_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".athenas").join("update_check.json"))
    }

    pub fn load_cache() -> Option<Self> {
        let path = Self::cache_path()?;
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }

    pub fn save_cache(&self) {
        if let Some(path) = Self::cache_path() {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let tmp = path.with_extension("json.tmp");
                if std::fs::write(&tmp, json).is_ok() {
                    let _ = std::fs::rename(&tmp, &path);
                }
            }
        }
    }

    pub fn is_fresh(&self, now_unix: i64) -> bool {
        now_unix.saturating_sub(self.checked_at) < UPDATE_CACHE_TTL_SECS
    }

    pub fn is_newer_than(&self, current: &str) -> bool {
        is_newer_version(current, &self.latest_version)
    }
}

/// Parse a semver-ish string (`v0.10.1`, `0.10.1-beta.1`, `1.2`) into `(major, minor, patch)`.
fn parse_semver_tuple(v: &str) -> Option<(u64, u64, u64)> {
    let trimmed = v.trim().trim_start_matches('v').trim_start_matches('V');
    // Strip pre-release / build metadata
    let core = trimmed.split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let major: u64 = parts.next()?.parse().ok()?;
    let minor: u64 = parts.next().unwrap_or("0").parse().ok()?;
    let patch: u64 = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

/// Returns `true` strictly when `latest` is a newer semantic version than `current`.
pub fn is_newer_version(current: &str, latest: &str) -> bool {
    match (parse_semver_tuple(current), parse_semver_tuple(latest)) {
        (Some(c), Some(l)) => l > c,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_comparison_detects_newer_versions() {
        assert!(is_newer_version("0.9.34", "0.10.0"));
        assert!(is_newer_version("v0.10.0", "v0.10.1"));
        assert!(is_newer_version("0.10.0", "1.0.0"));
        assert!(!is_newer_version("0.10.0", "0.10.0"));
        assert!(!is_newer_version("0.10.1", "0.10.0"));
        assert!(!is_newer_version("0.10.0", "0.9.34"));
        assert!(!is_newer_version("0.10.0", "invalid"));
    }

    #[test]
    fn cache_freshness_respects_ttl() {
        let info = UpdateInfo {
            latest_version: "0.10.1".to_string(),
            release_url: "https://example.com".to_string(),
            release_notes: None,
            checked_at: 1_000_000,
        };
        assert!(info.is_fresh(1_000_000 + 3600));
        assert!(!info.is_fresh(1_000_000 + UPDATE_CACHE_TTL_SECS + 1));
    }
}
