use std::sync::Arc;
use std::time::Duration;

use athenas_core::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Information about a llama-server slot (from GET /slots).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotInfo {
    pub id: i32,
    pub n_ctx: u32,
    pub n_past: u32,
    pub n_tokens: u32,
    pub is_processing: bool,
    pub prompt: Option<String>,
    pub cache: Option<SlotCacheState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotCacheState {
    pub tokens: u32,
    pub used_tokens: u32,
    pub parent: Option<i32>,
}

/// Manages llama-server slots for KV cache persistence.
/// The llama-server exposes:
///   GET  /slots           — list all slots
///   POST /slots/{id}      — control a slot (save/restore/erase)
///   POST /slots/{id}/save — save slot state to a named checkpoint
///   POST /slots/{id}/restore — restore slot state from checkpoint
pub struct SlotManager {
    /// Base URL of the llama-server (e.g., http://127.0.0.1:PORT)
    server_url: String,
    /// HTTP client
    client: reqwest::Client,
    /// Total number of parallel slots
    num_slots: i32,
    /// Map of slot_id -> session_id assignment
    assignments: Arc<Mutex<std::collections::HashMap<i32, String>>>,
}

impl SlotManager {
    pub fn new(server_url: String, num_slots: i32) -> Self {
        Self {
            server_url,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            num_slots,
            assignments: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Query all slots from the llama-server.
    pub async fn list_slots(&self) -> Result<Vec<SlotInfo>> {
        let url = format!("{}/slots", self.server_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| athenas_core::AthenasError::Backend(format!("Slot query failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(athenas_core::AthenasError::Backend(format!(
                "Slot query returned {}",
                resp.status()
            )));
        }

        let slots: Vec<SlotInfo> = resp
            .json()
            .await
            .map_err(|e| athenas_core::AthenasError::Backend(format!("Invalid slot response: {}", e)))?;

        Ok(slots)
    }

    /// Save a slot's KV cache to a named checkpoint.
    /// This allows restoring the context later without reprocessing.
    pub async fn save_slot(&self, slot_id: i32, checkpoint_name: &str) -> Result<()> {
        let url = format!("{}/slots/{}/save", self.server_url, slot_id);
        let body = serde_json::json!({ "name": checkpoint_name });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| athenas_core::AthenasError::Backend(format!("Slot save failed: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            warn!("Slot save returned error: {}", text);
            return Err(athenas_core::AthenasError::Backend(format!(
                "Slot save failed: {}",
                text
            )));
        }

        debug!("Saved slot {} to checkpoint '{}'", slot_id, checkpoint_name);
        Ok(())
    }

    /// Restore a slot's KV cache from a named checkpoint.
    pub async fn restore_slot(&self, slot_id: i32, checkpoint_name: &str) -> Result<()> {
        let url = format!("{}/slots/{}/restore", self.server_url, slot_id);
        let body = serde_json::json!({ "name": checkpoint_name });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| athenas_core::AthenasError::Backend(format!("Slot restore failed: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            warn!("Slot restore returned error: {}", text);
            return Err(athenas_core::AthenasError::Backend(format!(
                "Slot restore failed: {}",
                text
            )));
        }

        debug!(
            "Restored slot {} from checkpoint '{}'",
            slot_id, checkpoint_name
        );
        Ok(())
    }

    /// Erase a slot's KV cache (free memory).
    pub async fn erase_slot(&self, slot_id: i32) -> Result<()> {
        let url = format!("{}/slots/{}", self.server_url, slot_id);
        let body = serde_json::json!({ "erase": true });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| athenas_core::AthenasError::Backend(format!("Slot erase failed: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(athenas_core::AthenasError::Backend(format!(
                "Slot erase failed: {}",
                text
            )));
        }

        info!("Erased slot {}", slot_id);
        Ok(())
    }

    /// Assign a session to a specific slot.
    pub async fn assign_slot(&self, slot_id: i32, session_id: &str) -> Result<()> {
        if slot_id < 0 || slot_id >= self.num_slots {
            return Err(athenas_core::AthenasError::Backend(format!(
                "Invalid slot id {} (max: {})",
                slot_id, self.num_slots
            )));
        }

        self.assignments
            .lock()
            .await
            .insert(slot_id, session_id.to_string());
        Ok(())
    }

    /// Get the session assigned to a slot.
    pub async fn get_slot_assignment(&self, slot_id: i32) -> Option<String> {
        self.assignments
            .lock()
            .await
            .get(&slot_id)
            .cloned()
    }

    /// Get the number of slots.
    pub fn num_slots(&self) -> i32 {
        self.num_slots
    }

    /// Get the server URL.
    pub fn server_url(&self) -> &str {
        &self.server_url
    }
}
