use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use athenas_inference::{ChatMessage, MessageContent, Role};
use tokio::sync::Mutex;

/// Maximum number of messages stored per session (configurable).
const DEFAULT_MAX_HISTORY: usize = 100;

/// Default TTL for inactive sessions (2 hours).
const DEFAULT_TTL: Duration = Duration::from_secs(7200);

/// A conversation session with full message history and slot assignment.
#[derive(Debug)]
pub struct Session {
    pub id: String,
    pub model_id: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub system_prompt: Option<String>,
    /// Assigned llama-server slot index (for KV cache persistence).
    pub slot_id: Option<i32>,
    /// Whether the slot's KV cache is warm (context was sent before).
    pub slot_cache_warm: bool,
    pub created_at: Instant,
    pub last_active: Instant,
    pub max_history: usize,
}

impl Session {
    pub fn new(id: String, max_history: usize) -> Self {
        let now = Instant::now();
        Self {
            id,
            model_id: None,
            messages: Vec::new(),
            system_prompt: None,
            slot_id: None,
            slot_cache_warm: false,
            created_at: now,
            last_active: now,
            max_history,
        }
    }

    /// Append a message to the session, trimming to max_history.
    pub fn append(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
        self.last_active = Instant::now();
        // Trim oldest messages (keep system prompt separate)
        if self.messages.len() > self.max_history {
            let excess = self.messages.len() - self.max_history;
            self.messages.drain(0..excess);
        }
    }

    /// Build the full message list for a chat request, prepending system prompt.
    pub fn build_messages(&self, new_messages: &[ChatMessage]) -> Vec<ChatMessage> {
        let mut all = Vec::new();

        // Prepend system prompt if set
        if let Some(ref sys) = self.system_prompt {
            all.push(ChatMessage {
                role: Role::System,
                content: MessageContent::Text(sys.clone()),
            });
        }

        // Add history
        all.extend(self.messages.iter().cloned());

        // Add new messages
        all.extend(new_messages.iter().cloned());

        all
    }

    /// Check if the session has expired.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.last_active.elapsed() > ttl
    }

    /// Mark the slot cache as cold (e.g., after model reload).
    pub fn invalidate_cache(&mut self) {
        self.slot_cache_warm = false;
    }
}

/// Manages conversation sessions with optional slot assignment for KV cache persistence.
pub struct SessionManager {
    sessions: HashMap<String, Session>,
    max_history: usize,
    ttl: Duration,
    /// Next available slot index (round-robin assignment).
    next_slot: i32,
    /// Total number of parallel slots available on the llama-server.
    max_slots: i32,
}

impl SessionManager {
    pub fn new(max_history: usize, max_slots: i32) -> Self {
        Self {
            sessions: HashMap::new(),
            max_history,
            ttl: DEFAULT_TTL,
            next_slot: 0,
            max_slots,
        }
    }

    /// Create a new session and assign it a slot.
    pub fn create(&mut self, id: Option<String>) -> String {
        let session_id = id.unwrap_or_else(|| {
            format!("sess_{}", uuid::Uuid::new_v4().to_string().replace("-", ""))
        });

        let mut session = Session::new(session_id.clone(), self.max_history);

        // Assign a slot (round-robin)
        if self.max_slots > 0 {
            session.slot_id = Some(self.next_slot);
            self.next_slot = (self.next_slot + 1) % self.max_slots;
        }

        self.sessions.insert(session_id.clone(), session);
        session_id
    }

    /// Get a session by ID.
    pub fn get(&self, session_id: &str) -> Option<&Session> {
        self.sessions.get(session_id)
    }

    /// Get a mutable session by ID.
    pub fn get_mut(&mut self, session_id: &str) -> Option<&mut Session> {
        self.sessions.get_mut(session_id)
    }

    /// Delete a session.
    pub fn remove(&mut self, session_id: &str) -> bool {
        self.sessions.remove(session_id).is_some()
    }

    /// List all active sessions.
    pub fn list(&self) -> Vec<SessionInfo> {
        self.sessions
            .iter()
            .map(|(id, s)| SessionInfo {
                id: id.clone(),
                model_id: s.model_id.clone(),
                message_count: s.messages.len(),
                slot_id: s.slot_id,
                slot_cache_warm: s.slot_cache_warm,
                created_at_secs: s.created_at.elapsed().as_secs(),
                last_active_secs: s.last_active.elapsed().as_secs(),
                system_prompt: s.system_prompt.clone(),
            })
            .collect()
    }

    /// Purge expired sessions.
    pub fn purge_expired(&mut self) -> usize {
        let ttl = self.ttl;
        let before = self.sessions.len();
        self.sessions.retain(|_, s| !s.is_expired(ttl));
        before - self.sessions.len()
    }

    /// Invalidate all slot caches (e.g., after model reload).
    pub fn invalidate_all_caches(&mut self) {
        for s in self.sessions.values_mut() {
            s.invalidate_cache();
        }
    }

    /// Count active sessions.
    pub fn count(&self) -> usize {
        self.sessions.len()
    }
}

/// Summary info for listing sessions.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub model_id: Option<String>,
    pub message_count: usize,
    pub slot_id: Option<i32>,
    pub slot_cache_warm: bool,
    pub created_at_secs: u64,
    pub last_active_secs: u64,
    pub system_prompt: Option<String>,
}

/// Shared, thread-safe wrapper.
pub type SharedSessionManager = Arc<Mutex<SessionManager>>;

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_HISTORY, 1)
    }
}
