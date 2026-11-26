use athenas_core::ModelInfo;
use ratatui::widgets::ListState;

#[derive(Default)]
pub struct ModelListState {
    pub models: Vec<ModelInfo>,
    pub list_state: ListState,
    /// Live filter text (activated with `/`)
    pub filter: String,
    pub filtering: bool,
    /// Pending delete confirmation: (model_name, formatted_size)
    pub confirm_delete: Option<(String, String)>,
    /// Transient status banner on the Models tab
    pub status_message: Option<String>,
    pub status_is_error: bool,
}

impl ModelListState {
    pub fn set_models(&mut self, models: Vec<ModelInfo>) {
        self.models = models;
        self.clamp_selection();
    }

    pub fn filtered_models(&self) -> Vec<&ModelInfo> {
        let q = self.filter.trim().to_lowercase();
        if q.is_empty() {
            self.models.iter().collect()
        } else {
            self.models
                .iter()
                .filter(|m| {
                    m.name.to_lowercase().contains(&q)
                        || m.repo_id.to_lowercase().contains(&q)
                        || m.quantization
                            .as_deref()
                            .unwrap_or("")
                            .to_lowercase()
                            .contains(&q)
                        || m.architecture
                            .as_deref()
                            .unwrap_or("")
                            .to_lowercase()
                            .contains(&q)
                        || m.category
                            .as_deref()
                            .unwrap_or("")
                            .to_lowercase()
                            .contains(&q)
                })
                .collect()
        }
    }

    pub fn clamp_selection(&mut self) {
        let count = self.filtered_models().len();
        if count == 0 {
            self.list_state.select(None);
        } else {
            let sel = self.list_state.selected().unwrap_or(0).min(count - 1);
            self.list_state.select(Some(sel));
        }
    }

    pub fn next(&mut self) {
        let count = self.filtered_models().len();
        if count == 0 {
            return;
        }
        let i = self.list_state.selected().unwrap_or(0);
        let next = if i + 1 >= count { 0 } else { i + 1 };
        self.list_state.select(Some(next));
    }

    pub fn previous(&mut self) {
        let count = self.filtered_models().len();
        if count == 0 {
            return;
        }
        let i = self.list_state.selected().unwrap_or(0);
        let prev = if i == 0 { count - 1 } else { i - 1 };
        self.list_state.select(Some(prev));
    }

    pub fn selected(&self) -> Option<&ModelInfo> {
        let filtered = self.filtered_models();
        let idx = self.list_state.selected()?;
        filtered.get(idx).copied()
    }

    pub fn set_status(&mut self, msg: impl Into<String>) {
        self.status_message = Some(msg.into());
        self.status_is_error = false;
    }

    pub fn set_error(&mut self, msg: impl Into<String>) {
        self.status_message = Some(msg.into());
        self.status_is_error = true;
    }

    pub fn clear_status(&mut self) {
        self.status_message = None;
        self.status_is_error = false;
    }
}
