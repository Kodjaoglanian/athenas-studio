use athenas_core::ModelInfo;
use ratatui::widgets::ListState;

#[derive(Default)]
pub struct ModelListState {
    pub models: Vec<ModelInfo>,
    pub list_state: ListState,
}

impl ModelListState {
    pub fn set_models(&mut self, models: Vec<ModelInfo>) {
        self.models = models;
        if !self.models.is_empty() {
            self.list_state.select(Some(0));
        }
    }

    pub fn next(&mut self) {
        if self.models.is_empty() {
            return;
        }
        let i = self.list_state.selected().unwrap_or(0);
        let next = if i + 1 >= self.models.len() { 0 } else { i + 1 };
        self.list_state.select(Some(next));
    }

    pub fn previous(&mut self) {
        if self.models.is_empty() {
            return;
        }
        let i = self.list_state.selected().unwrap_or(0);
        let prev = if i == 0 { self.models.len() - 1 } else { i - 1 };
        self.list_state.select(Some(prev));
    }

    pub fn selected(&self) -> Option<&ModelInfo> {
        self.list_state.selected().and_then(|i| self.models.get(i))
    }
}
