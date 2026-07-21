use athenas_hub::ModelSearchResult;

pub enum BrowserPhase {
    Search,
    Results,
    Downloading,
    SelectFile,
}

pub struct ModelBrowserState {
    pub phase: BrowserPhase,
    pub search_input: String,
    pub search_results: Vec<ModelSearchResult>,
    pub results_selected: usize,
    pub gguf_only: bool,
    pub status_message: Option<String>,
    pub download_progress: Option<(u64, u64, f64)>,
    pub download_filename: Option<String>,
    pub file_options: Vec<(String, Option<u64>)>,
    pub file_selected: usize,
}

impl Default for ModelBrowserState {
    fn default() -> Self {
        Self {
            phase: BrowserPhase::Search,
            search_input: String::new(),
            search_results: Vec::new(),
            results_selected: 0,
            gguf_only: true,
            status_message: None,
            download_progress: None,
            download_filename: None,
            file_options: Vec::new(),
            file_selected: 0,
        }
    }
}

impl ModelBrowserState {
    pub fn next_result(&mut self) {
        if !self.search_results.is_empty() {
            self.results_selected = (self.results_selected + 1) % self.search_results.len();
        }
    }

    pub fn prev_result(&mut self) {
        if !self.search_results.is_empty() {
            if self.results_selected == 0 {
                self.results_selected = self.search_results.len() - 1;
            } else {
                self.results_selected -= 1;
            }
        }
    }

    pub fn selected_result(&self) -> Option<&ModelSearchResult> {
        self.search_results.get(self.results_selected)
    }

    pub fn next_file(&mut self) {
        if !self.file_options.is_empty() {
            self.file_selected = (self.file_selected + 1) % self.file_options.len();
        }
    }

    pub fn prev_file(&mut self) {
        if !self.file_options.is_empty() {
            if self.file_selected == 0 {
                self.file_selected = self.file_options.len() - 1;
            } else {
                self.file_selected -= 1;
            }
        }
    }

    pub fn reset_search(&mut self) {
        self.phase = BrowserPhase::Search;
        self.search_input.clear();
        self.search_results.clear();
        self.results_selected = 0;
        self.status_message = None;
    }

    /// Go back to search editing phase, keeping the current query text.
    pub fn back_to_search_edit(&mut self) {
        self.phase = BrowserPhase::Search;
        self.status_message = None;
    }
}
