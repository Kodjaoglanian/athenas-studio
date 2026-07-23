use athenas_core::{AppConfig, HardwareDetector, Result};
use athenas_tui::log_buffer::LogBuffer;
use athenas_tui::TuiApp;

pub async fn run_with_log_buffer(log_buffer: LogBuffer) -> Result<()> {
    let config = AppConfig::load()?;
    config.ensure_dirs()?;

    let hardware = HardwareDetector::detect()?;
    let mut app = TuiApp::with_log_buffer(config, hardware, log_buffer);
    app.run().await
}
