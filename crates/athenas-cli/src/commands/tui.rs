use athenas_core::{AppConfig, HardwareDetector, Result};
use athenas_tui::TuiApp;

pub async fn run() -> Result<()> {
    let config = AppConfig::load()?;
    config.ensure_dirs()?;

    let hardware = HardwareDetector::detect()?;
    let mut app = TuiApp::new(config, hardware);
    app.run().await
}
