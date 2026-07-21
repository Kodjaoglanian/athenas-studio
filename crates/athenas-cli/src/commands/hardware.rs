use athenas_core::{HardwareDetector, Result};

pub async fn show() -> Result<()> {
    let hw = HardwareDetector::detect()?;
    HardwareDetector::print_info(&hw);
    Ok(())
}
