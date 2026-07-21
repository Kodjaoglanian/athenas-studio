use athenas_core::{HardwareDetector, Result};
use comfy_table::{Table, presets::UTF8_FULL};

pub async fn list() -> Result<()> {
    let hw = HardwareDetector::detect()?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Backend", "Available", "Details"]);

    // llama.cpp
    let llamacpp_available = hw.has_cuda || hw.has_rocm || hw.has_vulkan || hw.has_metal;
    let llamacpp_details = if hw.has_cuda {
        "CUDA (NVIDIA GPU)"
    } else if hw.has_rocm {
        "ROCm (AMD GPU)"
    } else if hw.has_vulkan {
        "Vulkan"
    } else if hw.has_metal {
        "Metal (Apple)"
    } else {
        "CPU only"
    };
    table.add_row(vec!["llama.cpp", if llamacpp_available { "✓" } else { "✓ (CPU)" }, llamacpp_details]);

    // vLLM
    let vllm_available = hw.has_cuda || hw.has_rocm;
    let vllm_details = if hw.has_cuda {
        "CUDA (NVIDIA GPU)"
    } else if hw.has_rocm {
        "ROCm (AMD GPU)"
    } else {
        "Requires CUDA or ROCm"
    };
    table.add_row(vec!["vLLM", if vllm_available { "✓" } else { "✗" }, vllm_details]);

    println!("{}", table);

    if !hw.gpus.is_empty() {
        println!("\nGPUs:");
        for gpu in &hw.gpus {
            println!("  {}", gpu);
        }
    }

    Ok(())
}

pub async fn benchmark(model: Option<String>) -> Result<()> {
    let hw = HardwareDetector::detect()?;

    println!("Backend Benchmark");
    println!("=================\n");

    println!("Hardware:");
    println!("  CPUs: {}", hw.cpus);
    println!("  Memory: {}MB / {}MB", hw.memory_available_mb, hw.memory_total_mb);
    println!("  CUDA: {}", if hw.has_cuda { "yes" } else { "no" });
    println!("  ROCm: {}", if hw.has_rocm { "yes" } else { "no" });
    println!("  Vulkan: {}", if hw.has_vulkan { "yes" } else { "no" });

    if let Some(m) = model {
        println!("\nBenchmarking with model: {}", m);
        println!("(Benchmark implementation requires a loaded model)");
    } else {
        println!("\nNo model specified. Use --model <path> to benchmark.");
    }

    Ok(())
}
