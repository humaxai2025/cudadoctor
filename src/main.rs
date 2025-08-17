use cuda_doctor::*;
use clap::Parser;

#[derive(Parser)]
#[command(name = "cuda-doctor")]
#[command(about = "A comprehensive diagnostic tool for NVIDIA GPU and AI framework installations")]
#[command(version)]
struct Args {
    /// Enable verbose output showing detailed command execution
    #[arg(short, long)]
    verbose: bool,
    
    /// Show installation and fix suggestions when components are not found
    #[arg(long)]
    showfix: bool,
    
    /// Show detailed system information and GPU specifications
    #[arg(long)]
    sysinfo: bool,
    
    /// Display version compatibility matrix for CUDA, frameworks, and drivers
    #[arg(long)]
    compatibility: bool,
    
    /// Run performance benchmarks for GPU and frameworks
    #[arg(long)]
    benchmark: bool,
    
    /// Check for available updates for drivers and frameworks
    #[arg(long)]
    check_updates: bool,
    
    /// Detect and analyze multiple GPUs in the system
    #[arg(long)]
    multi_gpu: bool,
    
    /// Export current environment configuration to JSON file
    #[arg(long, value_name = "FILE")]
    export: Option<String>,
    
    /// Import and validate environment configuration from JSON file
    #[arg(long, value_name = "FILE")]
    import: Option<String>,
    
    /// Validate system configuration and environment variables
    #[arg(long)]
    validate_config: bool,
}

fn main() {
    let args = Args::parse();
    let verbose = args.verbose;
    let showfix = args.showfix;
    
    // Handle special modes that don't require standard diagnostics
    if let Some(export_file) = &args.export {
        export_environment(export_file, verbose);
        return;
    }
    
    if let Some(import_file) = &args.import {
        import_environment(import_file, verbose);
        return;
    }
    
    if args.sysinfo {
        show_system_info(verbose);
        return;
    }
    
    if args.compatibility {
        show_compatibility_matrix();
        return;
    }
    
    if args.check_updates {
        check_for_updates(verbose);
        return;
    }
    
    if args.validate_config {
        validate_configuration(verbose);
        return;
    }
    
    println!("=== CUDA Doctor - GPU and AI Framework Diagnostics ===\n");
    
    // Check NVIDIA GPU(s)
    if args.multi_gpu {
        print!("🖥️  Checking Multiple GPUs...");
        match check_multiple_gpus(verbose) {
            Ok(gpu_info) => {
                if verbose {
                    println!("\n{}", gpu_info);
                } else {
                    println!(" ✅ Found");
                }
            },
            Err(_) => {
                println!(" ❌ Not found");
                if showfix {
                    println!("\n{}\n", suggest_nvidia_gpu_fix());
                }
            },
        }
    } else {
        print!("🖥️  Checking NVIDIA GPU...");
        match check_nvidia_gpu(verbose) {
            Ok(gpu_info) => {
                if verbose {
                    println!("\n   ✅ GPU: {}", gpu_info.trim());
                } else {
                    println!(" ✅ Found");
                }
            },
            Err(_) => {
                println!(" ❌ Not found");
                if showfix {
                    println!("\n{}\n", suggest_nvidia_gpu_fix());
                }
            },
        }
    }
    
    // Check NVIDIA Driver
    print!("🔧 Checking NVIDIA Driver...");
    match get_nvidia_driver_version(verbose) {
        Ok(driver_version) => {
            if verbose {
                println!("\n   ✅ Driver Version: {}", driver_version.trim());
            } else {
                println!(" ✅ Found");
            }
        },
        Err(_) => {
            println!(" ❌ Not found");
            if showfix {
                println!("\n{}\n", suggest_nvidia_driver_fix());
            }
        },
    }
    
    // Check CUDA Toolkit
    print!("⚙️  Checking CUDA Toolkit...");
    match get_cuda_toolkit_version(verbose) {
        Ok(cuda_version) => {
            if verbose {
                println!("\n   ✅ CUDA Version: {}", cuda_version.trim());
            } else {
                println!(" ✅ Found");
            }
        },
        Err(_) => {
            println!(" ❌ Not found");
            if showfix {
                println!("\n{}\n", suggest_cuda_toolkit_fix());
            }
        },
    }
    
    // Check cuDNN
    print!("🧠 Checking cuDNN...");
    match get_cudnn_version(verbose) {
        Ok(cudnn_version) => {
            if verbose {
                println!("\n   ✅ cuDNN Version: {}", cudnn_version.trim());
            } else {
                println!(" ✅ Found");
            }
        },
        Err(_) => {
            println!(" ❌ Not found");
            if showfix {
                println!("\n{}\n", suggest_cudnn_fix());
            }
        },
    }
    
    // Check TensorFlow
    print!("🔥 Checking TensorFlow...");
    match get_tensorflow_version(verbose) {
        Ok(tf_version) => {
            if verbose {
                println!("\n   ✅ TensorFlow Version: {}", tf_version.trim());
            } else {
                println!(" ✅ Found");
            }
        },
        Err(_) => {
            println!(" ❌ Not found");
            if showfix {
                println!("\n{}\n", suggest_tensorflow_fix());
            }
        },
    }
    
    // Check PyTorch
    print!("🚀 Checking PyTorch...");
    match get_pytorch_version(verbose) {
        Ok(pytorch_version) => {
            if verbose {
                println!("\n   ✅ PyTorch Version: {}", pytorch_version.trim());
            } else {
                println!(" ✅ Found");
            }
        },
        Err(_) => {
            println!(" ❌ Not found");
            if showfix {
                println!("\n{}\n", suggest_pytorch_fix());
            }
        },
    }
    
    // Run benchmarks if requested
    if args.benchmark {
        println!("\n🔬 Running Performance Benchmarks...");
        run_benchmarks(verbose);
    }
    
    println!("\n=== CUDA Doctor Diagnostics Complete ===");
    
    if !verbose && !showfix && !args.benchmark {
        println!("\n💡 Use --verbose or -v flag to see detailed version information and debugging output.");
        println!("💡 Use --showfix flag to see installation guides for missing components.");
        println!("💡 Use --benchmark flag to run performance tests.");
        println!("💡 Use --sysinfo flag to see detailed system information.");
    } else if !showfix && !args.benchmark {
        println!("\n💡 Use --showfix flag to see installation guides for missing components.");
        println!("💡 Use --benchmark flag to run performance tests.");
    } else if !args.benchmark {
        println!("\n💡 Use --benchmark flag to run performance tests.");
    }
}