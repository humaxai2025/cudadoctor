use std::process::Command;
use walkdir::WalkDir;
use std::fs;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sysinfo::System;

use chrono::{DateTime, Utc};
use std::env;
use std::path::Path;

// Data structures for environment export/import
#[derive(Serialize, Deserialize, Debug)]
pub struct EnvironmentConfig {
    pub system_info: SystemInfo,
    pub cuda_info: CudaInfo,
    pub frameworks: FrameworkInfo,
    pub timestamp: DateTime<Utc>,
    pub hostname: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub total_memory_gb: f64,
    pub python_version: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CudaInfo {
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub cudnn_version: Option<String>,
    pub gpus: Vec<GpuInfo>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub memory_gb: Option<f64>,
    pub compute_capability: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FrameworkInfo {
    pub tensorflow: Option<String>,
    pub pytorch: Option<String>,
}

pub fn run_command(command: &str, verbose: bool) -> Result<String, String> {
    if verbose {
        println!("Running command: {}", command);
    }
    let output = if cfg!(target_os = "windows") {
        Command::new("cmd")
                .args(&["/C", command])
                .output()
    } else {
        Command::new("sh")
                .arg("-c")
                .arg(command)
                .output()
    };

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if verbose {
                println!("Command stdout: {}", stdout);
                println!("Command stderr: {}", stderr);
            }
            if output.status.success() {
                Ok(stdout)
            } else {
                Err(format!("Command failed with exit code {}: {}", output.status.code().unwrap_or(-1), stderr))
            }
        }
        Err(e) => Err(e.to_string()),
    }
}

// Helper function to get package version using pip show command
fn get_pip_package_version(package_name: &str, pip_cmd: &str, verbose: bool) -> Result<String, String> {
    let command = format!("{} show {}", pip_cmd, package_name);
    match run_command(&command, verbose) {
        Ok(output) => {
            let output = output.trim();
            if !output.is_empty() && !output.contains("WARNING: Package(s) not found") {
                // Extract version from pip show output
                if let Some(version_line) = output.lines().find(|line| line.contains("Version:")) {
                    let version = version_line.split(':').nth(1).unwrap_or("").trim();
                    if !version.is_empty() {
                        return Ok(version.to_string());
                    }
                }
            }
            Err(format!("Package {} not found with {}", package_name, pip_cmd))
        }
        Err(e) => Err(e),
    }
}

// Helper function to get package version using conda
fn get_conda_package_version(package_name: &str, verbose: bool) -> Result<String, String> {
    let command = format!("conda list {}", package_name);
    match run_command(&command, verbose) {
        Ok(output) => {
            let output = output.trim();
            if !output.is_empty() {
                // Extract version from conda list output
                if let Some(package_line) = output.lines().find(|line| line.starts_with(package_name)) {
                    let parts: Vec<&str> = package_line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return Ok(parts[1].to_string());
                    }
                }
            }
            Err(format!("Package {} not found with conda", package_name))
        }
        Err(e) => Err(e),
    }
}

pub fn get_tensorflow_version(verbose: bool) -> Result<String, String> {
    // Try direct Python import first (most reliable)
    let python_methods = vec![
        r#"python -c "import tensorflow as tf; print(tf.__version__)""#,
        r#"python3 -c "import tensorflow as tf; print(tf.__version__)""#,
    ];

    for method in python_methods {
        if verbose {
            println!("Trying method: {}", method);
        }
        match run_command(method, verbose) {
            Ok(output) => {
                let output = output.trim();
                if !output.is_empty() && !output.contains("not found") && !output.contains("No module") && !output.contains("WARNING") {
                    return Ok(output.to_string());
                }
            }
            Err(_) => continue,
        }
    }

    // Try pip package managers
    let pip_commands = vec!["pip", "pip3"];
    for pip_cmd in pip_commands {
        if verbose {
            println!("Trying pip method: {} show tensorflow", pip_cmd);
        }
        match get_pip_package_version("tensorflow", pip_cmd, verbose) {
            Ok(version) => return Ok(version),
            Err(_) => continue,
        }
    }

    // Try conda
    if verbose {
        println!("Trying conda method: conda list tensorflow");
    }
    match get_conda_package_version("tensorflow", verbose) {
        Ok(version) => return Ok(version),
        Err(_) => {},
    }
    
    Err("TensorFlow not found or not installed".to_string())
}

pub fn get_pytorch_version(verbose: bool) -> Result<String, String> {
    // Try direct Python import first (most reliable)
    let python_methods = vec![
        r#"python -c "import torch; print(torch.__version__)""#,
        r#"python3 -c "import torch; print(torch.__version__)""#,
    ];

    for method in python_methods {
        if verbose {
            println!("Trying method: {}", method);
        }
        match run_command(method, verbose) {
            Ok(output) => {
                let output = output.trim();
                if !output.is_empty() && !output.contains("not found") && !output.contains("No module") && !output.contains("WARNING") {
                    return Ok(output.to_string());
                }
            }
            Err(_) => continue,
        }
    }

    // Try pip package managers
    let pip_commands = vec!["pip", "pip3"];
    for pip_cmd in pip_commands {
        if verbose {
            println!("Trying pip method: {} show torch", pip_cmd);
        }
        match get_pip_package_version("torch", pip_cmd, verbose) {
            Ok(version) => return Ok(version),
            Err(_) => continue,
        }
    }

    // Try conda with different package names
    let conda_packages = vec!["pytorch", "torch"];
    for package in conda_packages {
        if verbose {
            println!("Trying conda method: conda list {}", package);
        }
        match get_conda_package_version(package, verbose) {
            Ok(version) => return Ok(version),
            Err(_) => continue,
        }
    }
    
    Err("PyTorch not found or not installed".to_string())
}

pub fn get_cudnn_version(verbose: bool) -> Result<String, String> {
    let mut search_paths = Vec::new();

    #[cfg(target_os = "windows")]
    {
        search_paths.push("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
        search_paths.push("C:\\Program Files\\NVIDIA Corporation\\NVSMI");
        search_paths.push("C:\\Windows\\System32");
        
        // Also search in PATH for cudnn_version.h
        if let Some(path_env) = std::env::var_os("PATH") {
            for path in std::env::split_paths(&path_env) {
                let cudnn_header_path = path.join("cudnn_version.h");
                if cudnn_header_path.exists() {
                    if verbose {
                        println!("Found cudnn_version.h in PATH at: {}", cudnn_header_path.display());
                    }
                    if let Ok(version) = extract_cudnn_version_from_header(&cudnn_header_path) {
                        return Ok(version);
                    }
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        search_paths.push("/usr/local/cuda");
        search_paths.push("/opt/cuda");
        search_paths.push("/usr/include");
        search_paths.push("/usr/local/include");
        
        // Also search in common include paths for cudnn_version.h
        if let Some(ld_library_path) = std::env::var_os("LD_LIBRARY_PATH") {
            for path in std::env::split_paths(&ld_library_path) {
                let cudnn_header_path = path.join("cudnn_version.h");
                if cudnn_header_path.exists() {
                    if verbose {
                        println!("Found cudnn_version.h in LD_LIBRARY_PATH at: {}", cudnn_header_path.display());
                    }
                    if let Ok(version) = extract_cudnn_version_from_header(&cudnn_header_path) {
                        return Ok(version);
                    }
                }
            }
        }
    }

    // Search in standard paths
    for base_path in search_paths {
        if verbose {
            println!("Searching for cudnn_version.h in: {}", base_path);
        }
        for entry in WalkDir::new(base_path)
            .into_iter()
            .filter_map(|e| e.ok()) {
            if entry.file_name().to_str() == Some("cudnn_version.h") {
                if verbose {
                    println!("Found cudnn_version.h at: {}", entry.path().display());
                }
                if let Ok(version) = extract_cudnn_version_from_header(entry.path()) {
                    return Ok(version);
                }
            }
        }
    }

    // Try alternative methods via Python libraries
    if let Ok(output) = run_command(r#"python -c "import torch; print(torch.backends.cudnn.version())""#, verbose) {
        let version_str = output.trim();
        if !version_str.is_empty() && version_str != "None" {
            return Ok(version_str.to_string());
        }
    }

    if let Ok(output) = run_command(r#"python3 -c "import torch; print(torch.backends.cudnn.version())""#, verbose) {
        let version_str = output.trim();
        if !version_str.is_empty() && version_str != "None" {
            return Ok(version_str.to_string());
        }
    }

    if let Ok(output) = run_command(r#"python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cudnn_version'])""#, verbose) {
        let version_str = output.trim();
        if !version_str.is_empty() && version_str != "None" {
            return Ok(version_str.to_string());
        }
    }

    if let Ok(output) = run_command(r#"python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cudnn_version'])""#, verbose) {
        let version_str = output.trim();
        if !version_str.is_empty() && version_str != "None" {
            return Ok(version_str.to_string());
        }
    }

    Err("cuDNN version not found".to_string())
}

fn extract_cudnn_version_from_header(header_path: &std::path::Path) -> Result<String, String> {
    let content = fs::read_to_string(header_path).map_err(|e| e.to_string())?;
    
    let major = content
        .lines()
        .find(|line| line.contains("#define CUDNN_MAJOR"))
        .and_then(|line| line.split_whitespace().last())
        .unwrap_or("");
    let minor = content
        .lines()
        .find(|line| line.contains("#define CUDNN_MINOR"))
        .and_then(|line| line.split_whitespace().last())
        .unwrap_or("");
    let patch = content
        .lines()
        .find(|line| line.contains("#define CUDNN_PATCHLEVEL"))
        .and_then(|line| line.split_whitespace().last())
        .unwrap_or("");
    
    if !major.is_empty() && !minor.is_empty() && !patch.is_empty() {
        Ok(format!("{}.{}.{}", major, minor, patch))
    } else {
        Err("Could not parse cuDNN version from header".to_string())
    }
}

pub fn get_cuda_toolkit_version(verbose: bool) -> Result<String, String> {
    // Try nvcc command first (most reliable)
    let nvcc_commands = vec![
        "nvcc --version",
        "/usr/local/cuda/bin/nvcc --version",
        "/opt/cuda/bin/nvcc --version",
    ];

    for command in nvcc_commands {
        if verbose {
            println!("Trying command: {}", command);
        }
        match run_command(command, verbose) {
            Ok(output) => {
                // Fix regex pattern - remove extra backslashes
                let re = Regex::new(r"release (\d+\.\d+)").unwrap();
                if let Some(captures) = re.captures(&output) {
                    return Ok(captures[1].to_string());
                }
                // Try alternative pattern
                let re2 = Regex::new(r"V(\d+\.\d+\.\d+)").unwrap();
                if let Some(captures) = re2.captures(&output) {
                    return Ok(captures[1].to_string());
                }
            },
            Err(_) => continue,
        }
    }

    // Try alternative methods
    #[cfg(target_os = "linux")]
    if let Ok(output) = run_command("cat /usr/local/cuda/version.txt", verbose) {
        let re = Regex::new(r"CUDA Version (\d+\.\d+)").unwrap();
        if let Some(captures) = re.captures(&output) {
            return Ok(captures[1].to_string());
        }
    }

    // Try Python-based detection with both python and python3
    let python_commands = vec![
        r#"python -c "import torch; print(torch.version.cuda)""#,
        r#"python3 -c "import torch; print(torch.version.cuda)""#,
    ];

    for command in python_commands {
        if let Ok(output) = run_command(command, verbose) {
            let version_str = output.trim();
            if !version_str.is_empty() && version_str != "None" {
                return Ok(version_str.to_string());
            }
        }
    }

    let tensorflow_commands = vec![
        r#"python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])""#,
        r#"python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])""#,
    ];

    for command in tensorflow_commands {
        if let Ok(output) = run_command(command, verbose) {
            let version_str = output.trim();
            if !version_str.is_empty() && version_str != "None" {
                return Ok(version_str.to_string());
            }
        }
    }

    // Search for nvcc in common paths
    let mut search_paths = Vec::new();

    #[cfg(target_os = "windows")]
    {
        search_paths.push("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
        // Search in PATH for nvcc
        if let Some(path_env) = std::env::var_os("PATH") {
            for path in std::env::split_paths(&path_env) {
                let nvcc_path = path.join("nvcc.exe");
                if nvcc_path.exists() {
                    let command = format!("\"{}\" --version", nvcc_path.display());
                    if verbose {
                        println!("Attempting to run from PATH: {}", command);
                    }
                    match run_command(&command, verbose) {
                        Ok(output) => {
                            let re = Regex::new(r"release (\d+\.\d+)").unwrap();
                            if let Some(captures) = re.captures(&output) {
                                return Ok(captures[1].to_string());
                            }
                        },
                        Err(_) => continue,
                    }
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        search_paths.push("/usr/local/cuda");
        search_paths.push("/opt/cuda");
        // Search in PATH for nvcc
        if let Some(path_env) = std::env::var_os("PATH") {
            for path in std::env::split_paths(&path_env) {
                let nvcc_path = path.join("nvcc");
                if nvcc_path.exists() {
                    let command = format!("{} --version", nvcc_path.display());
                    if verbose {
                        println!("Attempting to run from PATH: {}", command);
                    }
                    match run_command(&command, verbose) {
                        Ok(output) => {
                            let re = Regex::new(r"release (\d+\.\d+)").unwrap();
                            if let Some(captures) = re.captures(&output) {
                                return Ok(captures[1].to_string());
                            }
                        },
                        Err(_) => continue,
                    }
                }
            }
        }
    }

    // Search in file system
    for base_path in search_paths {
        if verbose {
            println!("Searching for nvcc in: {}", base_path);
        }
        for entry in WalkDir::new(base_path)
            .into_iter()
            .filter_map(|e| e.ok()) {
            if entry.file_name().to_str() == Some("nvcc") || entry.file_name().to_str() == Some("nvcc.exe") {
                let nvcc_path = entry.path();
                let command = format!("\"{}\" --version", nvcc_path.display());
                if verbose {
                    println!("Attempting to run: {}", command);
                }
                match run_command(&command, verbose) {
                    Ok(output) => {
                        let re = Regex::new(r"release (\d+\.\d+)").unwrap();
                        if let Some(captures) = re.captures(&output) {
                            return Ok(captures[1].to_string());
                        }
                    },
                    Err(_) => continue,
                }
            }
        }
    }
    
    Err("CUDA Toolkit not found".to_string())
}

#[cfg(target_os = "windows")]
pub fn check_nvidia_gpu(verbose: bool) -> Result<String, String> {
    // Use Windows Management Instrumentation Command-line (WMIC)
    match run_command("wmic path win32_videocontroller get name", verbose) {
        Ok(output) => {
            // Filter for NVIDIA GPUs
            let nvidia_gpus: Vec<&str> = output
                .lines()
                .filter(|line| line.to_lowercase().contains("nvidia") && !line.trim().is_empty() && *line != "Name")
                .collect();
            
            if nvidia_gpus.is_empty() {
                Err("No NVIDIA GPUs found".to_string())
            } else {
                Ok(nvidia_gpus.join(", "))
            }
        }
        Err(e) => Err(e),
    }
}

#[cfg(target_os = "linux")]
pub fn check_nvidia_gpu(verbose: bool) -> Result<String, String> {
    // Use lspci command
    match run_command("lspci", verbose) {
        Ok(output) => {
            // Filter for NVIDIA GPUs
            let nvidia_gpus: Vec<&str> = output
                .lines()
                .filter(|line| line.to_lowercase().contains("nvidia"))
                .collect();
            
            if nvidia_gpus.is_empty() {
                Err("No NVIDIA GPUs found".to_string())
            } else {
                Ok(nvidia_gpus.join(", "))
            }
        }
        Err(_) => {
            // Fallback: try alternative methods
            if let Ok(output) = run_command("nvidia-smi -L", verbose) {
                Ok(output.trim().to_string())
            } else {
                Err("No NVIDIA GPUs found or lspci/nvidia-smi not available".to_string())
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub fn check_nvidia_gpu(verbose: bool) -> Result<String, String> {
    // On modern macOS, NVIDIA GPUs are not supported, but let's check anyway
    match run_command("system_profiler SPDisplaysDataType", verbose) {
        Ok(output) => {
            let nvidia_gpus: Vec<&str> = output
                .lines()
                .filter(|line| line.to_lowercase().contains("nvidia"))
                .collect();
            
            if nvidia_gpus.is_empty() {
                Err("No NVIDIA GPUs found (Note: Modern macOS doesn't support NVIDIA GPUs)".to_string())
            } else {
                Ok(nvidia_gpus.join(", "))
            }
        }
        Err(e) => Err(e),
    }
}

pub fn run_benchmark(verbose: bool) -> Result<String, String> {
    run_command("python benchmark.py", verbose)
}

pub fn get_nvidia_driver_version(verbose: bool) -> Result<String, String> {
    run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader", verbose)
}

// Fix suggestion functions for when components are not found

pub fn suggest_nvidia_gpu_fix() -> String {
    if cfg!(target_os = "windows") {
        format!(r#"💡 NVIDIA GPU Not Found - Possible Fixes:

🔧 Hardware Issues:
   • Ensure NVIDIA GPU is properly seated in PCIe slot
   • Check power connections to GPU (6-pin/8-pin connectors)
   • Verify PSU has sufficient wattage for your GPU

🖥️ Windows Device Manager:
   • Open Device Manager → Display adapters
   • Look for NVIDIA GPU (may show as "Unknown device")
   • Right-click → Update driver

📦 Driver Installation:
   • Download latest drivers from: https://www.nvidia.com/Download/index.aspx
   • Use GeForce Experience for automatic updates
   • Try DDU (Display Driver Uninstaller) if having issues

🛠️ BIOS Settings:
   • Enable PCIe slots in BIOS
   • Set primary display adapter to PCIe (not onboard)
   • Disable integrated graphics if needed"#)
    } else if cfg!(target_os = "linux") {
        format!(r#"💡 NVIDIA GPU Not Found - Possible Fixes:

🔧 Hardware Issues:
   • Ensure NVIDIA GPU is properly seated in PCIe slot
   • Check power connections to GPU
   • Verify PSU has sufficient wattage

📦 Driver Installation (Ubuntu/Debian):
   • sudo apt update
   • sudo apt install nvidia-driver-535 (or latest version)
   • sudo reboot

📦 Driver Installation (Fedora/RHEL):
   • sudo dnf install akmod-nvidia
   • sudo akmods --force
   • sudo reboot

🔍 Check Hardware Detection:
   • lspci | grep -i nvidia
   • sudo lshw -c display

🛠️ Alternative Installation:
   • Download from: https://www.nvidia.com/Download/index.aspx
   • Install proprietery drivers via distribution's driver manager"#)
    } else {
        format!(r#"💡 NVIDIA GPU Not Found - Possible Fixes:

⚠️  macOS Note:
   • Modern macOS (10.14+) doesn't support NVIDIA GPUs
   • Only older macOS versions with legacy drivers supported

🔧 For Older macOS:
   • Check: https://www.nvidia.com/Download/index.aspx
   • Look for legacy macOS drivers (if available)

🔍 Hardware Check:
   • System Information → Graphics/Displays
   • Terminal: system_profiler SPDisplaysDataType"#)
    }
}

pub fn suggest_nvidia_driver_fix() -> String {
    if cfg!(target_os = "windows") {
        format!(r#"💡 NVIDIA Driver Not Found - Installation Guide:

📥 Download Options:
   • Official: https://www.nvidia.com/Download/index.aspx
   • GeForce Experience: Automatic driver updates
   • Windows Update: Basic drivers (may be outdated)

🛠️ Installation Steps:
   1. Download appropriate driver for your GPU model
   2. Close all applications
   3. Run installer as Administrator
   4. Choose "Custom (Advanced)" for clean install
   5. Restart computer

🔧 Troubleshooting:
   • Use DDU to completely remove old drivers first
   • Disable Windows automatic driver updates
   • Try NVIDIA Studio drivers for content creation"#)
    } else {
        format!(r#"💡 NVIDIA Driver Not Found - Installation Guide:

📦 Ubuntu/Debian:
   sudo apt update
   sudo apt install nvidia-driver-535
   sudo reboot

📦 Fedora/RHEL:
   sudo dnf install akmod-nvidia
   sudo akmods --force
   sudo reboot

📦 Arch Linux:
   sudo pacman -S nvidia nvidia-utils
   sudo reboot

🔍 Check Installation:
   • nvidia-smi (should show driver version)
   • cat /proc/driver/nvidia/version

🛠️ Alternative Methods:
   • Use distribution's driver manager GUI
   • Download from: https://www.nvidia.com/Download/index.aspx"#)
    }
}

pub fn suggest_cuda_toolkit_fix() -> String {
    if cfg!(target_os = "windows") {
        format!(r#"💡 CUDA Toolkit Not Found - Installation Guide:

📥 Download CUDA Toolkit:
   • Official: https://developer.nvidia.com/cuda-downloads
   • Choose Windows x86_64 → Version → Installer type

🛠️ Installation Steps:
   1. Download CUDA Toolkit installer
   2. Run as Administrator
   3. Choose "Custom" installation
   4. Select CUDA Toolkit components
   5. Add to PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin

📝 Environment Variables:
   • CUDA_PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X
   • PATH: Add CUDA bin directory

✅ Verify Installation:
   • Open Command Prompt: nvcc --version
   • Should show CUDA compiler version"#)
    } else {
        format!(r#"💡 CUDA Toolkit Not Found - Installation Guide:

📥 Download Options:
   • Official: https://developer.nvidia.com/cuda-downloads
   • Package manager installation available

📦 Ubuntu/Debian Installation:
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt update
   sudo apt install cuda-toolkit

📦 Alternative Installation:
   • Download .run file from NVIDIA
   • chmod +x cuda_X.X.X_linux.run
   • sudo ./cuda_X.X.X_linux.run

📝 Environment Setup:
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc

✅ Verify: nvcc --version"#)
    }
}

pub fn suggest_cudnn_fix() -> String {
    format!(r#"💡 cuDNN Not Found - Installation Guide:

📥 Download cuDNN:
   • Official: https://developer.nvidia.com/cudnn
   • Requires free NVIDIA Developer account
   • Choose version compatible with your CUDA version

🛠️ Windows Installation:
   1. Download cuDNN ZIP archive
   2. Extract to temporary folder
   3. Copy files to CUDA installation directory:
      • bin → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin
      • include → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\include  
      • lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\lib

🛠️ Linux Installation:
   1. Download cuDNN tar file
   2. Extract: tar -xzvf cudnn-X.X-linux-x64-vX.X.X.tgz
   3. Copy files:
      sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
      sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
      sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

📦 Alternative (Conda):
   conda install cudnn

✅ Verify Installation:
   • Check: /usr/local/cuda/include/cudnn_version.h (Linux)
   • Python: import torch; print(torch.backends.cudnn.version())"#)
}

pub fn suggest_tensorflow_fix() -> String {
    format!(r#"💡 TensorFlow Not Found - Installation Guide:

📦 CPU Version (Recommended for beginners):
   pip install tensorflow

📦 GPU Version (Requires CUDA + cuDNN):
   pip install tensorflow[and-cuda]

🐍 Python Environment Setup:
   # Create virtual environment (recommended)
   python -m venv tensorflow_env
   # Windows: tensorflow_env\Scripts\activate
   # Linux/Mac: source tensorflow_env/bin/activate
   pip install --upgrade pip
   pip install tensorflow

📦 Conda Installation:
   conda install tensorflow
   # or for GPU support:
   conda install tensorflow-gpu

🔧 GPU Requirements:
   • NVIDIA GPU with CUDA Compute Capability 3.5+
   • CUDA 11.2+ and cuDNN 8.1+
   • Compatible NVIDIA drivers

✅ Verify Installation:
   python -c "import tensorflow as tf; print(tf.__version__)"
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

📚 Official Guide: https://www.tensorflow.org/install"#)
}

pub fn suggest_pytorch_fix() -> String {
    format!(r#"💡 PyTorch Not Found - Installation Guide:

📦 Quick Installation:
   pip install torch torchvision torchaudio

🔧 Custom Installation (Recommended):
   Visit: https://pytorch.org/get-started/locally/
   Select your preferences:
   • OS: Windows/Linux/Mac
   • Package: Pip/Conda
   • Python version
   • CUDA version (if using GPU)

📦 CPU Only:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

📦 GPU (CUDA 11.8):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

📦 GPU (CUDA 12.1):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

🐍 Conda Installation:
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

✅ Verify Installation:
   python -c "import torch; print(torch.__version__)"
   python -c "import torch; print(torch.cuda.is_available())"

📚 Official Guide: https://pytorch.org/get-started/locally/"#)
}

// ===== NEW ADVANCED FEATURES =====

// System Information Feature
pub fn show_system_info(verbose: bool) {
    println!("=== 🖥️  Detailed System Information ===\n");
    
    let mut system = System::new_all();
    system.refresh_all();
    
    // Basic system info
    println!("🔧 System Details:");
    println!("   OS: {} {}", System::name().unwrap_or_default(), System::os_version().unwrap_or_default());
    println!("   Kernel: {}", System::kernel_version().unwrap_or_default());
    println!("   Architecture: {}", env::consts::ARCH);
    println!("   Hostname: {}", System::host_name().unwrap_or_default());
    
    // CPU information
    println!("\n💻 CPU Information:");
    if let Some(cpu) = system.cpus().first() {
        println!("   Model: {}", cpu.brand());
        println!("   Cores: {} physical, {} logical", system.physical_core_count().unwrap_or(0), system.cpus().len());
        println!("   Frequency: {:.2} GHz", cpu.frequency() as f64 / 1000.0);
    }
    
    // Memory information
    println!("\n🧠 Memory Information:");
    let total_memory = system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let available_memory = system.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let used_memory = system.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    println!("   Total RAM: {:.1} GB", total_memory);
    println!("   Available: {:.1} GB", available_memory);
    println!("   Used: {:.1} GB ({:.1}%)", used_memory, (used_memory / total_memory) * 100.0);
    
    // GPU information with compute capability
    println!("\n🎮 GPU Information:");
    match get_detailed_gpu_info(verbose) {
        Ok(gpu_info) => println!("{}", gpu_info),
        Err(e) => println!("   ❌ {}", e),
    }
    
    // Python environment
    println!("\n🐍 Python Environment:");
    match get_python_info(verbose) {
        Ok(python_info) => println!("{}", python_info),
        Err(e) => println!("   ❌ {}", e),
    }
    
    // Environment variables
    println!("\n📝 Important Environment Variables:");
    check_environment_variables();
}

fn get_detailed_gpu_info(verbose: bool) -> Result<String, String> {
    let mut result = String::new();
    
    // Try nvidia-smi for detailed info
    match run_command("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits", verbose) {
        Ok(output) => {
            for (i, line) in output.lines().enumerate() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    result.push_str(&format!("   GPU {}: {}\n", i, parts[0]));
                    result.push_str(&format!("   Memory: {} MB\n", parts[1]));
                    result.push_str(&format!("   Compute Capability: {}\n", parts[2]));
                    if i < output.lines().count() - 1 {
                        result.push_str("\n");
                    }
                }
            }
        },
        Err(_) => {
            // Fallback to basic detection
            match check_nvidia_gpu(verbose) {
                Ok(gpu_basic) => result.push_str(&format!("   {}\n   ⚠️  Use nvidia-smi for detailed specs", gpu_basic.trim())),
                Err(e) => return Err(e),
            }
        }
    }
    
    Ok(result)
}

fn get_python_info(_verbose: bool) -> Result<String, String> {
    let mut result = String::new();
    
    // Check Python version
    if let Ok(output) = run_command("python --version", false) {
        result.push_str(&format!("   Python: {}\n", output.trim()));
    } else if let Ok(output) = run_command("python3 --version", false) {
        result.push_str(&format!("   Python: {}\n", output.trim()));
    } else {
        result.push_str("   Python: Not found\n");
    }
    
    // Check pip version
    if let Ok(output) = run_command("pip --version", false) {
        let version = output.split_whitespace().nth(1).unwrap_or("unknown");
        result.push_str(&format!("   Pip: {}\n", version));
    }
    
    // Check virtual environment
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        result.push_str(&format!("   Virtual Env: {}\n", venv));
    } else {
        result.push_str("   Virtual Env: None\n");
    }
    
    Ok(result)
}

fn check_environment_variables() {
    let important_vars = vec![
        "CUDA_PATH", "CUDA_HOME", "PATH", "LD_LIBRARY_PATH", 
        "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"
    ];
    
    for var in important_vars {
        if let Ok(value) = env::var(var) {
            let display_value = if value.len() > 60 {
                format!("{}...", &value[..57])
            } else {
                value
            };
            println!("   {}: {}", var, display_value);
        } else {
            println!("   {}: Not set", var);
        }
    }
}

// Compatibility Matrix Feature
pub fn show_compatibility_matrix() {
    println!("=== 🔗 Version Compatibility Matrix ===\n");
    
    println!("📊 CUDA ↔ Driver Compatibility:");
    println!("   CUDA 12.3+  → Driver 545.23+");
    println!("   CUDA 12.2   → Driver 535.86+");
    println!("   CUDA 12.1   → Driver 530.30+");
    println!("   CUDA 12.0   → Driver 525.60+");
    println!("   CUDA 11.8   → Driver 520.61+");
    println!("   CUDA 11.7   → Driver 515.43+");
    println!("   CUDA 11.6   → Driver 510.47+");
    
    println!("\n🔥 TensorFlow ↔ CUDA Compatibility:");
    println!("   TensorFlow 2.15+ → CUDA 12.3, cuDNN 8.9");
    println!("   TensorFlow 2.14  → CUDA 12.2, cuDNN 8.9");
    println!("   TensorFlow 2.13  → CUDA 11.8, cuDNN 8.6");
    println!("   TensorFlow 2.12  → CUDA 11.8, cuDNN 8.6");
    println!("   TensorFlow 2.11  → CUDA 11.2, cuDNN 8.1");
    println!("   TensorFlow 2.10  → CUDA 11.2, cuDNN 8.1");
    
    println!("\n🚀 PyTorch ↔ CUDA Compatibility:");
    println!("   PyTorch 2.1+   → CUDA 11.8, 12.1");
    println!("   PyTorch 2.0    → CUDA 11.7, 11.8");
    println!("   PyTorch 1.13   → CUDA 11.6, 11.7");
    println!("   PyTorch 1.12   → CUDA 11.3, 11.6");
    println!("   PyTorch 1.11   → CUDA 11.1, 11.3");
    
    println!("\n🐍 Python Version Requirements:");
    println!("   TensorFlow 2.15+ → Python 3.9-3.12");
    println!("   TensorFlow 2.11+ → Python 3.7-3.11");
    println!("   PyTorch 2.1+     → Python 3.8-3.11");
    println!("   PyTorch 1.13+    → Python 3.7-3.11");
    
    println!("\n💡 Compute Capability Requirements:");
    println!("   TensorFlow 2.11+ → CC 3.5+");
    println!("   PyTorch 1.13+    → CC 3.7+");
    println!("   CUDA 12.0+       → CC 5.0+ (optimal)");
    
    println!("\n📝 Recommended Combinations:");
    println!("   🔥 Latest Stable: CUDA 12.2 + cuDNN 8.9 + TensorFlow 2.14 + PyTorch 2.1");
    println!("   ⚡ High Performance: CUDA 11.8 + cuDNN 8.6 + TensorFlow 2.13 + PyTorch 2.0");
    println!("   🛡️  Long Term Support: CUDA 11.2 + cuDNN 8.1 + TensorFlow 2.10 + PyTorch 1.13");
}

// Multiple GPU Feature
pub fn check_multiple_gpus(verbose: bool) -> Result<String, String> {
    let mut result = String::new();
    
    // Get detailed multi-GPU information
    match run_command("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits", verbose) {
        Ok(output) => {
            result.push_str("   📊 Multi-GPU Status:\n\n");
            
            for line in output.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 10 {
                    result.push_str(&format!("   🎮 GPU {}: {}\n", parts[0], parts[1]));
                    result.push_str(&format!("      Memory: {}MB used / {}MB total ({}MB free)\n", 
                        parts[3], parts[2], parts[4]));
                    result.push_str(&format!("      Utilization: {}% GPU, {}% Memory\n", 
                        parts[5], parts[6]));
                    result.push_str(&format!("      Temperature: {}°C\n", parts[7]));
                    result.push_str(&format!("      Power: {}W / {}W\n\n", parts[8], parts[9]));
                }
            }
            
            // Check for SLI/NVLink
            if let Ok(topo_output) = run_command("nvidia-smi topo -m", verbose) {
                result.push_str("   🔗 GPU Topology:\n");
                for line in topo_output.lines().take(10) {
                    if !line.trim().is_empty() {
                        result.push_str(&format!("      {}\n", line));
                    }
                }
            }
        },
        Err(_) => {
            // Fallback to basic detection
            match check_nvidia_gpu(verbose) {
                Ok(gpu_info) => {
                    result.push_str("   ✅ Found NVIDIA GPU(s):\n");
                    result.push_str(&format!("      {}\n", gpu_info.trim()));
                    result.push_str("      ⚠️  Install nvidia-smi for detailed multi-GPU analysis\n");
                },
                Err(e) => return Err(e),
            }
        }
    }
    
    Ok(result)
}

// Performance Benchmark Feature
pub fn run_benchmarks(verbose: bool) {
    println!("⚡ GPU Memory Test...");
    test_gpu_memory(verbose);
    
    println!("\n🧮 CUDA Performance Test...");
    test_cuda_performance(verbose);
    
    println!("\n🔥 TensorFlow GPU Test...");
    test_tensorflow_gpu(verbose);
    
    println!("\n🚀 PyTorch GPU Test...");
    test_pytorch_gpu(verbose);
    
    println!("\n🌡️  System Monitoring...");
    monitor_system_during_load(verbose);
}

fn test_gpu_memory(verbose: bool) {
    let test_script = r#"
import torch
import time
try:
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {device}")
        print(f"Total Memory: {memory_gb:.1f} GB")
        
        # Memory allocation test
        sizes = [1, 2, 4, 8]
        for size_gb in sizes:
            try:
                size_bytes = int(size_gb * 1e9 / 4)  # float32 = 4 bytes
                tensor = torch.randn(size_bytes, device='cuda')
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"Allocated {size_gb}GB: ✅ Success ({allocated:.1f}GB used)")
                del tensor
                torch.cuda.empty_cache()
                time.sleep(0.1)
            except Exception as e:
                print(f"Allocated {size_gb}GB: ❌ Failed - {str(e)[:50]}")
                break
    else:
        print("❌ CUDA not available")
except Exception as e:
    print(f"❌ Error: {e}")
"#;
    
    match run_command(&format!("python -c \"{}\"", test_script), verbose) {
        Ok(output) => {
            for line in output.lines() {
                println!("   {}", line);
            }
        },
        Err(_) => println!("   ❌ Memory test failed - PyTorch not available"),
    }
}

fn test_cuda_performance(verbose: bool) {
    let test_script = r#"
import time
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
        # Matrix multiplication benchmark
        sizes = [1024, 2048, 4096]
        for size in sizes:
            torch.cuda.synchronize()
            start_time = time.time()
            
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            c = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = end_time - start_time
            gflops = (2 * size ** 3) / (elapsed * 1e9)
            print(f"Matrix {size}x{size}: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")
            
            del a, b, c
            torch.cuda.empty_cache()
    else:
        print("❌ CUDA not available")
except Exception as e:
    print(f"❌ Error: {e}")
"#;
    
    match run_command(&format!("python -c \"{}\"", test_script), verbose) {
        Ok(output) => {
            for line in output.lines() {
                println!("   {}", line);
            }
        },
        Err(_) => println!("   ❌ CUDA performance test failed"),
    }
}

fn test_tensorflow_gpu(verbose: bool) {
    let test_script = r#"
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # Simple computation test
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print("✅ TensorFlow GPU computation successful")
    else:
        print("❌ No GPU devices found by TensorFlow")
except ImportError:
    print("❌ TensorFlow not installed")
except Exception as e:
    print(f"❌ Error: {e}")
"#;
    
    match run_command(&format!("python -c \"{}\"", test_script), verbose) {
        Ok(output) => {
            for line in output.lines() {
                println!("   {}", line);
            }
        },
        Err(_) => println!("   ❌ TensorFlow test failed"),
    }
}

fn test_pytorch_gpu(verbose: bool) {
    let test_script = r#"
try:
    import torch
    print(f"PyTorch {torch.__version__}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA available: {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            print(f"   GPU {i}: {name} (CC {capability[0]}.{capability[1]})")
        
        # Simple computation test
        device = torch.device('cuda:0')
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        print("✅ PyTorch GPU computation successful")
    else:
        print("❌ CUDA not available in PyTorch")
except ImportError:
    print("❌ PyTorch not installed")
except Exception as e:
    print(f"❌ Error: {e}")
"#;
    
    match run_command(&format!("python -c \"{}\"", test_script), verbose) {
        Ok(output) => {
            for line in output.lines() {
                println!("   {}", line);
            }
        },
        Err(_) => println!("   ❌ PyTorch test failed"),
    }
}

fn monitor_system_during_load(_verbose: bool) {
    // System monitoring during load
    if let Ok(output) = run_command("nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits", false) {
        println!("   Current GPU Status:");
        for (i, line) in output.lines().enumerate() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                println!("   GPU {}: {}°C, {}W, {}% utilization", i, parts[0], parts[1], parts[2]);
            }
        }
    } else {
        println!("   ❌ Unable to monitor GPU status");
    }
}

// Update Checker Feature
pub fn check_for_updates(verbose: bool) {
    println!("=== 🔄 Checking for Updates ===\n");
    
    println!("🔧 NVIDIA Driver Updates:");
    check_nvidia_driver_updates(verbose);
    
    println!("\n⚙️  CUDA Toolkit Updates:");
    check_cuda_updates(verbose);
    
    println!("\n🔥 TensorFlow Updates:");
    check_tensorflow_updates(verbose);
    
    println!("\n🚀 PyTorch Updates:");
    check_pytorch_updates(verbose);
}

fn check_nvidia_driver_updates(verbose: bool) {
    // Check current driver version
    match get_nvidia_driver_version(verbose) {
        Ok(current_version) => {
            println!("   Current Driver: {}", current_version.trim());
            println!("   Latest Info: Check https://www.nvidia.com/Download/index.aspx");
            println!("   💡 Tip: Use GeForce Experience for automatic updates");
        },
        Err(_) => println!("   ❌ No NVIDIA driver detected"),
    }
}

fn check_cuda_updates(_verbose: bool) {
    println!("   Current CUDA: Check with 'nvcc --version'");
    println!("   Latest Info: https://developer.nvidia.com/cuda-downloads");
    println!("   💡 Tip: CUDA 12.x recommended for latest frameworks");
}

fn check_tensorflow_updates(verbose: bool) {
    match get_tensorflow_version(verbose) {
        Ok(current_version) => {
            println!("   Current TensorFlow: {}", current_version.trim());
            println!("   Check latest: pip list --outdated | grep tensorflow");
            println!("   Update: pip install --upgrade tensorflow");
        },
        Err(_) => println!("   ❌ TensorFlow not installed"),
    }
}

fn check_pytorch_updates(verbose: bool) {
    match get_pytorch_version(verbose) {
        Ok(current_version) => {
            println!("   Current PyTorch: {}", current_version.trim());
            println!("   Check latest: pip list --outdated | grep torch");
            println!("   Update: Visit https://pytorch.org/get-started/locally/");
        },
        Err(_) => println!("   ❌ PyTorch not installed"),
    }
}

// Configuration Validator Feature
pub fn validate_configuration(verbose: bool) {
    println!("=== ✅ Configuration Validation ===\n");
    
    println!("📝 Environment Variables:");
    validate_environment_variables();
    
    println!("\n🔗 Library Linking:");
    validate_library_linking(verbose);
    
    println!("\n🛡️  Permissions:");
    validate_permissions();
    
    println!("\n🌐 Network/Firewall:");
    validate_network_access();
}

fn validate_environment_variables() {
    let cuda_vars = vec![
        ("CUDA_PATH", "CUDA installation path"),
        ("CUDA_HOME", "CUDA home directory"),
        ("PATH", "Should include CUDA bin directory"),
        ("LD_LIBRARY_PATH", "Should include CUDA lib64 (Linux)"),
    ];
    
    for (var, description) in cuda_vars {
        if let Ok(value) = env::var(var) {
            println!("   ✅ {}: Set ({} chars)", var, value.len());
                    if let Ok(_) = env::var("VERBOSE_ENV_CHECK") {
            println!("      Description: {}", description);
        }
        } else {
            println!("   ⚠️  {}: Not set ({})", var, description);
        }
    }
}

fn validate_library_linking(verbose: bool) {
    let libraries = vec![
        ("libcuda.so.1", "NVIDIA driver library"),
        ("libcudart.so", "CUDA runtime library"),
        ("libcublas.so", "CUDA BLAS library"),
        ("libcudnn.so", "cuDNN library"),
    ];
    
    for (lib, description) in libraries {
        if cfg!(target_os = "linux") {
            match run_command(&format!("ldconfig -p | grep {}", lib), verbose) {
                Ok(output) => {
                    if !output.trim().is_empty() {
                        println!("   ✅ {}: Found", lib);
                    } else {
                        println!("   ❌ {}: Not found ({})", lib, description);
                    }
                },
                Err(_) => println!("   ⚠️  {}: Cannot check ({})", lib, description),
            }
        } else {
            println!("   ⚠️  Library checking not implemented for this OS");
            break;
        }
    }
}

fn validate_permissions() {
    // Check CUDA device permissions
    if cfg!(target_os = "linux") {
        if Path::new("/dev/nvidia0").exists() {
            println!("   ✅ NVIDIA device files exist");
            // Check if current user can access
            match std::fs::metadata("/dev/nvidia0") {
                Ok(_) => println!("   ✅ NVIDIA device accessible"),
                Err(_) => println!("   ❌ NVIDIA device not accessible - check permissions"),
            }
        } else {
            println!("   ❌ NVIDIA device files not found");
        }
    } else {
        println!("   ⚠️  Permission checking not implemented for this OS");
    }
}

fn validate_network_access() {
    println!("   💡 Network validation not implemented yet");
    println!("   💡 Manually check: Can access nvidia.com, pytorch.org, tensorflow.org");
}

// Environment Export Feature
pub fn export_environment(filename: &str, verbose: bool) {
    println!("📤 Exporting environment to {}...", filename);
    
    let config = collect_environment_config(verbose);
    
    match serde_json::to_string_pretty(&config) {
        Ok(json) => {
            match std::fs::write(filename, json) {
                Ok(_) => println!("✅ Environment exported successfully!"),
                Err(e) => println!("❌ Failed to write file: {}", e),
            }
        },
        Err(e) => println!("❌ Failed to serialize config: {}", e),
    }
}

// Environment Import Feature
pub fn import_environment(filename: &str, verbose: bool) {
    println!("📥 Importing environment from {}...", filename);
    
    let current_config = collect_environment_config(verbose);
    
    match std::fs::read_to_string(filename) {
        Ok(content) => {
            match serde_json::from_str::<EnvironmentConfig>(&content) {
                Ok(imported_config) => {
                    compare_environments(&current_config, &imported_config);
                },
                Err(e) => println!("❌ Failed to parse config file: {}", e),
            }
        },
        Err(e) => println!("❌ Failed to read file: {}", e),
    }
}

fn collect_environment_config(verbose: bool) -> EnvironmentConfig {
    let mut system = System::new_all();
    system.refresh_all();
    
    // System info
    let system_info = SystemInfo {
        os: format!("{} {}", System::name().unwrap_or_default(), System::os_version().unwrap_or_default()),
        arch: env::consts::ARCH.to_string(),
        cpu: system.cpus().first().map(|cpu| cpu.brand().to_string()).unwrap_or_default(),
        total_memory_gb: system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
        python_version: get_python_version(),
    };
    
    // CUDA info
    let cuda_info = CudaInfo {
        driver_version: get_nvidia_driver_version(verbose).ok(),
        cuda_version: get_cuda_toolkit_version(verbose).ok(),
        cudnn_version: get_cudnn_version(verbose).ok(),
        gpus: get_gpu_list(verbose),
    };
    
    // Framework info
    let frameworks = FrameworkInfo {
        tensorflow: get_tensorflow_version(verbose).ok(),
        pytorch: get_pytorch_version(verbose).ok(),
    };
    
    EnvironmentConfig {
        system_info,
        cuda_info,
        frameworks,
        timestamp: Utc::now(),
        hostname: System::host_name().unwrap_or_default(),
    }
}

fn get_python_version() -> Option<String> {
    if let Ok(output) = run_command("python --version", false) {
        Some(output.trim().to_string())
    } else if let Ok(output) = run_command("python3 --version", false) {
        Some(output.trim().to_string())
    } else {
        None
    }
}

fn get_gpu_list(verbose: bool) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    
    if let Ok(output) = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits", verbose) {
        for line in output.lines() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 2 {
                gpus.push(GpuInfo {
                    name: parts[0].to_string(),
                    memory_gb: parts[1].parse::<f64>().ok().map(|mb| mb / 1024.0),
                    compute_capability: None, // Would need additional query
                });
            }
        }
    }
    
    gpus
}

fn compare_environments(current: &EnvironmentConfig, imported: &EnvironmentConfig) {
    println!("\n=== 📊 Environment Comparison ===");
    
    println!("\n🖥️  System Comparison:");
    println!("   Current:  {} ({})", current.system_info.os, current.system_info.arch);
    println!("   Imported: {} ({})", imported.system_info.os, imported.system_info.arch);
    
    if current.system_info.os != imported.system_info.os {
        println!("   ⚠️  Different operating systems detected!");
    }
    
    println!("\n🔧 CUDA Comparison:");
    compare_versions("Driver", &current.cuda_info.driver_version, &imported.cuda_info.driver_version);
    compare_versions("CUDA", &current.cuda_info.cuda_version, &imported.cuda_info.cuda_version);
    compare_versions("cuDNN", &current.cuda_info.cudnn_version, &imported.cuda_info.cudnn_version);
    
    println!("\n🤖 Framework Comparison:");
    compare_versions("TensorFlow", &current.frameworks.tensorflow, &imported.frameworks.tensorflow);
    compare_versions("PyTorch", &current.frameworks.pytorch, &imported.frameworks.pytorch);
    
    println!("\n📅 Import Info:");
    println!("   Exported: {}", imported.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
    println!("   Hostname: {}", imported.hostname);
}

fn compare_versions(component: &str, current: &Option<String>, imported: &Option<String>) {
    match (current, imported) {
        (Some(curr), Some(imp)) => {
            if curr == imp {
                println!("   ✅ {}: {} (matches)", component, curr);
            } else {
                println!("   ⚠️  {}: {} vs {} (different)", component, curr, imp);
            }
        },
        (Some(curr), None) => println!("   ➕ {}: {} (not in import)", component, curr),
        (None, Some(imp)) => println!("   ➖ {}: {} (missing locally)", component, imp),
        (None, None) => println!("   ❌ {}: Not available in either", component),
    }
}