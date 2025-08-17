# CUDA Doctor 🏥

A comprehensive cross-platform diagnostic tool for NVIDIA GPU setup and AI framework installations on **Windows**, **Linux**, and **macOS**.

## 🚀 Features

### 🔍 **Core Diagnostics**
- **🖥️ GPU Detection**: Identifies NVIDIA GPUs across all operating systems
  - Windows: Uses `wmic` (Windows Management Instrumentation)
  - Linux: Uses `lspci` with `nvidia-smi` fallback
  - macOS: Uses `system_profiler` (with note about NVIDIA support)
- **🔧 Driver Verification**: Checks NVIDIA driver version via `nvidia-smi`
- **⚙️ CUDA Toolkit**: Detects CUDA toolkit installation and version
- **🧠 cuDNN Detection**: Finds cuDNN library version
- **🔥 Framework Support**: Checks TensorFlow and PyTorch installations
- **🔄 Cross-Platform**: No Unix-specific commands like `grep` or pipes
- **📦 Multiple Detection Methods**: Uses various fallback strategies for robust detection

### 🆕 **Advanced Features**

#### **📊 System Information (`--sysinfo`)**
- Detailed hardware specifications (CPU, RAM, architecture)
- GPU compute capabilities and memory information
- Python environment analysis
- Environment variables inspection
- OS and kernel version details

#### **🔗 Compatibility Matrix (`--compatibility`)**
- CUDA ↔ Driver version compatibility
- TensorFlow ↔ CUDA/cuDNN compatibility 
- PyTorch ↔ CUDA version compatibility
- Python version requirements
- Compute capability requirements
- Recommended stable combinations

#### **⚡ Performance Benchmarks (`--benchmark`)**
- GPU memory allocation tests
- CUDA performance matrix operations
- TensorFlow GPU computation tests
- PyTorch GPU computation tests
- Real-time system monitoring during tests

#### **🔄 Update Checker (`--check-updates`)**
- Current vs latest driver versions
- CUDA toolkit update notifications
- Framework version update guidance
- Update installation instructions

#### **🎮 Multi-GPU Analysis (`--multi-gpu`)**
- Individual GPU specifications
- Memory usage and utilization per GPU
- Temperature and power monitoring
- GPU topology and interconnects (SLI/NVLink)

#### **📤📥 Environment Export/Import**
- **Export** (`--export file.json`): Save complete environment config
- **Import** (`--import file.json`): Compare environments across systems
- Team environment standardization
- CI/CD integration support
- Timestamped configuration snapshots

#### **✅ Configuration Validator (`--validate-config`)**
- Environment variables validation
- Library linking verification (Linux)
- CUDA device permissions check
- System configuration diagnostics

#### **💡 Installation Guides (`--showfix`)**
- Platform-specific installation instructions
- Hardware troubleshooting guidance
- Step-by-step fix procedures
- Official download links and resources

## 🎛️ Usage

### **Basic Diagnostics**
```bash
# Quick status check (clean output)
cuda-doctor

# Detailed diagnostic information
cuda-doctor --verbose

# Installation guides for missing components
cuda-doctor --showfix

# Combine flags for comprehensive analysis
cuda-doctor --verbose --showfix
```

### **Advanced Analysis**
```bash
# Detailed system information
cuda-doctor --sysinfo

# Version compatibility matrix
cuda-doctor --compatibility

# Performance benchmarking
cuda-doctor --benchmark

# Multi-GPU analysis
cuda-doctor --multi-gpu

# Check for updates
cuda-doctor --check-updates

# Validate system configuration
cuda-doctor --validate-config
```

### **Environment Management**
```bash
# Export current environment
cuda-doctor --export my-config.json

# Import and compare environment
cuda-doctor --import my-config.json

# Share team configurations
cuda-doctor --export team-standard.json
# (team members can import and compare)
```

### **Installation & Building**
```bash
# Build the project
cargo build --release

# Install globally (optional)
cargo install --path .

# Run from anywhere
cuda-doctor --sysinfo
```

## 📊 Output Examples

### **Default Mode (Clean & Fast)**
```
🖥️  Checking NVIDIA GPU... ✅ Found
🔧 Checking NVIDIA Driver... ✅ Found
⚙️  Checking CUDA Toolkit... ✅ Found
🧠 Checking cuDNN... ✅ Found
🔥 Checking TensorFlow... ❌ Not found
🚀 Checking PyTorch... ✅ Found

💡 Use --verbose for details, --showfix for installation guides
```

### **System Information Mode (`--sysinfo`)**
```
=== 🖥️  Detailed System Information ===

🔧 System Details:
   OS: Ubuntu 22.04 LTS
   Architecture: x86_64
   Hostname: workstation

💻 CPU Information:
   Model: Intel(R) Core(TM) i9-12900K
   Cores: 16 physical, 24 logical
   Frequency: 3.20 GHz

🧠 Memory Information:
   Total RAM: 64.0 GB
   Available: 45.2 GB
   Used: 18.8 GB (29.4%)

🎮 GPU Information:
   GPU 0: NVIDIA GeForce RTX 4090
   Memory: 24564 MB
   Compute Capability: 8.9
```

### **Compatibility Matrix (`--compatibility`)**
```
=== 🔗 Version Compatibility Matrix ===

📊 CUDA ↔ Driver Compatibility:
   CUDA 12.3+  → Driver 545.23+
   CUDA 12.2   → Driver 535.86+
   CUDA 12.1   → Driver 530.30+

🔥 TensorFlow ↔ CUDA Compatibility:
   TensorFlow 2.15+ → CUDA 12.3, cuDNN 8.9
   TensorFlow 2.14  → CUDA 12.2, cuDNN 8.9

📝 Recommended Combinations:
   🔥 Latest Stable: CUDA 12.2 + cuDNN 8.9 + TensorFlow 2.14
```

### **Environment Export/Import**
```bash
# Export
$ cuda-doctor --export prod-env.json
📤 Exporting environment to prod-env.json...
✅ Environment exported successfully!

# Import & Compare
$ cuda-doctor --import prod-env.json
📥 Importing environment from prod-env.json...

=== 📊 Environment Comparison ===
🖥️  System Comparison:
   Current:  Ubuntu 22.04 (x86_64)
   Imported: Ubuntu 20.04 (x86_64)
   ⚠️  Different operating systems detected!

🔧 CUDA Comparison:
   ✅ Driver: 535.86.05 (matches)
   ⚠️  CUDA: 12.2 vs 11.8 (different)
```

## 🌟 **All Available Commands**

| Command | Description | Example |
|---------|-------------|---------|
| (none) | Standard diagnostics | `cuda-doctor` |
| `--verbose` | Detailed output | `cuda-doctor -v` |
| `--showfix` | Installation guides | `cuda-doctor --showfix` |
| `--sysinfo` | System information | `cuda-doctor --sysinfo` |
| `--compatibility` | Version matrix | `cuda-doctor --compatibility` |
| `--benchmark` | Performance tests | `cuda-doctor --benchmark` |
| `--multi-gpu` | Multi-GPU analysis | `cuda-doctor --multi-gpu` |
| `--check-updates` | Update checker | `cuda-doctor --check-updates` |
| `--export` | Export environment | `cuda-doctor --export config.json` |
| `--import` | Import environment | `cuda-doctor --import config.json` |
| `--validate-config` | Config validator | `cuda-doctor --validate-config` |
| `--help` | Show help | `cuda-doctor --help` |
| `--version` | Show version | `cuda-doctor --version` |

## 🔧 Cross-Platform Compatibility

### Windows Support ✅
- Uses Windows-native commands (`wmic`, `cmd`)
- Supports both `pip` and `conda` package managers
- Compatible with both `python` and `python3` executables
- Searches Windows-specific CUDA installation paths

### Linux Support ✅
- Uses Linux-native commands (`lspci`, `sh`)
- Fallback to `nvidia-smi` when `lspci` unavailable
- Supports standard Linux CUDA paths (`/usr/local/cuda`, `/opt/cuda`)
- Library linking validation via `ldconfig`

### macOS Support ✅
- Uses macOS-native `system_profiler` command
- Includes helpful note about modern macOS NVIDIA support limitations

## 📦 Dependencies

- Rust 1.70+
- walkdir 2.4+
- regex 1.10+
- clap 4.4+ (CLI argument parsing)
- serde 1.0+ (JSON serialization)
- sysinfo 0.30+ (system information)
- chrono 0.4+ (timestamps)

## 🎯 Use Cases

### **👨‍💻 Developers**
- Quick environment verification
- Performance benchmarking
- Framework compatibility checking
- Installation troubleshooting

### **🏢 DevOps/Teams**
- Environment standardization via export/import
- CI/CD integration for environment validation
- Multi-machine setup verification
- Team onboarding automation

### **🔬 Researchers**
- Multi-GPU system analysis
- Performance baseline establishment
- Hardware capability assessment
- Environment documentation

### **🎓 Students/Beginners**
- Step-by-step installation guidance
- Compatibility learning resource
- Environment setup validation
- Troubleshooting assistance

## 🚀 **What Makes CUDA Doctor Special**

1. **🎯 Comprehensive**: All GPU/AI tools in one diagnostic
2. **🌍 Cross-Platform**: Works on Windows, Linux, macOS  
3. **📊 Advanced Features**: Beyond basic detection
4. **🛠️ Actionable**: Not just detection, but solutions
5. **👥 Team-Friendly**: Environment sharing capabilities
6. **🏥 True "Doctor"**: Diagnoses AND provides treatment

## 📚 Version History

- **v0.1.0**: Complete rewrite with advanced features
  - Added system information analysis
  - Added compatibility matrix
  - Added performance benchmarking
  - Added environment export/import
  - Added configuration validation
  - Added multi-GPU support
  - Added update checker
  - Enhanced installation guides

## 📄 License

MIT License
