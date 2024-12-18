# Python GPU Development Environment Setup Guide

This guide explains how to set up a Python development environment with GPU support using DevContainer.

## Prerequisites

### 1. System Requirements
- Windows 10/11 or Linux
- NVIDIA GPU
- At least 10GB of free disk space
- 8GB+ RAM recommended

### 2. Required Software
1. **Docker Desktop**
   - Download from [Docker's official website](https://www.docker.com/products/docker-desktop)
   - Install and start the Docker service

2. **Visual Studio Code**
   - Download from [VS Code's official website](https://code.visualstudio.com/)
   - Install the "Dev Containers" extension from VS Code marketplace

3. **NVIDIA Driver**
   - Check your current driver version:
     ```bash
     nvidia-smi
     ```
   - If not installed, download and install from [NVIDIA's official website](https://www.nvidia.com/Download/index.aspx)

4. **NVIDIA Container Toolkit**
   - For Ubuntu/Debian:
     ```bash
     # Add NVIDIA package repositories
     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

     curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

     # Install the toolkit
     sudo apt-get update
     sudo apt-get install -y nvidia-container-toolkit
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo systemctl restart docker
     ```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Project Structure
Ensure you have these files in your `.devcontainer` directory:
```
.devcontainer/
├── devcontainer.json
├── Dockerfile
└── setup.sh
```

### 3. Launch DevContainer
1. Open VS Code in the project directory:
   ```bash
   code .
   ```
2. Press `F1` or `Ctrl+Shift+P` and type "Dev Containers: Reopen in Container"
3. Wait for the container to build and start (this may take several minutes)

### 4. Verify Installation
Once the container is running:
1. Open a terminal in VS Code (`` Ctrl+` ``)
2. Verify GPU access:
   ```bash
   nvidia-smi
   ```
3. Verify Python environment:
   ```bash
   python --version
   mamba --version
   ```

## Customization

### Adding Python Packages
1. Open `.devcontainer/setup.sh`
2. Uncomment and modify the mamba install line:
   ```bash
   mamba install -qy package1 package2  # Add your required packages
   ```

### Adding System Packages
1. Open `.devcontainer/Dockerfile`
2. Add packages to the apt-get install line:
   ```dockerfile
   RUN apt-get update \
       && export DEBIAN_FRONTEND=noninteractive \
       && apt-get -y install --no-install-recommends your-package-name \
       && apt-get clean \
       && rm -rf /var/lib/apt/lists/*
   ```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Verify NVIDIA driver installation on host:
     ```bash
     nvidia-smi
     ```
   - Check NVIDIA Container Toolkit installation:
     ```bash
     sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
     ```

2. **Container Build Fails**
   - Check Docker service is running
   - Ensure adequate disk space
   - Review build logs for specific errors

3. **Python Package Installation Fails**
   - Try updating mamba:
     ```bash
     mamba update -n base mamba
     ```
   - Check package compatibility with your Python version

### Getting Help
- Check [VS Code DevContainer documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- Visit [Docker documentation](https://docs.docker.com/)
- Review [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Notes
- The first build may take significant time depending on your internet connection
- GPU support requires compatible NVIDIA drivers on the host system
- Remember to rebuild container if you modify Dockerfile or devcontainer.json