import subprocess
import re
import sys

# Minimum required NVIDIA driver version for CUDA 12.x.
MIN_DRIVER_VERSION = 525.60
# Minimum required CUDA Toolkit version.
MIN_CUDA_VERSION = 12.8

def run_command(command):
    """Runs a command and returns its output, handling errors."""
    try:
        # Execute the command without showing a console window for it (Windows-only feature)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            creationflags=0x08000000 # CREATE_NO_WINDOW
        )
        return result.stdout
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] The command '{command[0]}' was not found.")
        if command[0] == 'git':
            print("Please install Git from https://git-scm.com/downloads and ensure it's in your system's PATH.")
        elif command[0] == 'nvidia-smi':
            print("This means the NVIDIA driver is not installed or not in your system's PATH.")
            print("Please install the latest NVIDIA drivers for your GPU from https://www.nvidia.com/Download/index.aspx")
        elif command[0] == 'nvcc':
            print("This means the NVIDIA CUDA Toolkit is not installed or not in your system's PATH.")
            print(f"Please install CUDA Toolkit {MIN_CUDA_VERSION} or newer from https://developer.nvidia.com/cuda-toolkit")
        return None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] A command failed to run. Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return None

def check_cuda_version():
    """Checks for a compatible CUDA Toolkit version."""
    print("\n[INFO] Checking for CUDA Toolkit...")
    nvcc_output = run_command(["nvcc", "--version"])

    if nvcc_output is None:
        sys.exit(1)

    # Regex to find the version number, e.g., "V12.8.1"
    version_match = re.search(r"release (\d+\.\d+)", nvcc_output)
    if not version_match:
        print(f"\n[FATAL ERROR] Could not parse CUDA Toolkit version from:\n{nvcc_output}")
        sys.exit(1)

    cuda_version = float(version_match.group(1))
    if cuda_version < MIN_CUDA_VERSION:
        print(f"\n[FATAL ERROR] Your CUDA Toolkit version ({cuda_version}) is too old.")
        print(f"             Please upgrade to version {MIN_CUDA_VERSION} or newer from")
        print("             https://developer.nvidia.com/cuda-toolkit")
        sys.exit(1)
    print(f"[SUCCESS] CUDA Toolkit version ({cuda_version}) is compatible.")


def check_environment():
    """
    Checks for Git, a compatible NVIDIA GPU, driver version, and CUDA Toolkit.
    Exits with a non-zero status code if any check fails.
    """
    # --- Git Check ---
    print("[INFO] Checking for Git...")
    if run_command(["git", "--version"]) is None:
        sys.exit(1) # Exits the script with an error code
    print("[SUCCESS] Git is installed.")

    # --- NVIDIA GPU and Driver Check ---
    print("\n[INFO] Checking for compatible NVIDIA GPU and drivers...")
    smi_output = run_command(["nvidia-smi", "--query-gpu=gpu_name,driver_version", "--format=csv,noheader"])

    if smi_output is None:
        sys.exit(1)

    # --- GPU Name Check ---
    gpu_name = smi_output.split(',')[0].strip()
    # This regex now looks for GTX 10 or 16 series, or RTX Quadro, 20, 30, 40, or 50 series.
    if not re.search(r"(GTX (10|16)\d{2}|RTX Quadro|RTX (20|30|40|50)\d{2})", gpu_name, re.IGNORECASE):
        print(f"\n[FATAL ERROR] No compatible NVIDIA GTX (10/16 series), RTX (20/30/40/50 series), or RTX Quadro GPU detected.")
        print("             This software is specifically designed for these GPU series.")
        print(f"             Detected GPU: {gpu_name}")
        sys.exit(1)
    print(f"[SUCCESS] Compatible GPU found: {gpu_name}")

    # --- Driver Version Check ---
    driver_version_str = smi_output.split(',')[1].strip()
    driver_version_match = re.match(r"(\d+\.\d+)", driver_version_str)
    if not driver_version_match:
        print(f"\n[FATAL ERROR] Could not parse NVIDIA driver version from: '{driver_version_str}'")
        sys.exit(1)

    driver_version = float(driver_version_match.group(1))
    if driver_version < MIN_DRIVER_VERSION:
        print(f"\n[FATAL ERROR] Your NVIDIA driver version ({driver_version}) is too old for this software.")
        print(f"             Please update to version {MIN_DRIVER_VERSION} or newer.")
        print("             Visit https://www.nvidia.com/Download/index.aspx to get the latest drivers.")
        sys.exit(1)
    print(f"[SUCCESS] NVIDIA driver version ({driver_version}) is up to date.")

    # --- CUDA Toolkit Check ---
    check_cuda_version()

    print("\n[INFO] System check passed! Proceeding with installation.")
    sys.exit(0) # Exits the script with a success code

if __name__ == "__main__":
    check_environment()