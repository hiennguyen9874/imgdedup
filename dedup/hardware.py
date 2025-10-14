"""
Hardware capability detection and optimization for GPU/CPU processing.
"""

import torch
from typing import List, Dict


def check_flash_attention_available() -> bool:
    """Check if flash attention 2 is available and properly installed"""
    try:
        import flash_attn

        # Check if flash_attn is properly installed and has required version
        if hasattr(flash_attn, "__version__") and flash_attn.__version__ >= "2.0.0":
            # Test if flash attention actually works by importing a key function
            from flash_attn import flash_attn_func

            return True
        else:
            print("Warning: flash_attn version < 2.0.0 detected, may cause issues")
            return False
    except ImportError:
        return False
    except Exception as e:
        print(f"Warning: flash_attn detected but not properly installed: {e}")
        return False


def check_bfloat16_support() -> bool:
    """Check if bfloat16 is supported on current hardware"""
    if not torch.cuda.is_available():
        # bfloat16 is available on CPU but typically slower
        # Check if CPU supports it properly
        try:
            return (
                torch.cuda.is_bf16_supported()
                if hasattr(torch.cuda, "is_bf16_supported")
                else False
            )
        except:
            return False

    # For CUDA devices, check compute capability
    # bfloat16 is well supported on Ampere (8.0+) and newer
    try:
        major, minor = torch.cuda.get_device_capability()
        # Ampere (A100, RTX 30xx) and newer have good bfloat16 support
        return major >= 8
    except:
        return False


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """
    Determine optimal dtype based on hardware.
    Priority: bfloat16 > float16 > float32
    """
    if device.type == "cuda":
        if check_bfloat16_support():
            print("Using bfloat16 (hardware supported)")
            return torch.bfloat16
        else:
            print("Using float16 (bfloat16 not supported on this GPU)")
            return torch.float16
    else:
        # CPU: bfloat16 can work but float32 is safer
        print("Using float32 (CPU mode)")
        return torch.float32


def get_optimal_attention_implementation() -> str:
    """
    Determine optimal attention implementation.
    Priority: flash_attention_2 > sdpa > default (eager)
    """
    if check_flash_attention_available():
        print("Using flash_attention_2 (detected flash-attn package)")
        return "flash_attention_2"
    else:
        # SDPA (Scaled Dot Product Attention) is available in PyTorch 2.0+
        try:
            # Check if PyTorch version supports SDPA
            import torch.nn.functional as F

            if hasattr(F, "scaled_dot_product_attention"):
                print("Using sdpa attention (PyTorch 2.0+ native)")
                return "sdpa"
        except:
            pass

        print("Using default (eager) attention")
        return "eager"


def detect_available_gpus() -> List[int]:
    """
    Detect available CUDA GPUs and return their device IDs.
    Returns list of GPU device IDs that can be used.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, will use CPU")
        return []

    num_gpus = torch.cuda.device_count()
    available_gpus = []

    for i in range(num_gpus):
        try:
            # Test if GPU is accessible
            torch.cuda.set_device(i)
            # Simple test to verify GPU functionality
            test_tensor = torch.tensor([1.0], device=f"cuda:{i}")
            available_gpus.append(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Available")
        except Exception as e:
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Not available ({e})")

    return available_gpus


def get_gpu_memory_info(gpu_id: int) -> Dict[str, float]:
    """
    Get memory information for a specific GPU.
    Returns dict with total_memory, free_memory, used_memory in GB.
    """
    if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
        return {"total_memory": 0.0, "free_memory": 0.0, "used_memory": 0.0}

    torch.cuda.set_device(gpu_id)
    props = torch.cuda.get_device_properties(gpu_id)
    total_memory = props.total_memory / (1024**3)  # Convert to GB

    # Get current memory usage
    memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
    memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)

    return {
        "total_memory": total_memory,
        "free_memory": total_memory - memory_reserved,
        "used_memory": memory_allocated,
    }


def print_gpu_info():
    """Print detailed information about available GPUs"""
    available_gpus = detect_available_gpus()

    if not available_gpus:
        print("No CUDA GPUs available")
        return

    print(f"Found {len(available_gpus)} available GPU(s):")
    for gpu_id in available_gpus:
        memory_info = get_gpu_memory_info(gpu_id)
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"    Total Memory: {memory_info['total_memory']:.1f} GB")
        print(f"    Free Memory:  {memory_info['free_memory']:.1f} GB")
