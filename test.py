import sys
import torch
import torchvision
import torchaudio
import pytorch3d
import kornia

print(f"Python version: {sys.version.split()[0]}") # 显示Python版本
print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print(f"TorchAudio version: {torchaudio.__version__}")
print(f"PyTorch3D version: {pytorch3d.__version__}") # 可能需要检查特定属性如果__version__不是标准的
print(f"Kornia version: {kornia.__version__}")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} compute capability: {torch.cuda.get_device_capability(i)}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")