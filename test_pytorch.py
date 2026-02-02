import torch
import platform

print("="*60)
print("🔥 PyTorch M3 Mac Test")
print("="*60)

# System info
print(f"\n💻 System: {platform.system()} {platform.machine()}")
print(f"🐍 Python: {platform.python_version()}")
print(f"🔦 PyTorch: {torch.__version__}")

# Check MPS (Apple Silicon GPU acceleration)
print(f"\n⚡ MPS (Metal) Available: {torch.backends.mps.is_available()}")
print(f"⚡ MPS Built: {torch.backends.mps.is_built()}")

# Determine best device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"\n✅ Using: MPS (Apple Silicon GPU) - FAST! 🚀")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✅ Using: CUDA GPU")
else:
    device = torch.device("cpu")
    print(f"\n✅ Using: CPU")

# Quick performance test
print(f"\n🧪 Running quick tensor test on {device}...")

# Create random tensors
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

# Matrix multiplication
import time
start = time.time()
z = torch.matmul(x, y)
elapsed = time.time() - start

print(f"✅ Matrix multiplication (1000x1000): {elapsed*1000:.2f}ms")
print(f"✅ Result shape: {z.shape}")
print(f"✅ Device: {z.device}")

# Neural network test
print(f"\n🧠 Testing neural network operations...")
model = torch.nn.Sequential(
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 10)
).to(device)

input_tensor = torch.randn(32, 100).to(device)
output = model(input_tensor)

print(f"✅ Model input: {input_tensor.shape}")
print(f"✅ Model output: {output.shape}")
print(f"✅ Model device: {next(model.parameters()).device}")

print("\n" + "="*60)
print("🎉 PyTorch is working perfectly on your M3 Mac!")
print("="*60)