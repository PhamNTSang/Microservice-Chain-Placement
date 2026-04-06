import torch
import gymnasium as gym
import stable_baselines3 as sb3
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Gymnasium version: {gym.__version__}")
print(f"Stable Baselines3 version: {sb3.__version__}")
print("Setup successful!")
