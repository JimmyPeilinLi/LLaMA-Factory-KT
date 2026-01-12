from safetensors import safe_open
import torch
nan_count = 0
with safe_open('/mnt/data/lpl/ls/saves/nan_test_Kllama_deepseekV2/adapter_model.safetensors', framework='pt') as f:
    for key in f.keys():
        if torch.isnan(f.get_tensor(key)).any():
            print(f'NaN: {key}')
            nan_count += 1
print(f'Total NaN tensors: {nan_count}')