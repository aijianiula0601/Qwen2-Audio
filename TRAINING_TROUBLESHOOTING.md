# LlamaAudio Training Troubleshooting Guide

This guide helps resolve common training issues, especially Out of Memory (OOM) errors.

## 🔧 Quick Diagnosis

Before training, run the environment diagnosis script:
```bash
bash scripts/diagnose_training_env.sh
```

## ❌ Common Error: Process Killed (Exit Code -9)

If you see messages like:
```
[INFO] [launch.py:319:sigkill_handler] Killing subprocess
exits with return code = -9
```

This indicates **Out of Memory (OOM)**. The system killed the training process to prevent system crash.

## 🛠️ Solutions (in order of preference)

### 1. Use Optimized Configuration (Recommended)
The updated training script now uses memory-optimized settings:
- **DeepSpeed config**: `configs/deepspeed_config_8b.json` 
- **Micro batch size**: 1 per GPU (reduced from 2)
- **ZeRO Stage**: 2 (optimal for 8B models)
- **Gradient accumulation**: 8 steps

### 2. Enable CPU Offloading
Edit `configs/deepspeed_config_8b.json`:
```json
{
  "zero_optimization": {
    "stage": 2,
    "cpu_offload": true
  }
}
```

### 3. Use Smaller Model
Switch to the 3B model for lower memory requirements:
```bash
bash scripts/run_stage1_training.sh --config configs/stage1_training_config_3b.yaml
```

### 4. Reduce Audio Length
Edit your training config file:
```yaml
training:
  max_audio_length: 15.0  # Reduce from 20 seconds
```

### 5. Use ZeRO Stage 3
Edit `configs/deepspeed_config_8b.json`:
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    }
  }
}
```

## 🧐 Memory Monitoring

### Check GPU Memory During Training
```bash
watch -n 1 nvidia-smi
```

### Check System Memory
```bash
free -h
```

### Memory Usage by Process
```bash
ps aux --sort=-%mem | head -10
```

## 🏥 Emergency Recovery

If training keeps failing:

1. **Clear GPU memory cache**:
   ```bash
   python3 -c "import torch; torch.cuda.empty_cache()"
   ```

2. **Restart training with minimal config**:
   ```bash
   bash scripts/run_stage1_training.sh --config configs/stage1_training_config_3b.yaml
   ```

3. **Check for zombie processes**:
   ```bash
   ps aux | grep python
   pkill -f deepspeed  # If needed
   ```

## 📊 Expected Memory Usage

| Model Size | GPU Memory (per GPU) | System Memory | Recommended GPUs |
|------------|---------------------|---------------|------------------|
| 3B         | ~8-12 GB           | 32+ GB        | 4+ × V100/A100   |
| 8B         | ~12-18 GB          | 64+ GB        | 8+ × V100/A100   |
| 70B        | ~25-30 GB          | 128+ GB       | 8+ × A100        |

## 🎯 Optimization Tips

1. **Start small**: Use 3B model first to verify setup
2. **Monitor continuously**: Watch nvidia-smi during training
3. **Gradual scaling**: Increase batch size gradually
4. **Use mixed precision**: Keep FP16 enabled
5. **Profile memory**: Use DeepSpeed's memory profiler

## 🆘 Still Having Issues?

1. Run the diagnostic script: `bash scripts/diagnose_training_env.sh`
2. Check GPU temperature (should be < 80°C)
3. Ensure adequate cooling
4. Try training on fewer GPUs first
5. Consider using gradient checkpointing

## 📝 Configuration Reference

### Minimal Memory Configuration
For systems with limited GPU memory:

**DeepSpeed Config** (`configs/deepspeed_minimal.json`):
```json
{
  "train_batch_size": 8,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true
  }
}
```

**Training Config**:
```yaml
training:
  batch_size: 1
  max_audio_length: 10.0
  num_workers: 2
  gradient_accumulation_steps: 8
``` 