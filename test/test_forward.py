import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import Qwen2AudioModel, Qwen2AudioConfig

# 配置参数
llm_name = "Qwen/Qwen2-7B"
audio_encoder_name = "openai/whisper-large-v3"

# 随机参数
batch_size = 2
text_len = 64  # 增加序列长度，确保能容纳audio tokens
audio_seq = 32
hidden_size = 3584  # projector输出维度
whisper_input_length = 3000  # Whisper期望的输入长度

# 初始化模型
config = Qwen2AudioConfig(
    audio_encoder_name=audio_encoder_name,
    llm_name=llm_name,
    audio_projector_config={
        'input_size': 1280,
        'hidden_size': hidden_size,
        'intermediate_size': 4096,
        'num_layers': 2,
        'dtype': torch.bfloat16
    },
    dtype=torch.bfloat16,
    freeze_audio_encoder=True,
    freeze_llm=True  # 加快测试
    # 
)
model = Qwen2AudioModel(config).cuda()
model.eval()

# 随机生成input_ids（包含audio_bos和audio_eos）
tokenizer = model.tokenizer
bos = tokenizer.convert_tokens_to_ids(config.audio_start_token)
eos = tokenizer.convert_tokens_to_ids(config.audio_end_token)
input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, text_len)).cuda()
input_ids[:, 4] = bos
input_ids[:, 5] = eos  # 让eos紧跟在bos后面

# attention_mask
attention_mask = torch.ones_like(input_ids).cuda()

# labels
labels = input_ids.clone()

# 随机生成audio_features
# Whisper期望的输入shape为[batch, 128, 3000]，其中128是mel特征维度
audio_features = torch.randn(batch_size,128, whisper_input_length, dtype=config.dtype).cuda()

# forward
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        audio_values=audio_features,  # 这里直接传audio_features，encode_audio会先用Whisper处理
        attention_mask=attention_mask,
        labels=labels
    )
    print("loss:", outputs.loss.item() if hasattr(outputs, 'loss') else None)
    print("logits shape:", outputs.logits.shape) 