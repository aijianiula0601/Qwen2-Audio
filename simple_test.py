import torch

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA execution for better error messages
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Enable device-side assertions for debugging

from src import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor, Qwen2AudioConfig

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model_dir='/data1/hjh/huggingface/Qwen-7B'

# 首先加载配置
config = Qwen2AudioConfig.from_pretrained(model_dir)
# 设置正确的维度
config.audio_hidden_size = 1280  # Whisper 编码器的输出维度
config.llm_hidden_size = 3584    # Qwen2 的隐藏维度

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_dir,  # 使用预训练模型
    config=config,  # 使用加载的配置
    torch_dtype=torch.float16,  # Use float16 for better memory efficiency
    device_map="auto" if device == "cuda" else None
)

print("load model done!!!")

processor = Qwen2AudioProcessor()

if device == "cpu":
    model = model.to(device)

# Fix the special token mismatch
# Update the model's special token IDs to match the tokenizer
model.update_special_tokens(processor.tokenizer)

print("✅ 模型加载成功")


wav_file = "/mnt/cephfs/hjh/pycharm_projects/my_github/nlp/Qwen2-Audio/data/processed/librispeech/train-clean-100/19-198-0003.wav"

conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio": wav_file},
        {"type": "text", "text": "请描述这段音频的内容, 并给出音频的摘要，详细输出"}
    ]}
]

# 处理对话模板
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

print("---text---")
print(text)
print("----------")

# 处理音频和文本输入
inputs = processor(text=text, audios=[wav_file], return_tensors="pt")

print("-----inputs ids-----")
print(inputs["input_ids"])

print("-----inputs.shape-----")
for k in inputs:
    print(k, inputs[k].shape)
print("----------------------")

# 确保输入类型正确
if device == "cuda":
    for k, v in inputs.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        # Move all inputs to GPU and convert to appropriate dtypes
        if k == "input_ids" or k == "attention_mask":
            inputs[k] = inputs[k].to(device).to(torch.long)
        elif k == "audio_features":
            inputs[k] = inputs[k].to(device).to(torch.float16)  # Use float16 for audio features

# Check for invalid token IDs before generation
print("🔍 Checking for invalid token IDs...")
input_ids = inputs["input_ids"]
vocab_size = len(processor.tokenizer)
max_valid_id = vocab_size - 1

print(f"Vocabulary size: {vocab_size}")
print(f"Max valid token ID: {max_valid_id}")
print(f"Input token ID range: {input_ids.min().item()} to {input_ids.max().item()}")

# Check if any token IDs are out of range
invalid_tokens = input_ids[input_ids >= vocab_size]
print("-------------invalid_tokens:",invalid_tokens)
if len(invalid_tokens) > 0:
    print(f"⚠️  Found {len(invalid_tokens)} invalid token IDs: {invalid_tokens}")
    print("This could cause the CUDA device-side assertion error!")
    
    # Replace invalid tokens with UNK token
    unk_token_id = processor.tokenizer.unk_token_id or 0
    print(f"Replacing invalid tokens with UNK token ID: {unk_token_id}")
    inputs["input_ids"] = torch.where(input_ids >= vocab_size, unk_token_id, input_ids)
else:
    print("✅ All token IDs are valid")

# 生成回复
print("🚀 Starting generation...")
with torch.no_grad():
    try:
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True  # Enable KV cache for faster generation
        )
        print("✅ Generation completed successfully!")
    except Exception as e:
        print(f"❌ Generation failed with error: {e}")
        print("Trying with greedy decoding (do_sample=False)...")
        try:
            outputs = model.generate(
                **inputs,
                max_length=512,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for faster generation
            )
            print("✅ Generation completed with greedy decoding!")
        except Exception as e2:
            print(f"❌ Generation failed again: {e2}")
            raise e2

# 解码输出
response = processor.decode(outputs[0], skip_special_tokens=True)
print(f"💬 模型回复: {response}")

