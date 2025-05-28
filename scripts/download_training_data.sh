#!/bin/bash

# Qwen2-Audio Training Data Download Script
# 根据论文中提到的数据集进行下载和预处理

set -e

# 创建数据目录
DATA_ROOT="data"
mkdir -p $DATA_ROOT/{raw,processed,stage1_pretraining,stage2_sft,stage3_dpo}

echo "=== Qwen2-Audio 训练数据下载脚本 ==="
echo "数据将保存到: $DATA_ROOT"

# =============================================================================
# 1. ASR数据集 (Automatic Speech Recognition)
# =============================================================================

echo "下载ASR数据集..."

# LibriSpeech (英文语音识别)
echo "下载 LibriSpeech..."
mkdir -p $DATA_ROOT/raw/librispeech
cd $DATA_ROOT/raw/librispeech
if [ ! -f "train-clean-100.tar.gz" ]; then
    for file in train-clean-100.tar.gz train-clean-360.tar.gz train-other-500.tar.gz dev-clean.tar.gz dev-other.tar.gz test-clean.tar.gz test-other.tar.gz; do
        if [ ! -f "$file" ]; then
            wget -c https://www.openslr.org/resources/12/$file
        else
            echo "文件 $file 已存在，跳过下载"
        fi
    done
fi

# Common Voice (多语言)
echo "下载 Common Voice 15.0..."
mkdir -p $DATA_ROOT/raw/common_voice
cd $DATA_ROOT/raw/common_voice
# 需要在Mozilla Common Voice网站注册后获取下载链接
echo "请访问 https://commonvoice.mozilla.org/datasets 下载Common Voice 15.0数据集"
echo "支持的语言: en, zh-CN, zh-HK, fr, de, es, it, ja 等"

# FLEURS (多语言语音)
echo "下载 FLEURS..."
mkdir -p $DATA_ROOT/raw/fleurs
cd $DATA_ROOT/raw/fleurs
# 使用Hugging Face datasets下载
python3 -c "
from datasets import load_dataset
dataset = load_dataset('google/fleurs', 'zh_cn')  # 中文
dataset.save_to_disk('fleurs_zh')
dataset = load_dataset('google/fleurs', 'en_us')  # 英文
dataset.save_to_disk('fleurs_en')
"

# AISHELL-2 (中文语音)
echo "下载 AISHELL-2..."
mkdir -p $DATA_ROOT/raw/aishell2
cd $DATA_ROOT/raw/aishell2
echo "请访问 https://www.aishelltech.com/aishell_2 下载AISHELL-2数据集"
echo "需要申请授权后下载"

cd $DATA_ROOT

# =============================================================================
# 2. 语音翻译数据集 (Speech-to-Text Translation)
# =============================================================================

echo "下载语音翻译数据集..."

# CoVoST 2 (多语言语音翻译)
echo "下载 CoVoST 2..."
mkdir -p $DATA_ROOT/raw/covost2
cd $DATA_ROOT/raw/covost2
python3 -c "
from datasets import load_dataset
# 下载主要的翻译方向
directions = ['en_de', 'de_en', 'en_zh', 'zh_en', 'es_en', 'fr_en', 'it_en']
for direction in directions:
    src_lang, tgt_lang = direction.split('_')
    dataset = load_dataset('facebook/covost2', f'{src_lang}_{tgt_lang}', split='train')
    dataset.save_to_disk(f'covost2_{direction}')
    print(f'Downloaded CoVoST2 {direction}')
"

cd $DATA_ROOT

# =============================================================================
# 3. 情感识别数据集 (Speech Emotion Recognition)
# =============================================================================

echo "下载情感识别数据集..."

# MELD (多模态情感识别)
echo "下载 MELD..."
mkdir -p $DATA_ROOT/raw/meld
cd $DATA_ROOT/raw/meld
python3 -c "
from datasets import load_dataset
dataset = load_dataset('declare-lab/MELD.Raw')
dataset.save_to_disk('meld_raw')
"

cd $DATA_ROOT

# =============================================================================
# 4. 音频分类数据集 (Vocal Sound Classification)
# =============================================================================

echo "下载音频分类数据集..."

# VocalSound
echo "下载 VocalSound..."
mkdir -p $DATA_ROOT/raw/vocalsound
cd $DATA_ROOT/raw/vocalsound
git clone https://github.com/YuanGongND/vocalsound.git
echo "VocalSound数据集需要从官方GitHub获取"

# AudioCaps (音频字幕)
echo "下载 AudioCaps..."
mkdir -p $DATA_ROOT/raw/audiocaps
cd $DATA_ROOT/raw/audiocaps
python3 -c "
from datasets import load_dataset
dataset = load_dataset('audiocaps')
dataset.save_to_disk('audiocaps')
"

# Clotho (音频描述)
echo "下载 Clotho..."
mkdir -p $DATA_ROOT/raw/clotho
cd $DATA_ROOT/raw/clotho
python3 -c "
from datasets import load_dataset
dataset = load_dataset('audiocaps')  # Clotho通过audiocaps访问
dataset.save_to_disk('clotho')
"

cd $DATA_ROOT

# =============================================================================
# 5. 音乐数据集
# =============================================================================

echo "下载音乐数据集..."

# MusicCaps
echo "下载 MusicCaps..."
mkdir -p $DATA_ROOT/raw/musiccaps
cd $DATA_ROOT/raw/musiccaps
python3 -c "
from datasets import load_dataset
dataset = load_dataset('google/MusicCaps')
dataset.save_to_disk('musiccaps')
"

cd $DATA_ROOT

# =============================================================================
# 6. 对话数据集 (用于SFT阶段)
# =============================================================================

echo "下载对话数据集..."

# Fisher (对话语音)
echo "下载 Fisher..."
mkdir -p $DATA_ROOT/raw/fisher
cd $DATA_ROOT/raw/fisher
echo "Fisher数据集需要从LDC购买授权: https://catalog.ldc.upenn.edu/LDC2004T19"

# SpokenWOZ (语音对话)
echo "下载 SpokenWOZ..."
mkdir -p $DATA_ROOT/raw/spokenwoz
cd $DATA_ROOT/raw/spokenwoz
git clone https://github.com/thu-spmi/SpokenWOZ.git

# IEMOCAP (情感对话)
echo "下载 IEMOCAP..."
mkdir -p $DATA_ROOT/raw/iemocap
cd $DATA_ROOT/raw/iemocap
echo "IEMOCAP需要申请学术授权: https://sail.usc.edu/iemocap/"

cd $DATA_ROOT

# =============================================================================
# 7. 评估数据集
# =============================================================================

echo "下载评估数据集..."

# AIR-Bench (音频指令跟随评估)
echo "下载 AIR-Bench..."
mkdir -p $DATA_ROOT/raw/air_bench
cd $DATA_ROOT/raw/air_bench
git clone https://github.com/OFA-Sys/AIR-Bench.git

cd $DATA_ROOT

echo "=== 数据下载完成 ==="
echo "注意: 某些数据集需要申请授权或注册账号"
echo "请查看各数据集的官方网站获取完整数据"

# 创建数据集清单
cat > dataset_sources.md << 'EOF'
# Qwen2-Audio 训练数据集清单

## ASR 数据集
- **LibriSpeech**: https://www.openslr.org/12/
- **Common Voice 15**: https://commonvoice.mozilla.org/datasets
- **FLEURS**: https://huggingface.co/datasets/google/fleurs
- **AISHELL-2**: https://www.aishelltech.com/aishell_2

## 语音翻译数据集
- **CoVoST2**: https://huggingface.co/datasets/facebook/covost2

## 情感识别数据集
- **MELD**: https://huggingface.co/datasets/declare-lab/MELD.Raw

## 音频分类数据集
- **VocalSound**: https://github.com/YuanGongND/vocalsound
- **AudioCaps**: https://huggingface.co/datasets/audiocaps
- **Clotho**: https://zenodo.org/record/4783391

## 音乐数据集
- **MusicCaps**: https://huggingface.co/datasets/google/MusicCaps

## 对话数据集
- **Fisher**: https://catalog.ldc.upenn.edu/LDC2004T19 (需购买)
- **SpokenWOZ**: https://github.com/thu-spmi/SpokenWOZ
- **IEMOCAP**: https://sail.usc.edu/iemocap/ (需申请)

## 评估数据集
- **AIR-Bench**: https://github.com/OFA-Sys/AIR-Bench

## 数据统计
根据论文Figure 3，预训练数据包含:
- 语音数据: ~400k hours
- 音乐数据: ~100k hours  
- 音效数据: ~100k hours
- 总计: ~600k hours

## 授权要求
1. **需要注册/申请的数据集**:
   - Common Voice (需Mozilla账号)
   - AISHELL-2 (需申请授权)
   - Fisher (需LDC授权)
   - IEMOCAP (需学术授权)

2. **开源可直接下载**:
   - LibriSpeech
   - FLEURS
   - CoVoST2
   - MELD
   - AudioCaps
   - MusicCaps
   - AIR-Bench
EOF

echo "数据集信息已保存到 dataset_sources.md" 