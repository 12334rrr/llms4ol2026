# LLMs4OL 2026 - Flagship Task: End-to-End Ontology Learning

## 🏆 Competition Solution: Qwen2.5-7B + QLoRA Fine-tuning

基于 Qwen2.5-7B-Instruct + QLoRA 4-bit 微调的端到端本体学习方案，适配 RTX 3090 24GB。

## 📊 数据集分析

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| 样本数 | 4,303 | 2,018 |
| 总三元组数 | 55,606 | - |
| 平均三元组/样本 | 12.9 | - |
| 唯一谓词类型 | 28 种 | - |
| 上下文平均长度 | ~1,757 chars | - |

### 谓词分布 (Top 10)

| 谓词 | 数量 | 占比 |
|------|------|------|
| `is-a` | 49,532 | 89.1% |
| `instance-of` | 4,876 | 8.8% |
| `is defined by` | 351 | 0.6% |
| `equivalent class` | 284 | 0.5% |
| `disjoint with` | 261 | 0.5% |
| `type` | 114 | 0.2% |
| `exact match` | 40 | 0.1% |
| `tree view` | 28 | <0.1% |
| `part_of` | 25 | <0.1% |
| `has part` | 22 | <0.1% |

## 🚀 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
# 使用 Qwen2.5-7B-Instruct (推荐)
python train_ontology.py \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-4 \
    --lora_r 64 \
    --output_dir ./output/ontology_qwen7b_lora

# 使用 Qwen2.5-3B-Instruct (更快实验，效果稍差)
python train_ontology.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --epochs 5 \
    --batch_size 8 \
    --lr 2e-4
```

**预计训练时间 (RTX 3090 24GB):**

| 模型 | 时间 | 显存 |
|------|------|------|
| Qwen2.5-7B-Instruct | ~3-5 小时 | ~12-14 GB |
| Qwen2.5-3B-Instruct | ~1-2 小时 | ~8-10 GB |

### 推理生成提交文件

```bash
# 基础推理 (单次生成)
python predict_ontology.py \
    --model_path ./output/ontology_qwen7b_lora/final_model \
    --output submission.json

# 自一致性集成 (3次生成取投票 - 推荐，更稳健)
python predict_ontology.py \
    --model_path ./output/ontology_qwen7b_lora/final_model \
    --output submission_sc3.json \
    --num_sc 3

# Zero-shot 基线 (不加载微调权重)
python predict_ontology.py \
    --no_lora \
    --output submission_zeroshot.json
```

### 本地评估

```bash
# 在训练集上自我评估 (快速检查)
python evaluate_local.py data/train_task_a.json submission.json
```

## 🐳 Docker 部署 (服务器运行)

### 快速开始

```bash
# 1. 构建镜像
bash scripts/docker_run.sh build

# 2. 检查 GPU
bash scripts/docker_run.sh gpu-check

# 3. 训练
bash scripts/docker_run.sh train --gpu auto --epochs 3

# 4. 推理
bash scripts/docker_run.sh predict --num_sc 3

# 5. 交互调试
bash scripts/docker_run.sh shell
```

### Docker Compose (推荐)

```bash
# 训练
docker-compose build
docker-compose up train

# 推理
docker-compose up predict

# GPU 诊断
docker-compose run --rm gpu-check
```

### 手动 docker run

```bash
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/output:/app/output \
    -e HF_ENDPOINT=https://hf-mirror.com \
    llms4ol:latest \
    python train_ontology.py --gpu auto
```

### 服务器准备工作

```bash
# 1. 安装 NVIDIA Container Toolkit
#    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# 2. 安装 Docker + Docker Compose
curl -fsSL https://get.docker.com | bash

# 3. 克隆项目
git clone <your-repo-url>
cd LLMs4ol2026

# 4. 构建 + 运行
docker-compose build
docker-compose up train
```

## 📁 项目结构

```
├── src/                        # 核心代码库
│   ├── config.py               # 全局配置 + GPU自动检测
│   ├── data_utils.py           # 数据加载、ChatML格式化
│   ├── model_utils.py          # 模型加载 (4-bit → fp16回退)
│   ├── post_process.py         # 图后处理
│   └── report.py               # 🆕 训练报告生成
├── configs/                    # GPU 预设配置
│   ├── a100_40gb.json          (Qwen2.5-14B)
│   ├── v100_32gb.json          (Qwen2.5-7B)
│   ├── t4_16gb.json            (Qwen2.5-7B)
│   └── rtx3090_24gb.json       (Qwen2.5-7B)
├── utils/                      # 诊断工具
│   ├── check_env.py            # 完整环境检查
│   └── check_gpu.py            # GPU + 模型推荐
├── scripts/
│   └── docker_run.sh           # 🆕 Docker 快速启动
├── Dockerfile                  # 🆕 Docker 镜像
├── docker-compose.yml          # 🆕 Docker Compose
├── .dockerignore               # 🆕 Docker 排除规则
├── .gitignore                  # 🆕 Git 排除规则
├── train_ontology.py           # 训练入口
├── predict_ontology.py         # 推理入口
├── evaluate_local.py           # 评估入口
├── main.py                     # 统一入口
├── requirements.txt
└── README.md
```

## 🔧 核心策略

### 1. QLoRA 4-bit 微调
- **Base Model**: Qwen2.5-7B-Instruct (ChatML 原生支持)
- **LoRA 配置**: r=64, alpha=128, dropout=0.05
- **目标模块**: Q/K/V/O + Gate/Up/Down 投影层 (全线性层)
- **显存占用**: ~12-14GB (RTX 3090 24GB 充足)
- **有效 Batch Size**: 4 × 4 = 16 (gradient accumulation)
- **最大序列长度**: 4096 tokens (~P90 样本完整覆盖)

### 2. 低频谓词过采样 (5x)
- 出现次数 < 200 的 23 种谓词所在样本被 5x 过采样
- 涵盖 107 个样本，过采样后增加 428 个训练样本
- 确保模型学会所有 28 种谓词，不只是 `is-a`

### 3. 图后处理
- **去重**: 大小写不敏感的三元组去重
- **自环检测**: 移除 subject == object 的无效三元组
- **环检测与修复**: DFS 检测 is-a 层次中的环，打破最小环边
- **格式清理**: 统一空格、移除空值

### 4. 自一致性集成 (推荐)
- 多次生成 (temperature=0.3) 取投票结果
- 默认阈值 0.5 (过半数通过)
- `--num_sc 3` 即可获得稳健提升

## 📈 评估指标

比赛使用 **Graph Similarity Metric** (0-1):

```
Graph Similarity = (Edge F1 + Neighborhood Similarity + Taxonomy Similarity) / 3
```

| 子指标 | 描述 | 计算方式 |
|--------|------|---------|
| Edge F1 | 精确三元组匹配 | Precision/Recall/F1 over full triples |
| Neighborhood Similarity | 局部结构相似度 | Jaccard of (predicate, object) per node |
| Taxonomy Similarity | 分类层次相似度 | Jaccard of is-a ancestor sets |

## ⚙️ 硬件要求

- **GPU**: NVIDIA RTX 3090/4090 (24GB+)
- **RAM**: 32GB+
- **存储**: ~20GB (模型 + 数据)
- **OS**: Windows (WSL2 推荐) / Linux

## 🔬 进阶优化方向

1. **多模型集成**: 同时训练多个模型 (Qwen2.5-7B, LLaMA-3-8B, Mistral-7B) 并投票
2. **迭代自改进**: 让模型检查自己的输出 → 修复不一致 → 再检查
3. **领域感知 Prompt**: 从标题提取领域信息，动态调整 prompt
4. **RAG 增强**: 检索外部本体知识辅助推理
5. **数据增强**: 用 LLM 生成更多样化的训练样本
6. **约束解码**: 限制模型只能输出合法的 JSON 三元组格式
