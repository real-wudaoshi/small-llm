# Small LLM

一个基于 PyTorch 的小型语言模型实验项目，覆盖从分词器训练、预训练到推理和简单评估的完整流程，适合学习 Transformer 类模型的最小可运行链路。

## 功能概览

- 在 TinyStories 数据集上训练 BPE 分词器（输出 `tokenizer.json`）
- 训练小型 Transformer 语言模型（输出 `model.pth`）
- 交互式文本续写（KV Cache + top-k/top-p 采样）
- 计算语料 token 的 unigram 熵与困惑度
- 打印模型参数统计和结构摘要

## 项目结构

```text
small-llm-main/
├─ config.py            # 特殊 token 与模型超参数
├─ model.py             # 模型结构、RoPE、注意力、KV Cache、采样生成
├─ train_tokenizer.py   # 训练 BPE 分词器
├─ pretrain.py          # 预训练主脚本（accelerate）
├─ test.py              # 交互式推理
├─ evaluate.py          # 统计 unigram entropy/perplexity
├─ info.py              # 参数量统计 + torchinfo summary
└─ utils.py             # 工具函数（如 state_dict 清理）
```

## 环境准备

建议使用 Python 3.10+。

```bash
pip install torch accelerate transformers datasets tokenizers tqdm tensorboard torchinfo
```

如果希望使用 GPU，请安装 CUDA 版本的 PyTorch，并确保：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出为 `True`。

## 快速开始

### 1. 训练分词器

```bash
python train_tokenizer.py
```

执行完成后会生成 `tokenizer.json`。

### 2. 预训练模型

```bash
python pretrain.py --epoch 1 --batch_size 32 --lr 5e-5 --num_proc 4
```

训练期间会输出：

- `model.pth`：模型权重（每 1000 step 保存，epoch 结束也会保存）
- `logs/`：TensorBoard 日志

查看训练曲线：

```bash
tensorboard --logdir logs
```

### 3. 交互式推理

```bash
python test.py
```

脚本会默认读取当前目录下的 `tokenizer.json` 和 `model.pth`，然后进入交互输入模式。

可选参数：

```bash
python test.py --device cpu
```

> 注意：当前 `test.py` 实际会优先根据 `torch.cuda.is_available()` 自动选择设备。

### 4. 评估 token 统计

```bash
python evaluate.py
```

输出：

- `unigram_entropy`
- `unigram_perplexity`

### 5. 查看模型参数信息

```bash
python info.py
```

输出包括去重前后参数量、重复绑定统计以及 `torchinfo.summary(model)`。

## 关键配置（`config.py`）

### 特殊 token

- `<pad> <bos> <eos> <unk> <ins> <gpt> <usr>`

### 词表与上下文

- `MAX_VOCAB_SIZE = 2048`
- `WINDOW_SIZE = 512`
- `INFERENCE_WINDOW_SIZE = 2048`

### 模型规模

- `DIM = 512`
- `FFN_DIM = 1024`
- `HEADS = 8`
- `LAYERS = 12`

## `pretrain.py` 常用参数

- `--epoch`：训练轮数（默认 `1`）
- `--batch_size`：训练 batch size（默认 `32`）
- `--lr`：学习率（默认 `5e-5`）
- `--weight_decay`：权重衰减（默认 `0.01`）
- `--num_proc`：数据处理并行数（默认 `4`）
- `--warmup_ratio`：学习率预热比例（默认 `0.05`）
- `--map_batch_size`：`dataset.map` 批大小（默认 `1024`）

小显存参考：

```bash
python pretrain.py --epoch 1 --batch_size 8 --lr 3e-5 --num_proc 2
```

## 运行依赖关系

推荐按以下顺序执行：

1. `python train_tokenizer.py`
2. `python pretrain.py ...`
3. `python test.py` 或 `python evaluate.py`

其中：

- `pretrain.py`、`test.py`、`evaluate.py` 都依赖 `tokenizer.json`
- `test.py` 还依赖 `model.pth`

## 常见问题

### 1) `Torch not compiled with CUDA enabled`

原因：安装的是 CPU 版 PyTorch，但代码尝试用 CUDA。

解决方式：

- 使用 CPU 运行（例如 `python test.py --device cpu`）
- 或重新安装 CUDA 版 PyTorch

### 2) `Vocabulary size mismatch`

`pretrain.py` 会检查分词器词表大小是否等于 `config.MAX_VOCAB_SIZE`。不一致时请重新训练分词器：

```bash
python train_tokenizer.py
```

并确认 `config.py` 与 `tokenizer.json` 是同一套配置生成的。

### 3) `tokenizer.json` 或 `model.pth` 不存在

- 缺 `tokenizer.json`：先运行 `python train_tokenizer.py`
- 缺 `model.pth`：先运行 `python pretrain.py ...`

## 说明

该项目偏教学与实验用途，适合作为小模型训练流程模板。若用于更严肃训练任务，可继续扩展：

- 梯度累积、断点恢复和更稳定的 checkpoint 策略
- 更完整的验证指标与评测集管理
- 更高效的数据预处理和并行流水线
- 推理侧批量生成与多样化采样控制

