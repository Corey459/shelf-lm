# ShelfLM v2

**ShelfLM** 是一套**自研混合架构**的中文小/中型语言模型实验：不是「再包一层标准 Transformer」，而是把序列混合、稀疏注意力、压缩事实表与分层 MoE 重新组合后的训练代码开源，方便一起学习、复现与改进。

> 状态：研究 / 学习向。当前公开的是**模型定义 + 训练管线 + 词表工具**；完整预训练权重较大，需自行训练或后续 Release。预训练中期抽测仍会胡编/算错，**还不能当生产助手**。

English summary: custom hybrid LM (StateMix + sparse causal attention + FactTable + ShelfMoE), ~300M target, SentencePiece 24 576. See [ARCHITECTURE_NOTES.md](ARCHITECTURE_NOTES.md).

---

## 自研架构怎么「新」

相对常见的 GPT 式「每层全量 Causal Attention + Dense FFN」：

| 模块 | 常见做法 | ShelfLM v2 |
|------|----------|------------|
| 序列混合 | 每层 O(T²) Attention | **StateMix**（门控对角递推，O(T) 固定状态）为默认混合器 |
| Attention | 全层都有 | **每 `attn_every` 层**插一层因果 MHA（默认每 4 层） |
| FFN | Dense / 常规 MoE | **ShelfMoE**：共享 SwiGLU + shelf→expert 两级路由 + 低秩 expert delta |
| 额外记忆 | 通常无 | **FactTable**：bigram hash → 低秩表（目标 &lt;1% 参数） |
| 词表 | 大 BPE / 字级 | SentencePiece BPE **24 576** |

设计意图可以概括成一句话：**用线性递推扛日常混合，用少量 Attention 补长程，用压缩 FactTable 塞常识槽位，用分层低秩 MoE 扩容量但不把参数打满。**

细节与公式级说明见 **[ARCHITECTURE_NOTES.md](ARCHITECTURE_NOTES.md)**（客观技术笔记，非营销文）。

### 前向结构（概览）

1. Token Embedding（SPM）  
2. FactTable 注入（可选）  
3. `n_layers` 个 Block：每层 **StateMix + ShelfMoE**；每隔 N 层再加 **CausalSelfAttn**  
4. RMSNorm +（可选）tied LM head  

默认 ~300M：`d_model=1024`，`n_layers=18`，`n_heads=16`，`attn_every=4`（见 `configs/shelf_300m.json`）。另有 `shelf_tiny`（冒烟）与 `shelf_1b`（路线图）。

---

## 仓库内容

```
model_v2.py              # 架构实现
train_v2.py              # 预训练 / SFT 训练器
ARCHITECTURE_NOTES.md    # 架构说明
configs/                 # tiny / 300m / 1b
vocab/                   # SPM 训练与已提供的 24k 词表
data/                    # 语料工具 + 小样例（无公司私有数据）
scripts/                 # 参数统计、CPU smoke 等
examples/                # 相对路径启动示例
```

**刻意未包含：** 训练权重 `.pt`、完整大规模语料、实验室内网/跳板运维脚本、公司域名私有数据。清单见 `MANIFEST.txt`。

---

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU：按本机 CUDA 安装对应 torch，例如
# pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 快速开始（CPU 冒烟）

```bash
python scripts/count_params.py --config configs/shelf_300m.json

python train_v2.py \
  --config configs/shelf_tiny.json \
  --spm vocab/out/spm_zh_24576.model \
  --data data/seed_zh.jsonl data/samples/extra_zh_short.sample.jsonl \
  --out runs/tiny_smoke --steps 50 --device cpu --batch 2 --seq 64
```

单卡 ~300M 预训练示例见 `examples/run_train_example.sh`（用环境变量改数据路径与 batch）。

## 可选：自训词表

```bash
python data/prepare_corpus.py \
  --inputs data/seed_zh.jsonl data/samples/opensource_firefly.sample.jsonl \
  --out-txt runs/corpus_zh.txt --limit 50000

python vocab/train_spm.py --corpus runs/corpus_zh.txt --out-dir vocab/out --vocab-size 24576
```

## 数据说明

- `data/seed_zh.jsonl` 与 `data/samples/*.sample.jsonl` 仅为**演示用短样本**。  
- 更大开源语料可通过 `data/stream_pretrain.py` 自行从 Hugging Face / 镜像拉取。  
- **请勿**把私有/未授权语料直接推回本仓库。


## 预训练权重（GitHub Release）

完整 `model.pt`（约 3.5 GB）不进 git 主仓，放在 Release：

**[v0.1.0-zhongce-step330900](https://github.com/Corey459/shelf-lm/releases/tag/v0.1.0-zhongce-step330900)**

- tag `v2_300m_zhongce`，seed 42，step **330900**
- 下载 `model.pt.part-aa` + `model.pt.part-ab` 后拼接：

```bash
cat model.pt.part-aa model.pt.part-ab > model.pt
sha256sum -c model.pt.sha256
# expected: 25ffc06299446c087c4f677e17ce5e889c07664168a6f1a8850bd70a3e7b190d
```

## 已知局限（诚实版）

- 预训练中期对话抽测：偶发答对常识（如首都、月份），算术/翻译/开放题仍不稳定。  
- 「能聊」通常还需要**指令 SFT**；本仓提供训练入口，不附带成品助手权重。  
- v2 checkpoint **不能**与旧版字级 / 大 FactTable 的 v1 热启混用。

## 贡献与交流

Issue / Discussion 欢迎：复现报告、架构质疑、训练曲线、负样本（胡编案例）都很有价值。PR 请保持仓库「无密钥、无内网、无私有语料」。

## License

[MIT](LICENSE)
