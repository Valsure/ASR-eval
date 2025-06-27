# ASR评测

本项目用于对音频转文本系统的转写质量进行系统评估，分为三个阶段：
1. **转写阶段（Inference）**：调用多模态模型进行音频转文本。
2. **评判阶段（Judge）**：通过大模型评判转写文本的准确性，输出是否成功及提取内容。
3. **评估阶段（WER）**：计算词错误率（WER）统计最终表现。

支持中英文评估，兼容 Qwen 系列多模态模型与 VLLM 推理加速。

---

## 📦 项目结构说明

```
.
├── prompt.txt # 提示词模板
├── results/   # 中间/最终评估结果输出目录
├── test.sh    # 批量评测自动化脚本
├── test_set_demo.jsonl    # 示例测试集格式
└── ASR-eval.py    # 主入口脚本
```

---

## 🚀 使用方法

### 1. 准备音频数据（JSONL 格式）

每行一个样本，结构如下：

```json
{
  "audios": ["/path/to/audio.wav"],
  "messages": [
    {"role": "user", "content": "What does the audio say?"},
    {"role": "assistant", "content": "The expected transcription text."}
  ]
}
```

### 2. 运行完整评估流程

```bash
python main.py \
    --model_path /path/to/Qwen2.5-Omni \
    --model_name qwen2_5_omni_eval \
    --test_data /path/to/test.jsonl \
    --prompt_path ./prompt.txt \
    --is_Chinese \
    --stage all
```

### 3. 分阶段运行（可选）

- 仅推理：

  ```bash
  python main.py --stage inference ...
  ```

- 仅评判：

  ```bash
  python main.py --stage judge ...
  ```

- 仅WER评估：

  ```bash
  python main.py --stage wer ...
  ```

---

## ⚙️ 参数说明

| 参数 | 说明 |
|------|------|
| `--model_path` | 用于转录阶段的 Qwen2.5 模型路径 |
| `--model_name` | 模型名称前缀（用于结果文件命名） |
| `--test_data` | 待评估的测试数据（jsonl） |
| `--prompt_path` | 用于评判阶段的 prompt 模板 |
| `--is_Chinese` | 是否处理中文数据（会影响 WER 的处理逻辑） |
| `--stage` | 运行阶段：`inference` / `judge` / `wer` / `all` |

---

## ✅ 评估结果文件说明

所有中间/最终结果会输出至 `results/` 目录：

| 文件 | 内容 |
|------|------|
| `*_inference.csv` | 模型的原始输出结果 |
| `*_judge.csv` | vLLM模型提取的评判结果 |
| `*_wer.csv` | 成功提取样本的 WER 值 |
| `*_final_result.csv` | 整体统计（平均WER/成功数量等） |
| `*_bad_case.csv` | 无法成功提取的失败样本 |

---

## ⚡️ 特性

- ✅ 支持 VLLM 高性能推理
- ✅ 支持 Qwen2.5 多模态模型音频转写
- ✅ 支持中英文评估
- ✅ 自动跳过已处理样本，断点续跑
- ✅ 批量推理控制（默认 batch size = 3）

---

## 📌 环境依赖

- Python >= 3.8
- `torch`, `transformers`, `jiwer`, `soundfile`, `vllm`, `tqdm`, `pandas`

建议使用以下命令创建虚拟环境并安装依赖：

```bash
pip install -r requirements.txt
```

---


