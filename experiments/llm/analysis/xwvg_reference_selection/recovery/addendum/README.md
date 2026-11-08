# XWVG 数据恢复与续跑补充包

## 为什么需要这个补充包

原执行包假定可以取得旧的逐题概率缓存，并规定默认不做新推理。实际 Codex 环境中没有找到这些缓存，因此实现和构造数据检查完成后，实测停了下来。这个前提没有落实，是原方案的缺口。

本包处理这一缺口。保留已经完成的实验实现与检查，先定位原数据，再恢复可重建的题目文件；若确实拿不到原概率输出，则补采同一协议所需的评分，再接回现有分析程序。

**52 项组件测试、7 项附件测试、构造数据上的全量运行，是用户转贴的执行结果。本会话没有访问到那台机器上的报告或新实现，不能声称已复核它们。** 本包也没有运行真实模型评分。

## 两个缺失文件分别是什么

| 文件 | 内容 | 恢复方式 |
|---|---|---|
| `mmlu_pro_selected_560.jsonl` | 原实验选中的 560 道题及答案 | 从固定版本的数据集、原抽样代码重新生成，核对已有 SHA-256 |
| `permutation_averaged_semantic_scores.parquet` | 八个模型在每题、每条件下的完整选项概率 | 找回旧文件；由完整轮换缓存聚合；或重新运行模型评分 |

题目文本可以确定性重建。模型历史输出不能由论文均值表或题目文本还原。

## 交给 Codex 的入口

先读 `CODEX_CONTINUE.md`。本包的指令补充原执行包的数据取得步骤，并替换其“默认禁止新推理，因此缓存缺失就结束”的限制。实际提交本包执行时，任务包括必要的定向评分；不要求重跑整篇论文。

包内工具：

- `tools/locate_inputs.py`：只读检查原始路径、指定目录和相关 ZIP 的目录清单。记录搜索范围、权限问题和截断情况。文件名命中不等于内容验证通过。
- `tools/recover_selected_questions.py`：复用原归一化与抽样函数，恢复题目文件并核对归档校验值；不运行模型。
- `docs/FRESH_SCORE_ROUTE.md`：旧概率无法取得时，最小评分补采任务及实现依据。

定位工具只依赖 Python 标准库。题目恢复需要 `numpy` 和 `datasets`，优先使用现有兼容环境；依赖见 `requirements-recovery.txt`。固定版本的数据集若未缓存，需要下载该数据集；这一步不下载模型权重。

在本补充包目录中，可先运行下面两个现有命令。请由 Codex 按实际路径填写输出位置：

```bash
python3 tools/locate_inputs.py \
  --repo-root /work/Users/leiyo/GitHub/behavioral-coverage-audit \
  --out /path/to/new/recovery/input_discovery.json

python3 tools/recover_selected_questions.py \
  --repo-root /work/Users/leiyo/GitHub/behavioral-coverage-audit \
  --output /path/to/new/recovery/mmlu_pro_selected_560.jsonl
```

`--root` 可重复提供原实验数据盘或下载目录。定位报告拒绝覆盖已有文件，请为新的搜索使用新的报告路径。新 GPU 评分 driver 的命令接口在说明中明确标为待实现，不应当作本包已经提供的可执行脚本。

## 已核对的数据位置

**优先查找旧 V2.3 结果 ZIP。** 打包代码把 `outputs/analysis` 和 `outputs/validation` 整个收入 ZIP，因此它应包含 averaged 和 rotation-wise 概率缓存。状态里的 `raw_scores_omitted: true` 只说明原始评分分片未打包，不代表这两个分析缓存被排除。还需实际检查找到的 ZIP 成员。

```text
文件名：PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION_RESULTS.zip
历史大小：109877741 bytes（约 110 MB）
历史 SHA256：ba24123066a16aa29af60fa924589256fd7fe8566ecd9a2017f12dccaeb682a1
```

ZIP 内的对应路径是 `..._RESULTS/analysis/permutation_averaged_semantic_scores.parquet` 和 `..._RESULTS/validation/permutation_wise_semantic_scores.parquet`。它们不保留外部运行目录中的 `outputs/` 这一层，因此只按原完整路径搜索会漏掉解压后的结果包。详情见 `evidence/recovery_sources.json`。

源码和归档记录的历史实验位于下面的用户目录中；当前 GitHub 工作副本是另一个目录：

```text
/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/
/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION/
/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/
/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION/
```

路径来自原始记录，**并不表示这些目录现在仍然存在或已挂载**。优先查看原实验机器及其数据盘。程序必须记录它实际查过哪里，不能把“当前环境看不到”写成“数据已经丢失”。

## 完成目标

沿用已经实现的 160 个设置、10,240 个候选及 2,000 次描述性重采样。取得真实输入后，输出实测结果表、主图和据此撰写的 XWVG 回复草稿。结果是否改善与任务是否执行完成分别记录。
