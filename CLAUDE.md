# CLAUDE.md

DASH（Dynamic Analysis of Single-molecule FRET data with Hidden Markov Models）是一个端到端的 smFRET 数据分析管线，将 LSTM-CRF 深度学习分割器与网格搜索 HMM 相结合，从原始荧光轨迹中自动识别构象状态。

## 管线结构

四步流程，每步都是 `pipeline/` 下的独立脚本：

| 步骤 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 0 | `step0_preprocess.py` | 原始 `.npz`（donor/acceptor 双通道） | `*_labeled.pkl` / `*_unlabeled.pkl` |
| 1 | `step1_train.py` | `data_train_labeled.pkl` | LSTM-CRF 模型 checkpoint（`output/ckpt_CRF/`） |
| 2 | `step2_evaluate_CRF.py` | 测试 PKL + checkpoint | 光漂白片段预测（`output/eval/`） |
| 3 | `step3_HMM_test_metric.py` | FRET PKL + 片段预测 | 状态分配 CSV + 预测 PKL + 总结图 PNG |

如果已有预训练 checkpoint（`output/ckpt_CRF/` 下），可跳过 step1。

## 常用命令

```bash
conda activate dash

# 完整流程
python pipeline/step0_preprocess.py
python pipeline/step1_train.py
python pipeline/step2_evaluate_CRF.py
python pipeline/step3_HMM_test_metric.py --batch

# step3 其他模式
python pipeline/step3_HMM_test_metric.py --list               # 列出可用 fret 文件
python pipeline/step3_HMM_test_metric.py --file as0101_0000   # 处理单个文件
```

Docker 运行：
```bash
docker run --gpus all --rm -v "${PWD}/output:/workspace/output" shuqizhou/dash:v1          # 完整 demo
docker run --gpus all --rm -v "${PWD}/output:/workspace/output" shuqizhou/dash:v1 step3 --batch  # 单步
```

## 配置

所有路径和超参数在 `config.yaml` 中配置。`config_loader.py` 负责路径解析。

本地运行时必须修改 `root` 及所有绝对路径（`train_data_path`、`test_data_path`、`fret_dir`、`pred_seg_dir`）。Docker 环境默认 `root: "/workspace"`。

### HMM 网格搜索参数（直接在 `step3_HMM_test_metric.py` 中修改）

- `N_MIX_VALUES`：每个隐态的高斯混合分量数（默认 `[2]`）
- `NSTATES_VALUES`：FRET 状态数搜索空间（默认 `[2, 3, 4, 5]`）
- `GLOBAL_THRESHOLD_VALUES` / `LOCAL_THRESHOLD_VALUES`：状态合并阈值

## 核心模块

- **`build_model/seq_model.py`**：LSTM-CRF 模型定义（`LSTM_CRF` 类），使用 `pytorch-crf` 的 CRF 层。输入为双通道归一化强度，输出 7 类 token。
- **`build_model/seq_dataset.py`**：`SequenceDataset` + `BucketBatchSampler`，按长度分桶减少 padding。
- **`utils/utils.py`**：FRET 计算（`preprocess_fret_cross`，cross=0.05）、状态合并（`_merge_close_states`，阈值 0.09）、驻留时间统计。
- **`utils/utils_hmm.py`**：`fit_hmm()` 封装 hmmlearn 的 `GMMHMM`，支持按 `length_info` 拼接多条轨迹全局拟合。
- **`utils/utils_fret_analyse.py`**：`StateAnalyzer` 类，状态合并和短驻留修正的高层接口。
- **`utils/utils_full_plot.py`**：所有绘图函数，包括 `plot_hmm_summary`。

## 7 类 Token 定义

```
0 = background（正常 FRET 帧）
1 = seg1_start（donor 漂白段起始）
2 = seg1_mid（donor 漂白段中间）
3 = seg1_end（donor 漂白段终止）
4 = seg2_start（acceptor 漂白段起始）
5 = seg2_mid（acceptor 漂白段中间）
6 = seg2_end（acceptor 漂白段终止）
```

## 输入数据格式

`.npz` 文件必须包含：
- `traces`：`float32 (N, T_max, 2)` — 列0=donor，列1=acceptor，短轨迹用零填充
- `segments`：`object (N,)` — 分段标注（无标签数据设为 None）
- `metadata`：JSON 字符串，含每条轨迹的 `trace_length`

## FRET 效率计算

```python
fret = (Cy5 - 0.05*Cy3) / (Cy5 - 0.05*Cy3 + Cy3)  # crosstalk 校正因子 0.05
mask = (fret > -0.21) & (fret <= 1.21)               # 异常值过滤
```

## 依赖

Python 3.10，PyTorch 2.0.1+cu118，hmmlearn>=0.3.0，numpy<2.0。完整列表见 `requirements.txt`。

## 输出文件

| 文件 | 说明 |
|------|------|
| `hmm_result/step3_HMM_<name>.csv` | 所有 HMM 参数组合的评分表 |
| `hmm_result/step3_HMM_final_predictions_<name>.pkl` | 逐片段状态分配 |
| `hmm_result/step3_HMM_summary_<name>.png` | 总结图（直方图 + 占据率 + 轨迹） |
| `output/logs/step3_hmm_<timestamp>.log` | 运行日志 |
