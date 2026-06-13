# ABR 项目新手上手与源码学习指南

本文面向“第一次接触这个项目的人”。目标不是只告诉你怎么跑一个命令，而是把项目的研究问题、代码结构、数据流、模型流、实验流和常见坑讲清楚。读完后，你应该能回答下面几个问题：

1. 这个项目到底解决什么问题？
2. 每个目录和文件分别负责什么？
3. 从网络轨迹到最终模型输出码率，中间发生了什么？
4. baseline、经验池、NetLLM 训练和测试之间是什么关系？
5. 从 GitHub 重新拉一份代码后，哪些东西能直接跑，哪些大文件需要重新准备？

本文以当前仓库 `/data3/wangxh/NetLLM-master/adaptive_bitrate_streaming` 的真实代码为准。

## 0. 先给新手的总图

顶层仓库 `NetLLM-master` 是 NetLLM 论文代码仓库的一个改造版本。它包含三个网络任务：

| 子目录 | 任务 | 本文关注程度 |
|---|---|---|
| `adaptive_bitrate_streaming/` | ABR，自适应码率视频流 | 本文主线 |
| `cluster_job_scheduling/` | 集群作业调度 | 只作为顶层背景 |
| `viewport_prediction/` | 视口预测 | 只作为顶层背景 |

你当前真正做论文和实验的主项目是 `adaptive_bitrate_streaming/`。

这个 ABR 项目可以先记成一句话：

> 在变化的网络带宽下，模型每 4 秒为下一个视频块选择一个码率，使视频尽量清晰、尽量不卡顿、码率切换尽量平滑。

代码里有两条主线：

| 主线 | 入口 | 作用 |
|---|---|---|
| baseline 主线 | `run_baseline.py` | 运行 BBA、MPC、Genet、UDR 等传统或强化学习方法 |
| NetLLM 主线 | `generate_exp_pool.py` + `run_plm.py` | 先生成离线经验池，再用大模型学习 ABR 决策 |

最重要的流程是：

```text
网络 trace + 视频 chunk 大小
        |
        v
ABR 仿真环境 Environment
        |
        v
baseline 策略和环境交互
        |
        v
经验池 exp_pool.pkl
        |
        v
ExperienceDataset 切成固定窗口样本
        |
        v
状态编码器 + 大语言模型 + action head
        |
        v
训练得到 LoRA adapter 和非 PLM 模块参数
        |
        v
在同一个 ABR 环境里测试，输出结果日志和 mean_reward
```

如果你是纯新手，建议先把这条链路背下来，再去读代码。否则一上来读 `run_plm.py` 会很容易被参数淹没。

## 1. ABR 问题到底是什么

ABR 是 Adaptive Bitrate Streaming，自适应码率视频传输。

在线视频通常不是一次性下载完整视频，而是把视频切成很多小块，英文里叫 chunk。这个项目里：

| 概念 | 当前代码中的值 |
|---|---|
| 一个 chunk 时长 | `4` 秒 |
| 一个 episode 的 chunk 数 | `48` |
| 一个 episode 总视频时长 | `48 * 4 = 192` 秒 |
| 可选码率档位数 | `6` 档 |
| 码率列表 | `[300, 750, 1200, 1850, 2850, 4300] Kbps` |

每下载一个 chunk 前，客户端要选择一个码率。码率越高，画质越好，但是文件更大，下载更慢，容易造成 buffer 耗尽和卡顿。码率越低，不卡顿风险小，但是画质差。频繁从高码率跳到低码率也会影响观看体验。

所以 ABR 决策本质上是一个序列决策问题。每一步模型都要平衡三件事：

1. 选高码率，提高画质。
2. 控制卡顿，避免下载时间超过当前 buffer。
3. 控制码率波动，避免画质忽高忽低。

这也是代码中奖励函数的来源。

## 2. 项目目录怎么读

建议先从 `adaptive_bitrate_streaming/` 根目录看起。

```text
adaptive_bitrate_streaming/
├── README.md
├── config.py
├── run_baseline.py
├── generate_exp_pool.py
├── run_plm.py
├── run_qwen.sh
├── run_llama32.sh
├── artifacts/
├── data/
├── baseline_special/
├── plm_special/
├── 图片制作/
└── 论文书写/
```

每个部分的作用如下。

| 路径 | 作用 |
|---|---|
| `README.md` | 原始项目说明，包含环境和基本命令 |
| `config.py` | 全局路径配置，告诉代码 trace、video、模型和输出目录在哪里 |
| `run_baseline.py` | 测试 baseline 的主入口 |
| `generate_exp_pool.py` | 生成 NetLLM 训练用的经验池 |
| `run_plm.py` | NetLLM 训练和测试的主入口 |
| `run_qwen.sh` | 对 `run_plm.py` 的封装脚本，便于跑 Qwen/Llama 和方法变体 |
| `run_llama32.sh` | 对 `run_qwen.sh` 的再封装，默认使用本地 Llama3.2 small 路径 |
| `artifacts/exp_pools/` | 保存经验池，比如 `exp_pool.pkl` |
| `artifacts/results/` | 保存 baseline 和 NetLLM 测试结果 |
| `data/traces/` | 网络带宽轨迹数据 |
| `data/videos/` | 不同视频在不同码率下的 chunk 大小 |
| `data/all_models/` | baseline 的 TensorFlow checkpoint |
| `data/ft_plms/` | 训练后的 LLM adapter 和 checkpoint，本地有但 GitHub 默认不上传 |
| `baseline_special/` | Genet/Pensieve 环境、baseline 相关代码 |
| `plm_special/` | NetLLM 模型、数据集、训练、测试代码 |
| `图片制作/` | 论文图片生成脚本 |
| `论文书写/` | 论文正文、图片、表格和中间材料 |

初学者的阅读顺序建议是：

1. `README.md`
2. `config.py`
3. `baseline_special/utils/constants.py`
4. `baseline_special/env.py`
5. `run_baseline.py`
6. `generate_exp_pool.py`
7. `plm_special/data/exp_pool.py`
8. `plm_special/data/dataset.py`
9. `plm_special/models/state_encoder.py`
10. `plm_special/models/rl_policy.py`
11. `plm_special/trainer.py`
12. `plm_special/evaluate.py`
13. `plm_special/test.py`
14. `run_plm.py`

不要一开始就读 `run_plm.py` 的全部参数。它是总控文件，信息密度很高。先理解环境、状态、奖励、经验池，再回来看它，会顺很多。

## 3. GitHub 拉下来后哪些文件有用

从 GitHub 新拉一份代码时，要区分三类东西。

### 3.1 会上传到 GitHub，通常可以直接拿到的内容

这些是项目运行和理解的基础：

| 内容 | 当前状态 |
|---|---|
| 源代码 | 应该在 GitHub 上 |
| `data/traces/` | 带宽轨迹，小文件，当前被 Git 跟踪 |
| `data/videos/` | 视频 chunk 大小，小文件，当前被 Git 跟踪 |
| `data/all_models/` | baseline checkpoint，约 17M，当前被 Git 跟踪 |
| `artifacts/exp_pools/exp_pool.pkl` | 默认经验池，约 4.6M，当前被 Git 跟踪 |
| `artifacts/exp_pools/exp_pool_smoke.pkl` | 小型 smoke test 经验池，约 20K，当前被 Git 跟踪 |
| `artifacts/results/` | 一些测试结果，`.gitignore` 对这个目录做了例外保留 |

所以，新手从 GitHub 拉下来后，baseline 和经验池相关的小数据一般是齐的。

### 3.2 不会上传到 GitHub，但对复现训练或测试很重要的内容

这些是大文件，当前被 `.gitignore` 排除：

| 路径 | 内容 | 是否需要 |
|---|---|---|
| `../downloaded_plms/` | 本地下载的基础大模型，比如 Qwen、Llama3.2 | 训练和测试 NetLLM 需要 |
| `data/ft_plms/` | 训练后的 LoRA adapter、checkpoint、best model | 如果直接测试已有模型就需要 |

如果你从 GitHub 新拉代码后，只想看源码和跑 baseline，这两个目录不是必须的。  
如果你要训练或测试 NetLLM，就必须准备基础大模型；如果你要复现已有结果，还需要对应的 `data/ft_plms/.../early_stop_-1_best_model`。

### 3.3 不需要上传，也不影响运行的内容

| 路径 | 说明 |
|---|---|
| `__pycache__/` | Python 自动生成的缓存，运行时会重新生成 |
| `.vscode/settings.json` | 本机 VS Code 配置，通常只对当前电脑有意义 |

这两个从 GitHub 拉下来没有也完全正常。

## 4. 环境准备

这个项目最好准备两个 Python 环境，因为 baseline 和 NetLLM 的依赖年代差别很大。

### 4.1 baseline 环境

baseline 主要是 Genet/Pensieve 旧代码，依赖 TensorFlow 1.x。README 建议：

```bash
conda create -n abr_tf python=3.7
conda activate abr_tf
pip install tensorflow-gpu==1.15
pip install tensorboard==1.15.0
pip install tensorboard-plugin-wit==1.8.0
pip install tflearn==0.5.0
pip install numba==0.53.1
pip install gym==0.18.0
pip install stable-baselines[mpi]==2.10.1
pip install pandas==1.1.5
pip install tqdm==4.62.2
```

这个环境用于：

```text
run_baseline.py
generate_exp_pool.py
```

如果你看到 `ModuleNotFoundError: No module named 'tensorflow'`，大概率是用了 NetLLM 环境去跑 baseline。

### 4.2 NetLLM 环境

NetLLM 部分使用 PyTorch、Transformers、PEFT。README 中列出的关键依赖是：

```text
python==3.8.10
torch==2.1.0
numpy==1.24.4
munch==4.0.0
openprompt==1.0.1
transformers==4.34.1
peft==0.6.2
```

本机常用环境路径里有：

```bash
/data3/wangxh/conda-envs/abr_netllm/bin/python
```

这个环境用于：

```text
run_plm.py
plm_special/*
```

如果你看到 `ModuleNotFoundError: No module named 'torch'` 或 `No module named 'peft'`，大概率是用了 baseline 环境去跑 NetLLM。

### 4.3 推荐运行位置

建议所有 ABR 命令都从这个目录运行：

```bash
cd /data3/wangxh/NetLLM-master/adaptive_bitrate_streaming
```

原因是 `config.py` 会根据当前工作目录判断路径前缀。虽然代码也考虑了从仓库根目录运行的情况，但新手最好固定在 ABR 子目录，少踩路径坑。

## 5. 全局配置文件 `config.py`

`config.py` 是项目的路径地图。新手一定要看懂它。

关键配置如下：

| 配置 | 作用 |
|---|---|
| `baseline_model_paths` | Genet、UDR 等 baseline checkpoint 路径 |
| `trace_dirs` | 训练、验证、测试网络 trace 路径 |
| `video_size_dirs` | video1、video2 的 chunk size 路径 |
| `artifacts_dir` | 实验产物根目录 |
| `results_dir` | 测试结果目录 |
| `exp_pools_dir` | 经验池目录 |
| `plm_types` | 支持的大模型类型 |
| `plm_sizes` | 支持的大模型尺寸标签 |
| `plm_dir` | 基础大模型本地目录 |
| `plm_ft_dir` | 微调后模型保存目录 |
| `plm_embed_sizes` | 不同模型的 embedding 维度配置 |
| `plm_layer_sizes` | 不同模型的层数配置 |

最容易出问题的是 `plm_dir`。从 `adaptive_bitrate_streaming/` 目录运行时，基础模型默认应该在：

```text
../downloaded_plms/
```

例如：

```text
../downloaded_plms/qwen/base
../downloaded_plms/llama3.2/base
../downloaded_plms/llama_tiny_smoke/base
```

如果模型目录不存在，`run_plm.py` 会报 `Cannot find foundation model path`，这时需要传 `--plm-path`。

## 6. ABR 环境：`baseline_special/env.py`

环境是整个项目的地基。所有策略，无论是 BBA、MPC、Genet 还是 NetLLM，最终都要和同一个 `Environment` 交互。

环境接收一个动作：

```text
quality = 0, 1, 2, 3, 4, 5
```

然后返回：

```text
delay
sleep_time
buffer_size
rebuf
video_chunk_size
next_video_chunk_sizes
end_of_video
video_chunk_remain
```

你可以把它理解成一个播放器仿真器：

1. 当前模型说：“下一个 chunk 选码率 3。”
2. 环境去查 `video_size_3`，知道这个 chunk 有多大。
3. 环境沿着当前网络 trace 模拟下载过程。
4. 计算下载用了多久，是否卡顿，buffer 怎么变化。
5. 返回下一步决策需要的信息。

### 6.1 不是简单的 size / bandwidth

环境不是只用一个简单公式算下载时间。它会：

1. 根据 trace 中每个时间片的带宽逐步发送数据。
2. 使用 `PACKET_PAYLOAD_PORTION = 0.95` 估计有效载荷。
3. 加入 `LINK_RTT = 80 ms`。
4. 在 `fixed=False` 时给 delay 乘上 `[0.9, 1.1]` 的随机噪声。
5. 如果 buffer 超过 `BUFFER_THRESH = 60s`，会 sleep 一段时间。
6. 如果 trace 走到末尾，会回到开头继续播放。

所以它比“下载时间 = 文件大小 / 带宽”更接近网络仿真。

### 6.2 fixed 参数的意义

`Environment` 有一个重要参数 `fixed`：

| fixed | 行为 |
|---|---|
| `False` | trace 顺序打乱，delay 有随机噪声 |
| `True` | trace 顺序固定，delay 不加随机噪声 |

实验测试一般会使用 `--fixed-order`，这样每次结果更可复现。

## 7. 状态、动作、奖励

这是理解项目最重要的一节。

### 7.1 状态是 6 x 6 矩阵

状态来自 Pensieve/Genet 的经典 ABR 表示。代码里常量是：

```text
S_INFO = 6
S_LEN = 6
```

也就是每个决策时刻的 state 形状是：

```text
(6, 6)
```

6 行分别表示：

| 行号 | 含义 | 代码中的更新方式 |
|---|---|---|
| `state[0]` | 上一个 chunk 选的码率 | `VIDEO_BIT_RATE[bit_rate] / MAX_VIDEO_BIT_RATE` |
| `state[1]` | 当前 buffer 大小 | `buffer_size / BUFFER_NORM_FACTOR` |
| `state[2]` | 历史吞吐 | `video_chunk_size / delay / M_IN_K` |
| `state[3]` | 历史下载时间 | `delay / M_IN_K / BUFFER_NORM_FACTOR` |
| `state[4]` | 下一 chunk 在各码率下的大小 | `next_video_chunk_sizes / 1000 / 1000` |
| `state[5]` | 剩余 chunk 比例 | `video_chunk_remain / CHUNK_TIL_VIDEO_END_CAP` |

每一步都会先执行类似这样的操作：

```python
state = np.roll(state, -1, axis=1)
```

含义是：把旧历史往左挪一格，把最新观测写到最后一列。  
所以 `state[2]` 和 `state[3]` 是过去 6 步的历史，`state[0]`、`state[1]`、`state[5]` 主要只用最后一列，`state[4]` 存当前下一块在 6 个码率下的大小。

### 7.2 动作是码率档位

NetLLM 最终输出的是 6 个码率档位之一：

```text
0, 1, 2, 3, 4, 5
```

它们对应：

```text
0 -> 300 Kbps
1 -> 750 Kbps
2 -> 1200 Kbps
3 -> 1850 Kbps
4 -> 2850 Kbps
5 -> 4300 Kbps
```

注意：Genet/UDR 这类 Pensieve 模型内部有一个区别。它们的 actor 网络输出的是 3 个 jump-action：

```text
0 -> 降一档
1 -> 保持
2 -> 升一档
```

然后通过 `plm_special/utils/utils.py` 中的 `action2bitrate()` 转成最终码率档位。

### 7.3 奖励函数

奖励公式在 `run_baseline.py`、`generate_exp_pool.py`、`plm_special/evaluate.py`、`plm_special/test.py` 中保持一致：

```text
reward =
    VIDEO_BIT_RATE[bit_rate] / 1000
    - 4.3 * rebuf
    - abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / 1000
```

三个部分分别是：

| 项 | 含义 | 作用 |
|---|---|---|
| 码率收益 | 当前码率越高越好 | 鼓励高清 |
| 卡顿惩罚 | `4.3 * rebuf` | 强烈惩罚卡顿 |
| 平滑惩罚 | 当前码率和上一步码率差值 | 惩罚频繁大幅切换 |

这就是 QoE 的核心定义。论文中写 reward 或 QoE 时，必须和这个公式对齐。

## 8. baseline 主线

baseline 主线的入口是：

```text
run_baseline.py
```

它的作用是：在 ABR 环境中测试已有策略，输出每条 trace 的播放日志和平均 reward。

### 8.1 支持哪些 baseline

当前支持：

| 名称 | 类型 | 说明 |
|---|---|---|
| `bba` | 规则方法 | 只根据 buffer 大小选码率 |
| `mpc` | 规划方法 | 根据历史带宽估计未来带宽，搜索未来 5 步码率组合 |
| `genet` | 强化学习方法 | 使用 Genet/Pensieve actor checkpoint |
| `udr_1` | 强化学习方法 | UDR 训练出来的 Pensieve 变体 |
| `udr_2` | 强化学习方法 | UDR 训练出来的 Pensieve 变体 |
| `udr_3` | 强化学习方法 | UDR 训练出来的 Pensieve 变体 |
| `udr_real` | 强化学习方法 | 另一组 UDR checkpoint |

### 8.2 BBA 怎么理解

BBA 是 Buffer-Based Adaptation。它基本不预测带宽，只看 buffer：

```text
buffer 很低 -> 选低码率，保命
buffer 很高 -> 选高码率，提画质
中间区域 -> 按 buffer 线性插值
```

代码中：

```text
RESEVOIR = 5
CUSHION = 10
```

也就是 buffer 低于 5 秒时选最低码率，高于 15 秒时选最高码率，中间线性变化。

### 8.3 MPC 怎么理解

MPC 是 Model Predictive Control。它会：

1. 看最近 5 步吞吐。
2. 用调和平均估计未来带宽。
3. 根据过去预测误差做保守修正。
4. 枚举未来 5 个 chunk 的码率组合。
5. 选一个未来总 reward 最大的组合。
6. 只执行组合里的第一个动作。

这种方法不需要训练，但是依赖对未来带宽的估计。

### 8.4 Genet/UDR 怎么理解

Genet/UDR 属于 Pensieve 系列强化学习策略。代码中通过 TensorFlow 1.x 恢复 actor 网络：

```text
data/all_models/genet/nn_model_ep_9900.ckpt
data/all_models/udr_3/nn_model_ep_58000.ckpt
```

这些 checkpoint 是 Git 跟踪的小模型，不是 `data/ft_plms/` 里的 LLM 大 checkpoint。

### 8.5 baseline 运行命令

进入 ABR 目录：

```bash
cd /data3/wangxh/NetLLM-master/adaptive_bitrate_streaming
```

激活 TensorFlow 环境：

```bash
conda activate abr_tf
```

运行 Genet：

```bash
python run_baseline.py \
  --model genet \
  --test-trace fcc-test \
  --video video1 \
  --test-trace-num 100 \
  --seed 100003 \
  --fixed-order \
  --cuda-id 0
```

运行 MPC：

```bash
python run_baseline.py \
  --model mpc \
  --test-trace fcc-test \
  --video video1 \
  --test-trace-num 100 \
  --seed 100003 \
  --fixed-order
```

运行 BBA：

```bash
python run_baseline.py \
  --model bba \
  --test-trace fcc-test \
  --video video1 \
  --test-trace-num 100 \
  --seed 100003 \
  --fixed-order
```

结果会保存到类似：

```text
artifacts/results/fcc-test_video1/trace_num_100_fixed_True/genet/seed_100003/
```

每个 trace 会生成一个 `result_sim_abr_*` 文件，每行格式是：

```text
time_stamp    bit_rate    buffer_size    rebuf    chunk_size    download_time    smoothness    reward
```

最后通过 `calc_mean_reward(..., skip_first_reward=True)` 统计平均 reward。这里会跳过每条 trace 的第一条 reward，这是为了和 Pensieve/Genet 的统计口径保持一致。

## 9. 经验池主线

经验池是 NetLLM 训练数据的来源。入口是：

```text
generate_exp_pool.py
```

一句话理解：

> 先让 baseline 策略在环境里跑，把每一步的 state、action、reward、done 存下来，形成一个离线数据集，再让 NetLLM 模仿和学习这些决策轨迹。

### 9.1 经验池里存什么

`plm_special/data/exp_pool.py` 中的 `ExperiencePool` 很简单：

```python
self.states = []
self.actions = []
self.rewards = []
self.dones = []
```

每一步存：

| 字段 | 含义 |
|---|---|
| `state` | 当前 ABR 状态，形状 `(6, 6)` |
| `action` | 当前选择的码率档位，范围 `0..5` |
| `reward` | 当前动作得到的 QoE reward |
| `done` | 当前 episode 是否结束 |

当前默认经验池：

```text
artifacts/exp_pools/exp_pool.pkl
```

本地检查结果：

| 项 | 数值 |
|---|---:|
| 样本数 | `19928` |
| episode 数 | `424` |
| state 形状 | `(6, 6)` |
| action 范围 | `0..5` |
| reward 最小值 | `-85.0127` |
| reward 最大值 | `4.3` |

还有一个小型测试经验池：

```text
artifacts/exp_pools/exp_pool_smoke.pkl
```

它只有 `94` 条样本，适合快速检查流程是否能跑通。

### 9.2 为什么生成经验池时跳过第一个样本

`generate_exp_pool.py` 中每个 episode 结束时，会把：

```python
states[1:]
actions[1:]
rewards[1:]
dones[1:]
```

加入总经验池。

原因是第一步的 state 通常是全零初始化状态，还没有真实历史信息。跳过它可以减少无意义样本。后续测试统计中也常用 `skip_first_reward=True`，这和经验池处理口径一致。

### 9.3 生成经验池命令

激活 baseline 环境：

```bash
conda activate abr_tf
cd /data3/wangxh/NetLLM-master/adaptive_bitrate_streaming
```

用 Genet 生成经验池：

```bash
python generate_exp_pool.py \
  --models genet \
  --trace fcc-train \
  --video video1 \
  --trace-num -1 \
  --seed 100003 \
  --fixed-order \
  --cuda-id 0
```

也可以混合多个老师策略：

```bash
python generate_exp_pool.py \
  --models bba mpc genet \
  --trace fcc-train \
  --video video1 \
  --trace-num -1 \
  --seed 100003 \
  --fixed-order \
  --cuda-id 0
```

注意：建议总是显式写 `--models genet` 或 `--models bba mpc genet`，不要依赖 argparse 的默认值。

输出路径会类似：

```text
artifacts/exp_pools/fcc-train_video1/genet/seed_100003_trace_num_-1_fixed_True/exp_pool.pkl
```

训练时可以通过 `--exp-pool-path` 指定它。

## 10. `ExperienceDataset` 怎么把经验池变成训练样本

代码在：

```text
plm_special/data/dataset.py
```

经验池只是连续轨迹。模型训练不能直接把整个经验池一次性塞进去，而是要做三件事：

1. 归一化 reward。
2. 按 episode 计算 return。
3. 切成长度为 `w` 的固定窗口。

### 10.1 reward 归一化

代码中：

```python
rewards = (rewards - min_reward) / (max_reward - min_reward)
```

这样 reward 会映射到大致 `[0, 1]` 范围，便于模型学习。

### 10.2 return 计算

每个 episode 内，从后往前计算折扣回报：

```text
G_t = r_t + gamma * G_{t+1}
```

然后除以 `scale`。当前常用：

```text
gamma = 1.0
scale = 1000
```

所以这里的 return 是经过缩放后的累计归一化奖励。

### 10.3 固定窗口切片

`ExperienceDataset` 的核心参数是：

```text
max_length = w
```

当前常用：

```text
w = 20
```

它会构造：

```python
dataset_indices = range(0, exp_pool_size - max_length + 1, sample_step)
```

如果 `sample_step=None`，就默认 `sample_step = max_length`。也就是说默认每 20 步切一个窗口，不重叠。

默认 `exp_pool.pkl` 在 `w=20` 时，本地检查得到：

| 项 | 数值 |
|---|---:|
| 原始经验条数 | `19928` |
| 切窗后训练样本数 | `996` |
| `max_timestep` | `46` |
| `max_action` | `5` |
| `min_action` | `0` |

每个训练样本返回：

```text
states[start:end]
actions[start:end]
rewards[start:end]
returns[start:end]
timesteps[start:end]
```

这就是后面模型一次 forward 看到的数据。

## 11. NetLLM 模型主线

NetLLM 的入口是：

```text
run_plm.py
```

它做的事很多，但主流程可以拆成 6 步：

1. 设置随机种子。
2. 创建 ABR 测试或验证环境。
3. 读取经验池，构造 `ExperienceDataset`。
4. 加载基础大模型。
5. 构造状态编码器和 `OfflineRLPolicy`。
6. 根据参数执行训练 `--adapt` 或测试 `--test`。

### 11.1 基础大模型

基础模型通过 `plm_special/utils/plm_utils.py` 加载。支持的类型包括：

```text
gpt2
llama
t5-lm
opt
mistral
qwen
```

当前本地常见模型路径：

```text
../downloaded_plms/qwen/base
../downloaded_plms/llama3.2/base
../downloaded_plms/llama_tiny_smoke/base
```

如果默认路径找不到，可以传：

```bash
--plm-path ../downloaded_plms/llama3.2/base
```

### 11.2 LoRA 适配

LoRA 逻辑在：

```text
plm_special/models/low_rank.py
```

当 `--rank != -1` 时，代码会启用 LoRA：

```text
r = rank
lora_alpha = 32
lora_dropout = 0.05
bias = none
task_type = FEATURE_EXTRACTION
```

对 Llama、Qwen、Mistral、OPT 等模型，目标模块是：

```text
q_proj
v_proj
```

也就是说，不是全量微调大模型，而是：

1. 冻结基础大模型大部分参数。
2. 在注意力的 Q/V 投影上训练低秩 adapter。
3. 同时训练状态编码器、return/action embedding、LayerNorm、action head 等非 PLM 模块。

保存时也分两部分：

| 文件 | 内容 |
|---|---|
| `adapter_config.json`、`adapter_model.bin` | LoRA adapter |
| `modules_except_plm.bin` | 状态编码器、action head 等非 PLM 模块 |

所以如果你要拷贝一个训练好的模型，不能只拷贝 `adapter_model.bin`，还要带上 `modules_except_plm.bin`。

## 12. 状态编码器：从 ABR 数字状态到大模型 token

状态编码器在：

```text
plm_special/models/state_encoder.py
```

它的任务是把 `(6, 6)` 的 ABR 数值状态变成大模型能接收的 embedding token。

当前支持三种：

```text
legacy
patch_reprogram
semantic_reprogram
```

### 12.1 `legacy`

这是原始 NetLLM/Pensieve 风格的编码器，类名：

```text
EncoderNetwork
```

它把 6 类状态分别编码：

| 状态类型 | 编码方式 |
|---|---|
| 上一时刻码率 | Linear + LeakyReLU |
| buffer | Linear + LeakyReLU |
| 吞吐历史 | Conv1d + LeakyReLU |
| 下载时间历史 | Conv1d + LeakyReLU |
| 下一 chunk 大小 | Conv1d + LeakyReLU |
| 剩余 chunk | Linear + LeakyReLU |

最终输出 6 个状态特征，再映射到 PLM embedding 维度。

这是最基础、最接近原始代码的路径。

### 12.2 `patch_reprogram`

这是受 TIME-LLM patch reprogramming 启发的编码器，类名：

```text
PatchReprogrammingEncoder
```

它的思想是：

1. 把 ABR 数值序列切成小 patch。
2. 每个 patch 映射成一个数值 token。
3. 用一组从词表 embedding 中初始化的 text prototypes 做 cross-attention。
4. 把数值 patch 重编程到语言模型 embedding 空间。

这条线是“显式 patch 化”的重编程方法。

### 12.3 `semantic_reprogram`

这是你当前论文主线更关注的编码器，类名：

```text
SemanticReprogrammingEncoder
```

它的核心思想是：

> 不把 ABR 数字状态硬塞给大模型，而是先构造和网络语义相关的锚点，再把数字特征对齐到这些语义空间里。

它内部有两类文本：

1. `FEATURE_DESCRIPTIONS`
2. `ANCHOR_TEXTS`

例如：

```text
selected bitrate
buffer size
throughput history
download time history
next chunk sizes
remaining video chunks
```

以及：

```text
low bitrate
high bitrate
buffer low
buffer safe
bandwidth drop
bandwidth stable
download time long
video ending
```

代码会用 tokenizer 把这些短语转成 token id，再取词向量平均，得到语义锚点。然后用 attention 把数值 token 对齐到这些锚点。

这就是“语义重编程”的核心。

## 13. 当前项目的增强模块

`run_plm.py` 和 `state_encoder.py` 里有一些你后续加的增强模块。新手一定要分清楚：它们不是默认全部开启，而是通过参数控制。

### 13.1 pre-align intra-step attention

参数：

```bash
--use-pre-align-intra-step-attn
```

作用位置：

```text
SemanticReprogrammingEncoder
```

含义：

在数值状态对齐到语言语义空间之前，先让同一步内的状态 token 和上下文 token 做一次轻量 attention。

这里的上下文 token 包括：

```text
prev_reward
prev_action
```

也就是上一时刻奖励和上一时刻动作。

为什么有用：

ABR 决策强依赖“刚才选了什么”和“刚才效果怎么样”。例如：

1. 上一步选高码率后 reward 很差，可能说明刚才造成了卡顿。
2. 上一步选低码率但 reward 还不错，可能说明网络条件偏差但策略稳定。

这些上下文信息能帮助模型判断下一步应该激进还是保守。

### 13.2 pre-align mask

参数：

```bash
--use-pre-align-intra-step-mask
--pre-align-intra-step-mask-mode state_to_prev_reward
```

当前代码支持的 mask 模式：

```text
context_readonly
state_to_prev_action
state_to_prev_reward
state_only
```

它控制状态 token 和上下文 token 怎么互相看。比如 `state_to_prev_reward` 可以理解为：让状态 token 读到上一时刻 reward 相关信息，但限制某些不必要的混合。

论文中常说的 `maskprevreward`，对应的就是这类上下文预对齐增强方法。

### 13.3 conditional attention

参数：

```bash
--use-pre-align-conditional-attn
```

它和 `--use-pre-align-intra-step-attn` 不能同时开。

含义：

状态 token 不彼此混合，而是把 `prev_reward` 和 `prev_action` 当成条件信息，由状态 token 去读这些上下文。

这是一种更受控的上下文注入方式。

### 13.4 history multiscale mixer

参数：

```bash
--use-history-multiscale-mixer
```

它要求：

```bash
--state-encoder-type semantic_reprogram
```

当前实现只对 `throughput history` 做多尺度混合，没有同时混合下载时间历史。代码里：

```text
throughput_history_mixer = MultiScaleHistoryMixer(...)
download_history_mixer = None
```

它的思想是：

1. 把最近 6 步吞吐历史看成短时间序列。
2. 做多尺度下采样，比如 `6 -> 3`。
3. 分解 seasonal 和 trend。
4. bottom-up 混合 seasonal 信息。
5. top-down 混合 trend 信息。
6. 融合后通过 residual gate 注入原吞吐特征。

你论文里的 `hmix_v2lite` 可以理解为这条较轻量、较保守的多尺度历史增强路径。

### 13.5 intra-state attention

参数：

```bash
--use-intra-state-attn
```

作用位置：

```text
OfflineRLPolicy
```

含义：

状态编码后，每个时间步里有多个状态 token，例如码率、buffer、吞吐、下载时间、chunk 大小、剩余长度。intra-state attention 让同一个时间步内的这些 token 先互相交流。

### 13.6 gated intra-state attention

参数：

```bash
--use-gated-intra-state-attn
```

它和 `--use-intra-state-attn` 不能同时开。

区别是：

它用一个初始为 0 的 gate 控制 attention residual 注入强度。这样一开始模型更接近原始路径，训练中再逐渐学会是否使用新特征。

### 13.7 temporal state attention

参数：

```bash
--use-temporal-state-attn
--use-temporal-causal-mask
```

作用：

让同一类状态 token 跨时间步交流。例如吞吐 token 看过去多个时间步的吞吐 token。

如果加 `--use-temporal-causal-mask`，则当前时间步只能看自己和过去，不能看未来。

## 14. `OfflineRLPolicy` 如何把所有东西拼起来

核心策略模型在：

```text
plm_special/models/rl_policy.py
```

它包含：

| 组件 | 作用 |
|---|---|
| `state_encoder` | 把 `(6,6)` 状态转成状态 token |
| `embed_timestep` | 时间步 embedding |
| `embed_return` | 目标 return embedding |
| `embed_action` | 动作 embedding |
| `embed_ln` | 输入大模型前的 LayerNorm |
| `plm` | 基础大语言模型 |
| `action_head` | 把 PLM hidden state 映射成 6 个码率 logits |

### 14.1 训练时的输入序列

训练 forward 中，序列会被组织成：

```text
R_1, s_1_token_1, ..., s_1_token_n, a_1,
R_2, s_2_token_1, ..., s_2_token_n, a_2,
...
R_t, s_t_token_1, ..., s_t_token_n, a_t
```

其中：

| 记号 | 含义 |
|---|---|
| `R_t` | return embedding |
| `s_t_token_i` | 状态编码器输出的第 i 个状态 token |
| `a_t` | 动作 embedding |

模型使用 PLM 处理这些 embedding，然后在状态 token 的位置取 hidden state，通过 `action_head` 预测动作。

### 14.2 训练标签

标签来自经验池中的真实动作：

```text
labels = actions.long()
```

损失函数是：

```text
CrossEntropyLoss
```

也就是说，训练目标是让模型根据历史 return、state、action 结构预测经验池中的码率动作。

### 14.3 测试时的采样

测试时调用：

```text
model.sample(...)
```

这里不要误以为一定是 `argmax`。当前实现是：

```python
pi = softmax(logits)
idx = random.choices(np.arange(pi.size), pi)[0]
```

也就是根据 softmax 概率采样动作。

这会让随机种子影响测试结果，因为相同 logits 下，不同 seed 可能采到不同动作序列。

## 15. 训练流程

训练入口：

```bash
python run_plm.py --adapt ...
```

或者使用封装脚本：

```bash
./run_qwen.sh --mode adapt ...
```

### 15.1 训练时发生什么

`run_plm.py` 中的 `adapt()` 会：

1. 创建 `AdamW` 优化器。
2. 创建 warmup 学习率调度器。
3. 使用 `CrossEntropyLoss`。
4. 创建 `Trainer`。
5. 每个 epoch 遍历 `ExperienceDataset`。
6. 每隔 `eval_per_epoch` 在环境里验证。
7. 如果验证 return 更好，就保存 best model。
8. 定期保存 checkpoint。
9. 把训练 loss 写到 `train_losses.txt`。
10. 把控制台输出写到 `early_stop_-1_console.log`。

### 15.2 梯度累积

`Trainer` 中 batch size 默认是 1，这是为了避免显存爆炸。实际有效 batch 通过梯度累积实现：

```text
--grad-accum-steps 32
```

含义是：连续算 32 个小 batch 的梯度后再 optimizer step。

### 15.3 输出目录怎么命名

模型保存到：

```text
data/ft_plms/{plm_type}_{plm_size}/{exp_pool_name}_ss{sample_step}/{run_tag}/
```

例如：

```text
data/ft_plms/llama_small/exp_pool_ssna/
  sr_sfd256_h4_preisa_prev_ar_h8_hd1024_d0p1_maskprevreward_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s100003/
```

长目录名看起来吓人，但它其实是实验配置的编码：

| 片段 | 含义 |
|---|---|
| `sr` | semantic reprogram |
| `sfd256` | state feature dim = 256 |
| `h4` | reprogram heads = 4 |
| `preisa` | pre-align intra-step attention |
| `prev_ar` | 使用 previous action/reward 相关上下文 |
| `maskprevreward` | mask 模式偏向 prev_reward |
| `hmix` | 开启 history multiscale mixer |
| `isa_off` | intra-state attention 关闭 |
| `tsa_off` | temporal-state attention 关闭 |
| `r128` | LoRA rank = 128 |
| `w20` | 窗口长度 = 20 |
| `g1` | gamma = 1 |
| `lr0p0001` | 学习率 = 0.0001 |
| `wd0p0001` | weight decay = 0.0001 |
| `wu2000` | warmup steps = 2000 |
| `e60` | 训练 epoch = 60 |
| `s100003` | seed = 100003 |

这个命名方式有点长，但好处是不用打开日志也能知道实验配置。

## 16. 测试流程

测试入口：

```bash
python run_plm.py --test ...
```

或者：

```bash
./run_qwen.sh --mode test ...
```

### 16.1 测试时发生什么

`plm_special/test.py` 中的 `test_on_env()` 会：

1. 加载 ABR 环境。
2. 从默认码率开始播放。
3. 每一步把当前状态传给模型。
4. 模型采样出下一个码率。
5. 环境根据码率返回下载结果。
6. 写入每条 trace 的播放日志。
7. 最后用 `calc_mean_reward(..., skip_first_reward=True)` 计算平均 reward。

### 16.2 测试结果目录

测试结果保存到：

```text
artifacts/results/{trace}_{video}/trace_num_{N}_fixed_{bool}/{plm_type}_{plm_size}/{run_tag}_stop{which_layer}_tgt{target_return_scale}/
```

例如：

```text
artifacts/results/fcc-test_video1/trace_num_100_fixed_True/llama_small/...
```

每个 trace 一个结果文件。每行 8 列：

```text
time_stamp
bit_rate
buffer_size
rebuf
video_chunk_size
download_time
smoothness
reward
```

### 16.3 为什么第一条 reward 不算

测试统计中默认：

```python
skip_first_reward=True
```

因为第一步是从全零状态和默认码率开始，和后续有完整历史的状态不完全一致。为了和 Pensieve/Genet 口径一致，统计时跳过每条 trace 第一条 reward。

## 17. 常用命令

下面给一组新手最容易用上的命令。所有命令默认从：

```bash
cd /data3/wangxh/NetLLM-master/adaptive_bitrate_streaming
```

开始。

### 17.1 快速看 baseline

```bash
conda activate abr_tf

python run_baseline.py \
  --model genet \
  --test-trace fcc-test \
  --video video1 \
  --test-trace-num 100 \
  --seed 100003 \
  --fixed-order \
  --cuda-id 0
```

### 17.2 生成经验池

```bash
conda activate abr_tf

python generate_exp_pool.py \
  --models genet \
  --trace fcc-train \
  --video video1 \
  --trace-num -1 \
  --seed 100003 \
  --fixed-order \
  --cuda-id 0
```

### 17.3 使用 Qwen 跑 legacy encoder

```bash
./run_qwen.sh \
  --mode both \
  --plm-type qwen \
  --plm-size base \
  --plm-path ../downloaded_plms/qwen/base \
  --state-encoder-type legacy \
  --device cuda:0 \
  --rank 128 \
  --num-epochs 60 \
  --eval-per-epoch 2 \
  --seed 100003 \
  --fixed-order
```

### 17.4 使用 Llama3.2 small 跑 semantic_only

```bash
./run_llama32.sh \
  --mode both \
  --state-encoder-type semantic_reprogram \
  --device cuda:0 \
  --rank 128 \
  --num-epochs 60 \
  --eval-per-epoch 2 \
  --seed 100003 \
  --fixed-order \
  --exp-tag semantic_only
```

### 17.5 跑 maskprevreward

```bash
./run_llama32.sh \
  --mode both \
  --state-encoder-type semantic_reprogram \
  --use-pre-align-intra-step-attn \
  --use-pre-align-intra-step-mask \
  --pre-align-intra-step-mask-mode state_to_prev_reward \
  --device cuda:0 \
  --rank 128 \
  --num-epochs 60 \
  --eval-per-epoch 2 \
  --seed 100003 \
  --fixed-order \
  --exp-tag maskprevreward
```

### 17.6 跑 hmix_v2lite

```bash
./run_llama32.sh \
  --mode both \
  --state-encoder-type semantic_reprogram \
  --use-pre-align-intra-step-attn \
  --use-pre-align-intra-step-mask \
  --pre-align-intra-step-mask-mode state_to_prev_reward \
  --use-history-multiscale-mixer \
  --device cuda:0 \
  --rank 128 \
  --num-epochs 60 \
  --eval-per-epoch 2 \
  --seed 100003 \
  --fixed-order \
  --exp-tag hmix_v2lite
```

### 17.7 只测试已有 best model

如果你知道模型目录：

```bash
./run_llama32.sh \
  --mode test \
  --state-encoder-type semantic_reprogram \
  --use-pre-align-intra-step-attn \
  --use-pre-align-intra-step-mask \
  --pre-align-intra-step-mask-mode state_to_prev_reward \
  --use-history-multiscale-mixer \
  --model-dir data/ft_plms/llama_small/.../early_stop_-1_best_model \
  --device cuda:0 \
  --rank 128 \
  --seed 100003 \
  --fixed-order
```

如果不传 `--model-dir`，`run_qwen.sh` 会尝试在对应 `data/ft_plms/{plm_type}_{plm_size}` 下找最新的 `early_stop_*_best_model`。

### 17.8 dry run 检查命令

脚本支持：

```bash
./run_llama32.sh --mode adapt --dry-run ...
```

它会打印实际调用的 `python run_plm.py ...` 命令，但不真正运行。新手改参数时很有用。

## 18. 结果怎么看

### 18.1 baseline 结果

baseline 结果目录类似：

```text
artifacts/results/fcc-test_video1/trace_num_100_fixed_True/genet/seed_100003/
```

每个 `result_sim_abr_*` 对应一条 trace。你可以打开其中一个看每个 chunk 的播放过程。

### 18.2 NetLLM 结果

NetLLM 结果目录类似：

```text
artifacts/results/fcc-test_video1/trace_num_100_fixed_True/llama_small/{run_tag}_stop-1_tgt1/
```

看结果时最常用的指标是：

| 指标 | 含义 |
|---|---|
| `mean_reward` | 平均 QoE reward，越高越好 |
| `bitrate` | 平均码率，反映画质 |
| `rebuf` | 平均卡顿时间，越低越好 |
| `smooth` | 平滑惩罚，越低越好 |

不要只看 bitrate。ABR 不是单纯追求高清，卡顿惩罚很重，低 rebuf 往往比盲目高 bitrate 更重要。

### 18.3 已有中文实验记录

项目里已有几份实验文档，适合在理解主流程后阅读：

| 文件 | 作用 |
|---|---|
| `PROJECT_FLOW_ANALYSIS_ZH.md` | 已有的项目流程分析，偏代码流程 |
| `THESIS_EXPERIMENT_GUIDE_ZH.md` | 论文实验整理和补实验建议 |
| `PREISA_EXPERIMENT_COMPARISON.md` | semantic reprogram、maskprevreward、hmix、hmix_v2lite 实验记录 |
| `NEXT_EXPERIMENTS_PLAYBOOK_ZH.md` | 后续实验计划 |
| `UNDERGRAD_THESIS_PLAN_ZH.md` | 本科论文层面的写作和实验规划 |

本文是新手总入口。读完本文后，再读这些实验文档会更容易。

## 19. 从源码角度逐文件学习

这一节按“老师带读代码”的方式讲每个关键文件应该抓什么。

### 19.1 `baseline_special/utils/constants.py`

先看常量：

```text
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
S_INFO = 6
S_LEN = 6
TOTAL_VIDEO_CHUNK = 48
VIDEO_CHUNK_LEN = 4000 ms
```

这就是 ABR 问题的基本设定。论文里的码率、状态维度、奖励公式都要和这里对齐。

### 19.2 `baseline_special/env.py`

重点看：

```python
get_video_chunk(self, quality)
```

你需要理解：

1. 输入 `quality` 是码率档位。
2. 环境读取当前 chunk 大小。
3. 沿着 trace 模拟下载。
4. 计算 delay、rebuf、buffer。
5. 返回下一 chunk 的各码率大小和剩余 chunk 数。

这是所有方法共享的“世界”。

### 19.3 `run_baseline.py`

重点看：

1. 参数解析。
2. BBA、MPC、Pensieve 三种策略。
3. 状态更新。
4. reward 计算。
5. 结果日志写入。
6. `calc_mean_reward(..., skip_first_reward=True)`。

读这个文件时，不要被 MPC 的枚举细节卡住。先抓主循环：

```text
get_video_chunk -> 计算 reward -> 更新 state -> 选择下一 bit_rate -> 写日志
```

### 19.4 `generate_exp_pool.py`

它和 `run_baseline.py` 很像，但不是为了写测试结果，而是为了收集训练数据。

重点看：

```python
collect_experience(...)
```

以及：

```python
exp_pool.add(state, action, reward, done)
```

这说明经验池不是从数据集文件凭空来的，而是 baseline 策略和 ABR 环境真实交互产生的。

### 19.5 `plm_special/data/dataset.py`

重点看：

1. `_normalize_rewards()`
2. `_compute_returns()`
3. `__getitem__()`
4. `dataset_indices`

这能回答一个关键问题：

> 训练样本到底是什么？

答案是：固定长度窗口中的 state/action/reward/return/timestep 序列。

### 19.6 `plm_special/models/state_encoder.py`

这里是方法创新最集中的地方之一。

先读：

```text
EncoderNetwork
```

理解传统状态编码。

再读：

```text
SemanticReprogrammingEncoder
```

理解语义重编程、anchor text、pre-align、history mixer。

最后读：

```text
PatchReprogrammingEncoder
```

它是另一种 patch 化重编程思路。

### 19.7 `plm_special/models/rl_policy.py`

这是最核心的模型文件。

重点看：

1. `__init__()` 中有哪些模块。
2. `_encode_states()` 如何调用状态编码器。
3. `_pack_sequence_inputs()` 如何组织 `return + state + action`。
4. `forward()` 如何训练。
5. `sample()` 如何测试。
6. `_sample()` 为什么是 softmax sampling。

读懂这个文件后，你就能解释“LLM 到底在 ABR 里做了什么”。

### 19.8 `plm_special/trainer.py`

重点看：

1. DataLoader。
2. `train_step()`。
3. loss 的计算。
4. 梯度累积。
5. 梯度裁剪。

这个文件告诉你模型怎么学习经验池中的动作。

### 19.9 `plm_special/evaluate.py` 和 `plm_special/test.py`

两者很像：

| 文件 | 用途 |
|---|---|
| `evaluate.py` | 训练过程中定期验证，用于选 best model |
| `test.py` | 最终测试，写结果文件，统计 mean_reward |

重点看：

1. target return 如何递减。
2. prev_reward 如何传给模型。
3. state 如何更新。
4. episode 结束如何重置。

### 19.10 `run_plm.py`

最后读它。

你要抓住：

1. 参数区有哪些开关。
2. `run(args)` 的 6 个步骤。
3. 如何加载经验池。
4. 如何加载 PLM。
5. 如何创建 state encoder。
6. 如何创建 `OfflineRLPolicy`。
7. 如何生成保存目录。
8. `--adapt` 和 `--test` 分别走哪里。

`run_plm.py` 是总调度，不是某一个算法细节的全部来源。

## 20. 新手学习路线

如果一个完全新的人要接手，建议按 7 天节奏学习。

### 第 1 天：只理解 ABR 问题

目标：

1. 知道什么是 chunk。
2. 知道什么是 bitrate。
3. 知道 buffer 和 rebuf。
4. 能写出 reward 公式。

应该读：

```text
README.md
baseline_special/utils/constants.py
baseline_special/env.py
```

练习：

1. 找出码率列表。
2. 找出一个 episode 有多少个 chunk。
3. 解释为什么卡顿惩罚比平滑惩罚更重要。

### 第 2 天：跑通 baseline

目标：

1. 跑 `bba`。
2. 跑 `mpc`。
3. 跑 `genet`。
4. 打开一个结果文件看每列含义。

应该读：

```text
run_baseline.py
plm_special/utils/utils.py
```

练习：

1. 对比 BBA 和 MPC 的平均 reward。
2. 找一条 trace，看码率是否频繁波动。
3. 看 `skip_first_reward=True` 对统计有何影响。

### 第 3 天：理解经验池

目标：

1. 知道经验池从哪里来。
2. 知道经验池存什么。
3. 知道为什么跳过 episode 第一条样本。

应该读：

```text
generate_exp_pool.py
plm_special/data/exp_pool.py
plm_special/data/dataset.py
```

练习：

1. 用 `exp_pool_smoke.pkl` 打印样本数。
2. 看一个 state 的形状。
3. 理解 `w=20` 时一个训练样本包含什么。

### 第 4 天：理解基础 NetLLM

目标：

1. 理解 PLM 不直接接收 token id，而是接收 `inputs_embeds`。
2. 理解状态编码器为什么必要。
3. 理解 action head 输出 6 个 logits。

应该读：

```text
plm_special/models/state_encoder.py
plm_special/models/rl_policy.py
plm_special/utils/plm_utils.py
```

练习：

1. 画出 `return + state + action` 输入序列。
2. 找到 `action_head`。
3. 找到 `_sample()`，说明测试为什么有随机性。

### 第 5 天：理解 LoRA 和训练

目标：

1. 知道 `rank=128` 是 LoRA 秩。
2. 知道保存模型时有 adapter 和 `modules_except_plm.bin` 两部分。
3. 知道训练 loss 是 CrossEntropyLoss。

应该读：

```text
plm_special/models/low_rank.py
plm_special/trainer.py
run_plm.py
```

练习：

1. 找到 LoRA 的 target modules。
2. 找到 `save_model()`。
3. 找到 best model 保存逻辑。

### 第 6 天：理解方法增强

目标：

1. 区分 `semantic_only`、`maskprevreward`、`hmix_v2lite`。
2. 知道 pre-align 和 history mixer 分别在哪里实现。
3. 知道哪些功能是可选开关。

应该读：

```text
plm_special/models/state_encoder.py
run_plm.py
PREISA_EXPERIMENT_COMPARISON.md
```

练习：

1. 写出 `semantic_only` 的命令。
2. 写出 `maskprevreward` 的命令。
3. 写出 `hmix_v2lite` 的命令。

### 第 7 天：理解实验结果和论文口径

目标：

1. 知道 mean_reward、bitrate、rebuf、smooth 的含义。
2. 知道结果目录怎么找。
3. 知道论文主线方法怎么命名。

应该读：

```text
THESIS_EXPERIMENT_GUIDE_ZH.md
PREISA_EXPERIMENT_COMPARISON.md
PROJECT_FLOW_ANALYSIS_ZH.md
```

练习：

1. 找到 `hmix_v2lite` 的结果目录。
2. 解释为什么不能只看 bitrate。
3. 用一句话解释 `maskprevreward` 相比 `semantic_only` 加了什么。

## 21. 常见坑

### 21.1 TensorFlow 和 PyTorch 环境混用

现象：

```text
No module named tensorflow
No module named torch
```

解决：

| 命令 | 应用环境 |
|---|---|
| `run_baseline.py` | `abr_tf` |
| `generate_exp_pool.py` | `abr_tf` |
| `run_plm.py` | `abr_netllm` |

### 21.2 基础大模型路径不存在

现象：

```text
Cannot find foundation model path
```

解决：

1. 确认 `../downloaded_plms/qwen/base` 或 `../downloaded_plms/llama3.2/base` 存在。
2. 如果模型放在别处，传 `--plm-path`。

### 21.3 只从 GitHub 拉代码后不能直接测试 NetLLM

原因：

GitHub 通常没有：

```text
../downloaded_plms/
data/ft_plms/
```

解决：

1. 下载基础大模型。
2. 重新训练，或手动拷贝已有 best model。

### 21.4 只拷贝 LoRA adapter 不够

一个完整的训练结果至少需要：

```text
adapter_config.json
adapter_model.bin
modules_except_plm.bin
```

如果缺 `modules_except_plm.bin`，状态编码器和 action head 的参数就没了，测试不能正确恢复。

### 21.5 显存不够

常见处理：

1. 换更小模型，比如 `llama_small` 或 smoke 模型。
2. 调小 `--grad-accum-steps` 或训练规模。
3. 使用 `--device-out` 和 `--device-mid` 做模型切分。
4. 减少 `--trace-num` 做快速检查。
5. 先用 `exp_pool_smoke.pkl` 验证流程。

### 21.6 结果不完全一致

可能原因：

1. 测试采样不是 argmax，而是 softmax sampling。
2. `fixed-order` 没开时环境有随机性。
3. seed 不同。
4. 基础模型或 checkpoint 不一致。
5. `skip_first_reward` 统计口径不同。

### 21.7 `--models` 参数写错

`generate_exp_pool.py` 中 `--models` 是 `nargs='*'`。建议显式写：

```bash
--models genet
```

或者：

```bash
--models bba mpc genet
```

不要写成 `--model genet`，那是 `run_baseline.py` 的参数。

## 22. 给后来者的最短复现路线

如果新手只想最快验证项目能跑，建议按下面顺序。

### 22.1 检查代码和小数据

```bash
cd /data3/wangxh/NetLLM-master/adaptive_bitrate_streaming
ls artifacts/exp_pools/exp_pool.pkl
ls data/traces/test/fcc-test | head
ls data/videos/video1_sizes
ls data/all_models/genet
```

这些都在，说明基础小数据齐了。

### 22.2 跑一个 baseline

```bash
conda activate abr_tf
python run_baseline.py --model bba --test-trace fcc-test --video video1 --test-trace-num 5 --fixed-order
```

先用 5 条 trace，快。

### 22.3 检查经验池

```bash
/data3/wangxh/conda-envs/abr_netllm/bin/python - <<'PY'
import pickle
p = 'artifacts/exp_pools/exp_pool.pkl'
exp = pickle.load(open(p, 'rb'))
print(len(exp), exp.states[0].shape, min(exp.actions), max(exp.actions))
PY
```

如果能打印出 `19928 (6, 6) 0 5` 附近的信息，说明经验池能读。

### 22.4 检查基础大模型

```bash
ls ../downloaded_plms/qwen/base
ls ../downloaded_plms/llama3.2/base
```

如果没有，先准备模型。没有基础模型就不能跑 NetLLM。

### 22.5 用脚本 dry run

```bash
./run_llama32.sh \
  --mode adapt \
  --state-encoder-type semantic_reprogram \
  --dry-run
```

能打印 `python run_plm.py ...` 就说明脚本解析没问题。

## 23. 论文方法名和代码参数的对应关系

论文或答辩中常用的方法名，和代码参数可以这样对应：

| 论文/汇报名 | 代码开关 |
|---|---|
| `semantic_only`，语义重编程基线模型 | `--state-encoder-type semantic_reprogram`，不加 pre-align，不加 hmix |
| `maskprevreward`，上下文预对齐增强模型 | semantic reprogram + `--use-pre-align-intra-step-attn` + `--use-pre-align-intra-step-mask` + `--pre-align-intra-step-mask-mode state_to_prev_reward` |
| `hmix_v2lite`，轻量化多尺度历史增强模型 | `maskprevreward` 基础上再加 `--use-history-multiscale-mixer` |
| `legacy` 原始 NetLLM 编码器 | `--state-encoder-type legacy` |
| `patch_reprogram` patch 重编程 | `--state-encoder-type patch_reprogram` |

不要把所有增强都说成默认模型。当前代码里这些是不同开关控制的实验分支。

## 24. 术语表

| 术语 | 解释 |
|---|---|
| ABR | Adaptive Bitrate Streaming，自适应码率视频流 |
| chunk | 视频分块，当前每块 4 秒 |
| trace | 网络带宽随时间变化的轨迹 |
| buffer | 播放器缓存 |
| rebuf | 卡顿时间，下载不及播放导致 |
| QoE | 用户体验质量，当前由码率、卡顿、平滑性组成 |
| bitrate | 视频码率，越高通常越清晰 |
| smoothness | 码率切换平滑性 |
| episode | 一条 trace 上播放完整视频的过程 |
| state | 模型看到的状态，当前是 `(6,6)` |
| action | 选择的码率档位，NetLLM 中是 `0..5` |
| reward | 当前动作的 QoE 得分 |
| return | 从当前时刻往后的累计 reward |
| experience pool | baseline 和环境交互得到的离线轨迹集合 |
| LoRA | 低秩参数高效微调方法 |
| PLM | Pre-trained Language Model，预训练语言模型 |
| state encoder | 把 ABR 数值状态映射成模型 embedding 的模块 |
| networking head | 将大模型 hidden state 映射成网络任务输出的头，这里是 action head |
| semantic reprogramming | 把数值状态通过语义锚点对齐到语言模型空间 |
| pre-align | 在语义对齐前先注入上下文或做轻量 attention |
| hmix | history multiscale mixer，多尺度历史混合模块 |
| best model | 验证 return 最好时保存的模型 |
| checkpoint | 定期保存的训练中间状态 |

## 25. 最后给新手的一句话路线

如果你完全不知道从哪里开始，就按这个顺序：

```text
先理解 ABR 的 state/action/reward
再跑 baseline
再看经验池怎么生成
再看 ExperienceDataset 怎么切窗
再看 state_encoder 怎么把数字状态变成 token
再看 OfflineRLPolicy 怎么把 return/state/action 塞给大模型
最后看 run_plm.py 怎么把训练、验证、测试串起来
```

这个项目的难点不在某一行代码，而在于它把三套东西接在了一起：

1. ABR 网络仿真环境。
2. 离线强化学习式的经验池训练。
3. 大语言模型的参数高效适配。

只要把这三层分开，再看它们如何连接，整个项目就会清楚很多。
