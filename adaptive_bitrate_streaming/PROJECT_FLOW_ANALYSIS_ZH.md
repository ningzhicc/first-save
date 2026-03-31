# Adaptive Bitrate Streaming 项目完整流程分析

本文基于当前 `adaptive_bitrate_streaming` 目录里的实际代码整理，不是只按论文概念复述。重点会放在：

- 项目到底分成哪几条执行链路
- 从开始到结束每一步做了什么
- 为什么要这样做
- 关键实现细节、隐藏假设和容易踩坑的地方

## 1. 这个项目到底在做什么

这个子项目要解决的是 ABR（Adaptive Bitrate Streaming，自适应码率视频传输）问题。核心任务是：

- 客户端每下载一个视频 chunk，都要决定“下一个 chunk 用哪个码率”
- 网络带宽会变化，所以码率不能一直拉满
- 决策目标不是只追求高清，而是同时平衡：
  - 画质高
  - 卡顿少
  - 码率切换不要太剧烈

这套代码实际上有两条主线：

- `baseline` 主线：直接运行传统 ABR 方法做测试，主要是 `genet / udr_* / mpc / bba`
- `plm` 主线：先用 baseline 和环境交互生成经验池，再把经验池变成离线训练数据，用大模型做 NetLLM 适配，然后在同一个 ABR 环境里测试大模型

一句话概括整个项目：

> baseline 负责“产生日志和老师数据”，NetLLM 负责“把老师数据学成一个可直接做 ABR 决策的大模型策略”。

## 2. 统一问题抽象：状态、动作、奖励

不管是 baseline 还是 NetLLM，底层共享的是同一个 ABR 环境抽象。

### 2.1 一个 episode 是什么

- 一个 episode = 在一条带宽轨迹上把整段视频从头播到尾
- 当前实现里总 chunk 数是 `48`，每个 chunk 长度是 `4s`
- 所以一个 episode 本质上就是“48 次连续的码率决策”

这些常量在 `baseline_special/utils/constants.py` 里定义：

- `TOTAL_VIDEO_CHUNK = 48.0`
- `VIDEO_CHUNK_LEN = 4000.0`
- `VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]`

### 2.2 状态是什么

状态是一个 `6 x 6` 的张量，语义来自 Pensieve/Genet，代码见：

- `run_baseline.py`
- `generate_exp_pool.py`
- `plm_special/evaluate.py`
- `plm_special/test.py`

6 个通道分别是：

- `state[0]`：上一时刻选的码率
- `state[1]`：当前 buffer 大小
- `state[2]`：过去若干步的吞吐历史
- `state[3]`：过去若干步的下载时延历史
- `state[4]`：下一 chunk 在 6 个码率下的文件大小
- `state[5]`：距离视频结束还剩多少 chunk

为什么要这么设计：

- ABR 本来就是时序决策问题，只看当前瞬时带宽不够
- 过去吞吐和下载时间能帮助估计短期网络趋势
- 下一 chunk 的不同码率大小能帮助判断“选高码率会不会把 buffer 打空”
- 剩余 chunk 数能影响策略是否“保守收尾”还是“积极冲高画质”

### 2.3 动作是什么

这里有一个非常重要的细节：不同方法的动作语义并不完全一样。

- 对 `mpc` 和 `bba`：动作就是直接选 6 档码率之一，范围 `0~5`
- 对 Pensieve/Genet 类模型：神经网络本身输出的是 `3` 个跳变动作，含义是
  - `0`：降一档
  - `1`：保持
  - `2`：升一档
- 然后再通过 `plm_special/utils/utils.py` 里的 `action2bitrate()` 映射成最终码率 `0~5`

为什么 Pensieve 不直接输出 6 个码率：

- 这种“升/降/保持”的 jump-action 更平滑
- 可以天然限制剧烈跳变
- 当前仓库里的 Genet/UDR checkpoint 也是按这套动作空间训练出来的，所以测试时必须保持兼容

### 2.4 奖励函数是什么

奖励在多个文件里都一致，公式是：

`reward = 当前码率收益 - 重缓冲惩罚 - 平滑度惩罚`

具体形式：

`VIDEO_BIT_RATE[bit_rate] / 1000 - 4.3 * rebuf - |curr - last| / 1000`

为什么这么定义：

- 第一项鼓励更高画质
- 第二项强烈惩罚卡顿
- 第三项惩罚频繁切换码率，避免用户观感抖动

这也是整个项目的统一优化目标，所以 baseline 和 NetLLM 的结果才能直接公平对比。

## 3. 目录结构里每个关键文件在流程中的作用

### 3.1 入口层

- `config.py`
  - 定义 trace、video、baseline checkpoint、PLM 路径、输出目录等
- `run_baseline.py`
  - baseline 测试主入口
- `generate_exp_pool.py`
  - 用 baseline 跑环境，生成离线经验池
- `run_plm.py`
  - NetLLM 训练和测试主入口
- `test.py`
  - 只是一个本地 Qwen 加载测试脚本，不是正式实验入口

### 3.2 baseline 层

- `baseline_special/env.py`
  - ABR 仿真环境，负责“给定码率 -> 模拟下载 -> 返回 buffer/rebuf/chunk 信息”
- `baseline_special/a3c.py`
  - Pensieve/Genet 使用的 actor/critic 网络定义
- `baseline_special/utils/constants.py`
  - 状态维度、奖励参数、码率档位等常量
- `baseline_special/utils/utils.py`
  - 加载 trace、统计 CDF 等工具

### 3.3 PLM 层

- `plm_special/data/exp_pool.py`
  - 最原始的经验池容器
- `plm_special/data/dataset.py`
  - 把经验池加工成离线训练样本
- `plm_special/models/state_encoder.py`
  - 把 ABR 状态编码成特征
- `plm_special/models/rl_policy.py`
  - 真正把“return + state + action”组织成 Transformer 输入序列的核心策略模型
- `plm_special/models/low_rank.py`
  - LoRA 适配逻辑
- `plm_special/utils/plm_utils.py`
  - 加载不同种类的大模型
- `plm_special/trainer.py`
  - 训练循环
- `plm_special/evaluate.py`
  - 每轮训练后在线验证
- `plm_special/test.py`
  - 最终测试并写结果文件

## 4. 从开始到结束的完整流程

下面按“真正跑项目”的顺序讲完整主线。

### 4.1 第一步：准备环境和路径

README 里明确要求分两个环境：

- baseline 用 `TensorFlow 1.x`
- NetLLM 用 `PyTorch + transformers + peft`

为什么必须分开：

- baseline 代码沿用 Genet/Pensieve 的老实现，依赖 `tensorflow-gpu==1.15`
- LLM 适配部分依赖较新的 PyTorch/Transformers
- 这两套依赖直接混在一起很容易冲突

实际意义：

- 你生成经验池时，要进 baseline 环境
- 你训练/测试大模型时，要进 PLM 环境

### 4.2 第二步：baseline 如何从头跑到尾

入口文件是 `run_baseline.py`。

#### 4.2.1 参数检查和资源定位

代码先检查：

- `--model` 是否在支持列表中
- `--test-trace` 是否在 `cfg.trace_dirs`
- `--video` 是否在 `cfg.video_size_dirs`

然后从 `config.py` 拿到：

- baseline checkpoint 路径
- 测试 trace 目录
- 视频 chunk 大小目录

作用：

- 保证整个测试链路用的是同一套数据定义
- 避免硬编码路径散落在各处

#### 4.2.2 加载 trace 和视频尺寸

`baseline_special/utils/utils.py` 里的 `load_traces()` 会读取：

- 每条 trace 的时间序列
- 每个时刻的带宽序列
- 文件名
- 可选的 `mahimahi` 起始指针

同时环境还会从 `video_size_0 ~ video_size_5` 读入每一档码率下每个 chunk 的大小。

为什么这一步重要：

- trace 决定“网络怎么抖动”
- video size 决定“同一个 chunk 在不同码率下下载代价多大”
- 没有这两个输入，就不能真实模拟 ABR 决策的后果

#### 4.2.3 创建环境 `Environment`

核心在 `baseline_special/env.py`。

环境的职责是：

- 接收当前选择的码率
- 在给定网络 trace 上模拟这个 chunk 的下载过程
- 计算下载时延、是否卡顿、buffer 如何变化
- 返回下一步决策所需的所有信息

这里有几个关键细节：

- `fixed=False` 时：
  - trace 顺序会打乱
  - 时延会乘上 `[0.9, 1.1]` 的随机噪声
- `fixed=True` 时：
  - trace 顺序固定
  - 不加时延噪声

为什么要区分 `fixed`：

- 训练/泛化时希望有随机性
- 测试/复现实验时希望尽量确定

另外，环境内部并不是简单地“chunk_size / bandwidth”：

- 它模拟了传输过程中每个时间片的可发送 payload
- 加了 RTT
- buffer 超过阈值时会 sleep
- trace 走到末尾时会回卷

所以它更像一个“可复现实验用的网络播放器仿真器”，而不是一个简化公式。

#### 4.2.4 根据模型类型初始化控制器

`run_baseline.py` 支持三类控制器：

- Pensieve/Genet/UDR：从 TensorFlow checkpoint 恢复 `ActorNetwork`
- MPC：提前枚举未来 5 步的码率组合，做滚动规划
- BBA：不需要模型，只用 buffer 阈值规则

为什么要这样分：

- Genet/UDR 是已有神经策略 checkpoint，直接恢复推理即可
- MPC 是经典 model-based 规划法，需要在线评估未来组合
- BBA 是经典启发式基线，用来提供最简单的对照组

#### 4.2.5 进入主循环：每个 chunk 做一次决策

这是 baseline 和 PLM 共享的核心时间轴：

1. 当前 `bit_rate` 已经确定
2. 调 `env.get_video_chunk(bit_rate)` 下载当前 chunk
3. 得到 `delay / buffer / rebuf / next_video_chunk_sizes / end_of_video`
4. 计算 reward
5. 更新 `6 x 6` 状态张量
6. 控制器根据新状态选择“下一 chunk 的码率”

为什么是这个顺序：

- ABR 决策本来就是“先决定下载哪个码率，再观察这个决定造成的后果”
- 所以 reward 属于“当前动作的后果”，下一步动作要基于新的状态再选

#### 4.2.6 各类 baseline 是怎么做决策的

`Pensieve / Genet / UDR`

- `actor.predict(state)` 输出 3 维 jump-action 概率
- 通过随机采样得到动作
- 再用 `action2bitrate()` 映射成真实码率

为什么不是直接取 argmax：

- 原始 Pensieve 策略本身就是带采样的
- 仓库保持了这种行为方式，以兼容原 checkpoint 的策略分布

`MPC`

- 取最近 5 步吞吐，做 harmonic mean
- 结合最近预测误差，构造一个更保守的未来带宽估计
- 枚举未来 5 个 chunk 的可行动作组合
- 选累计 reward 最大的组合，执行其中第一步

为什么要用 harmonic mean 和最大误差修正：

- harmonic mean 对突发低带宽更敏感，适合网络不稳定场景
- 再除以 `1 + max_error` 是一个“保守预测”策略，减少过度乐观导致的卡顿

`BBA`

- 只看 buffer
- 小于 reservoir：用最低码率
- 大于 reservoir + cushion：用最高码率
- 中间线性插值

为什么 BBA 仍然重要：

- 它非常简单，却经常是有竞争力的经典 heuristic baseline
- 用它可以看出“复杂模型到底带来了多少真实收益”

#### 4.2.7 保存结果

每个测试 trace 会生成一个 `result_sim_abr_*` 文件，记录：

- 时间戳
- 当前码率
- buffer
- rebuffer
- chunk size
- download delay
- smoothness
- reward

最后用 `calc_mean_reward()` 统一计算平均 reward。

为什么日志格式这么统一：

- baseline 和 PLM 都复用同一种结果格式
- 后处理和横向比较会更简单

### 4.3 第三步：经验池是怎么生成的

入口文件是 `generate_exp_pool.py`。

这是整个 NetLLM 训练链路里最关键的桥梁步骤。

#### 4.3.1 经验池生成的本质

它做的事情是：

- 让 baseline 控制器在环境里跑
- 收集每一步的 `(state, action, reward, done)`
- 把这些轨迹压平后存成 `exp_pool.pkl`

为什么要先做这个：

- NetLLM 不是直接在线 RL 训练
- 它走的是离线数据驱动路线
- 所以必须先有一批“老师策略跑出来的轨迹数据”

#### 4.3.2 收集时具体记录了什么

`collect_experience()` 在每个 chunk 记录：

- `state`：做当前决策之前的状态
- `action`：当前 chunk 采用的最终码率 `0~5`
- `reward`：这个动作产生的即时 reward
- `done`：视频是否结束

一个非常关键的细节：

- 对 Pensieve 类模型，经验池里存的不是 `3` 个 jump-action
- 而是映射后的最终码率 `0~5`

为什么这很重要：

- 后面的 PLM 训练目标就是“直接预测最终码率类别”
- 所以经验池必须统一成最终动作空间

#### 4.3.3 为什么每个 episode 的第一条样本会被丢掉

在 `generate_exp_pool.py` 里，episode 结束时保存的是：

- `states[1:]`
- `actions[1:]`
- `rewards[1:]`
- `dones[1:]`

也就是每个视频的第一步会被丢弃。

这样做的原因是：

- 第一 chunk 没有完整的历史上下文
- 平滑项里也没有真正意义上的“上一动作”
- baseline 测试和 PLM 评测里也都有“skip first reward”的逻辑

所以这是在强行对齐训练数据和评测口径。

当前仓库默认经验池的实际统计也能证明这点：

- `exp_pool.pkl` 里一共有 `19928` 条样本
- `done=True` 一共有 `424` 次
- `19928 / 424 = 47`

这正好说明：

- 每个 episode 原本有 `48` 个 chunk
- 丢掉第一个后，保留 `47` 条训练样本

#### 4.3.4 经验池的最终作用

经验池不是给 baseline 用的，而是给 `ExperienceDataset` 用的。

它相当于把网络交互问题转换成了：

> “给定过去上下文和目标 return，学习下一个码率决策”

这一步是 NetLLM 能成立的前提。

### 4.4 第四步：经验池如何变成训练数据

入口类是 `plm_special/data/dataset.py` 里的 `ExperienceDataset`。

#### 4.4.1 先归一化 reward

数据集会先把 reward 线性归一化到 `[0, 1]`：

- 保存原始 `min_reward / max_reward`
- 后续在线评测时也用同样的区间和 `scale` 做归一化

为什么要这么做：

- 不同 trace 下 reward 波动很大
- 直接把原始 reward 喂给 PLM 不稳定
- 在线测试时也必须用同一套数值尺度，否则 target return 会失真

#### 4.4.2 再计算 discounted return

数据集会按 episode 逐段计算：

- `R_t = r_t + gamma * R_{t+1}`

然后再除以 `scale`。

为什么要引入 return：

- 这个项目借鉴的是 Decision Transformer / return-conditioned policy 的思路
- 模型不是只学“看到状态就模仿动作”
- 而是学“看到状态，并知道自己还想达到多高回报时，该怎么动作”

#### 4.4.3 再记录 timestep

每个样本还会记录当前在 episode 内的时间步：

- 第几个 chunk
- 最大 timestep 多大

作用：

- 给时间嵌入使用
- 让模型知道当前是在视频开头、中间还是结尾

#### 4.4.4 再切成固定窗口

数据集最终不是一条条单步样本，而是长度为 `w` 的窗口。

默认逻辑：

- `w` 决定上下文窗口长度
- `sample_step` 决定采样步长
- 如果没设 `sample_step`，就等于 `w`

为什么要切窗口：

- PLM 擅长处理序列
- ABR 决策也依赖过去一段上下文

但这里有一个非常重要的实现细节：

- 数据集是直接在“压平后的经验池”上切窗口
- 它不会主动阻止窗口跨越 episode 边界

这意味着：

- 某些训练样本可能会同时包含“上一个视频尾部”和“下一个视频开头”
- `done` 虽然被保存在经验池里，但训练输入里并没有直接用到

这不一定是作者设计的主要目标，但它是当前代码的真实行为。

### 4.5 第五步：NetLLM 模型是怎么拼起来的

入口仍然是 `run_plm.py`。

#### 4.5.1 先加载基础 PLM

`plm_special/utils/plm_utils.py` 负责把不同模型类型映射到具体模型类。

当前配置里支持：

- `gpt2`
- `llama`
- `mistral`
- `opt`
- `t5-lm`
- `qwen`

当前仓库里和你本地文件最相关的是：

- `config.py` 里给 `qwen/base` 配了 `embed_size=1024`
- `config.py` 里给 `qwen/base` 配了 `layer_size=28`
- `plm_utils.py` 里 `qwen` 走的是 `Qwen3Model + AutoTokenizer`

为什么这里不是直接用文本 token：

- 这个项目不是做自然语言生成
- 它把 ABR 状态先变成 embedding，然后直接走 `inputs_embeds`
- 也就是说，PLM 在这里被当作“通用序列建模器”，不是普通聊天模型

#### 4.5.2 再套 LoRA

如果 `rank != -1`，就会在 PLM 上启用 LoRA。

主要逻辑：

- 冻结原始 PLM 参数
- 只在注意力里的 `q_proj / v_proj` 等模块上插低秩适配
- 打开 gradient checkpointing

为什么要这样做：

- 大模型全量微调成本太高
- ABR 是个小领域任务，没必要把全部参数都改动
- LoRA 能把训练重点放在“让模型学会做网络决策”，而不是重学整个语言模型

#### 4.5.3 再建状态编码器 `EncoderNetwork`

这个模块非常重要，因为它保留了 Pensieve 的领域先验。

它不是把 `6 x 6` 状态直接展平成一个向量，而是分通道处理：

- 上一码率：全连接
- 当前 buffer：全连接
- 吞吐历史：卷积
- 下载时延历史：卷积
- 下一 chunk 大小：卷积
- 剩余 chunk：全连接

为什么这种结构有价值：

- 不同通道的统计性质完全不同
- 过去吞吐/下载时间天然更像一维时间序列
- Pensieve 这套编码方式已经被验证过适合 ABR
- 所以 NetLLM 没有完全抛弃传统网络结构，而是先做“领域特征抽取”，再交给大模型

#### 4.5.4 再建离线 RL 策略 `OfflineRLPolicy`

这是整个 NetLLM 最核心的模块。

它会把一个时间步的信息拆成 8 段 embedding：

- `return embedding`
- `state embedding 1`
- `state embedding 2`
- `state embedding 3`
- `state embedding 4`
- `state embedding 5`
- `state embedding 6`
- `action embedding`

然后把多个时间步按下面顺序拼成一条序列：

`(R_1, s_1-1, s_1-2, ..., s_1-6, a_1, R_2, s_2-1, ..., a_2, ...)`

为什么这样拼：

- 这是一种 return-conditioned autoregressive 建模方式
- 让模型在“前面给了目标 return 和状态”的条件下，去预测动作
- 本质上是在把 ABR 决策问题改写成 Transformer 能擅长处理的序列预测问题

默认 `w=20` 时，每个样本的有效序列长度是：

- `20 * 8 = 160`

这对于当前 `qwen/base` 的 `1024` 维配置来说，远没有达到内部截断上限。

### 4.6 第六步：训练为什么看起来像分类，而不是在线 RL

这一点非常关键。

`run_plm.py` 里的训练不是在线 policy gradient，也不是 DQN。

它的训练流程是：

1. 从经验池读一个窗口
2. 取出 `states / actions / returns / timesteps`
3. 用 `process_batch()` 把动作标签整理成分类目标
4. `OfflineRLPolicy.forward()` 输出每个时间步的 6 类码率 logits
5. 用 `CrossEntropyLoss` 做监督训练

所以它更接近：

> “带 return 条件的行为克隆 + 序列建模”

而不是：

> “在线和环境反复交互的强化学习”

为什么这么做：

- 大模型在线 RL 的代价太高
- baseline 已经能提供质量不错的离线轨迹
- 用离线监督方式训练更稳定，也更容易复现

#### 4.6.1 `Trainer` 里关键训练步骤的意义

- `batch_size=1`
  - 代码里明确假设 batch 很小，主要是显存压力考虑
- `grad_accum_steps`
  - 用梯度累积模拟更大的 batch
- `clip_grad_norm_(..., 0.25)`
  - 防止训练不稳定
- `LambdaLR`
  - 用 warmup 让训练初期更平稳

为什么 checkpoint 和 evaluate 要周期性执行：

- checkpoint 用于保留中间状态
- evaluate 用于在真实环境里选“最好”的模型，而不是只看训练 loss

这很重要，因为：

- 训练 loss 变小不代表在线 ABR reward 一定更高
- 最终目标还是环境中的表现

### 4.7 第七步：训练中的在线验证是怎么做的

对应 `plm_special/evaluate.py`。

验证流程和 baseline 主循环很像：

- 创建同一个 `Environment`
- 用当前模型逐 chunk 地做决策
- 更新状态
- 计算 reward
- 把 reward 从目标 return 里减掉

这里最核心的思想是：

- `target_return` 表示“我希望这条轨迹最终达到的总回报”
- 每走一步，就扣掉已经得到的 reward
- 模型每一步都在“还剩多少目标没完成”的条件下做动作

为什么这一步关键：

- 它不是普通行为克隆
- 模型能通过 target return 获得一种“面向目标表现”的控制信号

### 4.8 第八步：最终测试怎么做

对应 `run_plm.py` 的 `--test` 和 `plm_special/test.py`。

流程是：

- 加载 best model 或你指定的 `--model-dir`
- 在相同环境中逐 chunk 推理
- 保存和 baseline 同格式的 `result_sim_abr_*`
- 用 `calc_mean_reward()` 算平均 reward

为什么测试和 baseline 复用同样的日志格式：

- 方便做公平对比
- 方便后续脚本统一统计

## 5. 项目里几个“为什么这么设计”的关键原因

### 5.1 为什么先用 baseline 生成经验池，再训练大模型

- baseline 已经掌握了较强的 ABR 行为模式
- 直接让 PLM 从零在线探索，成本高且不稳定
- 经验池相当于把专家/强基线策略蒸馏成了可监督学习的数据源

### 5.2 为什么状态编码器保留 Pensieve 风格

- ABR 状态不是自然语言
- 直接把原始数值硬塞给 PLM，效果未必好
- 先做领域结构化编码，相当于先把“网络控制问题”翻译成更适合 Transformer 的表示

### 5.3 为什么用 return-conditioned 序列建模

- 只模仿动作会让模型更像“平均行为”
- 加入 return，可以给模型一个性能目标
- 这让模型不是简单复制老师动作，而是在目标约束下做条件决策

### 5.4 为什么 baseline 和 PLM 都必须跑同一个 `Environment`

- 否则 reward 不可比
- 否则 trace 难度不一致
- 否则论文和实验结果很难有说服力

## 6. 当前仓库里已经存在的本地产物

### 6.1 默认经验池

当前目录里已经有默认经验池：

- `artifacts/exp_pools/exp_pool.pkl`

我在本地读到的统计是：

- 总样本数：`19928`
- `done=True` 次数：`424`
- reward 最小值：`-85.0127377757031`
- reward 最大值：`4.3`

这说明默认经验池不是空壳，而是已经可直接用于 `run_plm.py` 的真实训练数据。

### 6.2 已存在的 Qwen 微调结果

当前仓库里已经有一套 `qwen_base` 的微调目录：

- `data/ft_plms/qwen_base/.../early_stop_-1_best_model`

其中包含：

- `adapter_config.json`
- `adapter_model.bin`
- `modules_except_plm.bin`
- `README.md`

这说明：

- 这套仓库不是只停留在代码层
- 至少已经做过一次 Qwen 的 LoRA 微调

### 6.3 现有训练日志

已有 `early_stop_-1_console.log`。

从现有日志可以读到：

- 训练 loss 从一开始的较高值逐步下降
- `best_return` 至少达到过 `4.428757963045628`
- checkpoint 至少保存到了 `epoch 50`

这说明这套训练链路在作者环境里确实跑通过。

## 7. 必须知道的重要细节和隐藏坑

这一节是最值得你重点看的部分。

### 7.1 `run_plm.py` 的默认路径更像是“假设你在 `adaptive_bitrate_streaming` 目录内执行”

`config.py` 里的 `plm_dir` 拼法对当前仓库根目录执行方式并不稳妥。

实际影响：

- 如果你在仓库根目录直接运行 `python adaptive_bitrate_streaming/run_plm.py`
- `downloaded_plms` 的解析路径可能会指错

更稳的理解方式是：

- 这套 README 命令默认假设你先 `cd adaptive_bitrate_streaming`
- 再执行 `python run_plm.py`

### 7.2 `Environment.__init__()` 里把 numpy 随机种子写死成了常量

环境初始化时会执行：

- `np.random.seed(998244353)`

这会带来一个后果：

- trace 顺序打乱
- 初始 `mahimahi` 指针采样

这些初始化随机性，并不完全由命令行里的 `--seed` 控制。

为什么这是个重要细节：

- 你以为 `--seed` 改了，可能只改了后半段的随机行为
- 环境刚创建时的一些随机选择仍然是固定常量种子决定的

### 7.3 `generate_exp_pool.py` 在多个 Pensieve 类模型一起生成经验池时有实现风险

这是当前代码里一个很值得注意的地方。

`run()` 里有一个外层 `actor = None`，然后循环不同 `model_name` 调 `collect_experience()`。

但在 `collect_experience()` 里：

- 只有 `actor is None` 时才会从 checkpoint restore
- 一旦第一次加载过 Pensieve actor
- 后面再遇到 `udr_1 / udr_2 / udr_3 / genet / udr_real` 这类模型时，不会重新加载新 checkpoint

实际后果：

- 如果 `args.models` 同时包含多个 Pensieve 家族模型
- 后续模型可能仍然在用前一个模型的参数采样

这会直接影响经验池质量。

### 7.4 `evaluate.py` 和 `test.py` 没有在 episode 切换时清空 `OfflineRLPolicy` 的 deque

`OfflineRLPolicy` 明明实现了 `clear_dq()`，但代码里没有任何地方调用它。

影响是：

- 模型在一个视频结束后，上一条视频的历史 embedding 可能还残留在内部 deque 里
- 新 episode 虽然把 `state`、`timestep`、`target_return` 重置了
- 但模型内部上下文缓存没有同步清掉

这意味着：

- 跨 trace 的上下文泄漏是有可能发生的

这是理解测试行为时不能忽略的一个实现细节。

### 7.5 数据集窗口可能跨 episode 边界

前面提过，`ExperienceDataset` 直接按压平后的经验池切窗口。

影响是：

- 训练样本可能跨视频边界
- `done` 没有直接输入模型
- 只能靠 timestep 重置和 return 变化间接体现“episode 断开”

这不是传统 RL 数据管线里最严格的切法，但它是当前代码的真实行为。

### 7.6 baseline 和 PLM 都有“跳过第一步 reward”的口径

你会在多个地方看到类似逻辑：

- baseline 汇总时 `skip_first_reward=True`
- PLM 评测时 `if timestep > 0`
- 经验池保存时丢掉 `states[0]` 这一条

为什么这是一个统一口径：

- 第一 chunk 的历史上下文不完整
- 平滑项也带有初始化偏差
- 所以整个项目都在尽量避免把第一步纳入核心统计

### 7.7 根目录里的 `test.py` 不是正式评测脚本

`adaptive_bitrate_streaming/test.py` 的内容只是：

- 加载本地 Qwen tokenizer/model
- 打印类型

它的作用更像：

- 本地 sanity check
- 确认 `downloaded_plms/qwen/base` 能否正常被 transformers 读取

不要把它和 `plm_special/test.py` 混淆。

## 8. 如果你想按正确顺序理解或复现实验，建议顺序是这样

### 8.1 先看 baseline 侧

建议顺序：

- `config.py`
- `baseline_special/utils/constants.py`
- `baseline_special/env.py`
- `run_baseline.py`

先把这条线看懂，你就会明白：

- 状态从哪来
- reward 怎么算
- baseline 是怎么和环境交互的

### 8.2 再看经验池

建议顺序：

- `plm_special/data/exp_pool.py`
- `generate_exp_pool.py`
- `plm_special/data/dataset.py`

重点理解：

- 为什么经验池是 NetLLM 的数据源
- 为什么动作空间被统一成最终码率
- 为什么第一条样本被丢弃

### 8.3 最后看 NetLLM 主体

建议顺序：

- `plm_special/models/state_encoder.py`
- `plm_special/models/rl_policy.py`
- `plm_special/models/low_rank.py`
- `plm_special/utils/plm_utils.py`
- `plm_special/trainer.py`
- `plm_special/evaluate.py`
- `plm_special/test.py`
- `run_plm.py`

重点抓三件事：

- 状态怎样变成 embedding
- embedding 怎样被拼成 return-state-action 序列
- 模型怎样在同一个环境里被在线验证和测试

## 9. 最后给一个最短总结

如果只保留最关键的一句话，这个项目的完整流程就是：

1. 用 `Environment` 在真实带宽轨迹上模拟视频播放
2. 用 baseline 策略先跑出高质量轨迹
3. 把轨迹做成带 return 的离线数据集
4. 用状态编码器 + PLM + LoRA 学习“给定历史和目标回报时，下一步该选哪个码率”
5. 再把训练好的模型放回同一个 ABR 环境里做公平测试

如果再补一句最重要的实现理解：

> 这不是“让大模型直接读文本做 ABR”，而是“先把 ABR 状态做成结构化 embedding，再把 PLM 当成一个 return-conditioned 序列决策器来用”。
