# PREISA / Semantic Reprogram 实验总览

本文档汇总当前 `llama_small + semantic_reprogram` 路线下所有已经完成并落盘的主要测试结果，包括：

- 纯语义基线
- `pre-align intra-step attention` 系列
- `state_to_prev_reward` 掩码系列
- 新增的 `TimeMixer` 版本
- 其它相关语义建模消融（`ISA` / `GISA` / `TSA` / `legacy`）

文中同时保留此前已经整理过的历史结论，并补上后续新增实验的结果分析。

## 指标说明

- `best_return`：训练过程中记录到的最佳验证回报
- `test mean_reward`：在 `fcc-test` 上的平均奖励
- `bitrate`：测试结果文件中统计得到的平均码率
- `rebuf`：测试结果文件中统计得到的平均重缓冲时间
- `smooth`：测试结果文件中统计得到的平均平滑性惩罚

## 口径说明

- 旧版 PREISA 表中的数值保留此前整理时采用的记录口径，便于和之前讨论保持一致。
- 本次新增的补充表，直接基于当前 `artifacts/results/.../llama_small/*` 下已经落盘的 100 条测试 trace 重新统计。
- 纯语义 `seed=100003` 的结果目录命名沿用了旧 tag：
  - `sr_sfd256_h4_isa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s100003_stop-1_tgt1`
  - 它在本文中统一视为 `semantic_only`。

## 总体结论

1. 在当前所有已完成结果中，`TimeMixer + preisa_prev_ar + maskprevreward` 是测试 `mean_reward` 最好的版本。
2. 在不加 `TimeMixer` 的 PREISA 系列里，`preisa_prev_ar + maskprevreward` 仍然是最强版本，也是第一个稳定超过纯语义基线的 PREISA 变体。
3. `state_to_prev_reward` 这条线的提升，核心来自更好的重缓冲控制；而 `TimeMixer` 的额外收益，则主要来自在保持较低重缓冲的同时，把码率和平滑性重新拉回到更均衡的位置。
4. 纯语义基线仍然非常强，尤其在单次最优 seed 上表现接近最优，但它的不同 seed 间波动较大。
5. 其它语义建模模块（`ISA` / `GISA` / `TSA`）目前都没有超过 `state_to_prev_reward` 主线，更没有超过加入 `TimeMixer` 的最新版本。

## 核心对比

| 版本 | 上下文设计 | 额外模块 | best_return | test mean_reward | 相比纯语义 | bitrate | rebuf | smooth | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `semantic_only` | 无 pre-align 时刻内注意力 | 无 | 4.5243 | 0.9167 | 0.0000 | 1081.0 | 0.0635 | 0.0756 | 强基线。整体策略偏激进，单次最优结果很强。 |
| `preisa_prev_ar_h8_hd1024_maskprevreward` | `state + prev_action + prev_reward` | `state_to_prev_reward` | 4.5235 | 0.9175 | +0.0008 | 1039.3 | 0.0528 | 0.0791 | 目前最强的非 TimeMixer PREISA 版本。靠更稳的重缓冲控制首次超过纯语义。 |
| `hmix_preisa_prev_ar_h8_hd1024_maskprevreward` | `state + prev_action + prev_reward` | `state_to_prev_reward` + `history multiscale mixer` | 4.5224 | **0.9212** | **+0.0045** | 1066.5 | 0.0165 | **0.0744** | 当前总榜最优。不是单靠降重缓冲，而是把码率、重缓冲、平滑性一起重新平衡到了更优点。 |

## 历史 PREISA 系列

下面这张表保留此前已经整理好的 PREISA 主线结果，便于和旧讨论保持一致。

| 版本 | 上下文设计 | 时刻内掩码 | Heads / Hidden | best_return | test mean_reward | 相比纯语义 | bitrate | rebuf | smooth | 特点与变化 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `semantic_only` | 无 pre-align 时刻内注意力 | 无 | - | 4.5243 | 0.9167 | 0.0000 | 1081.0 | 0.0635 | 0.0756 | 强基线。码率较高，同时重缓冲和平滑性仍保持在可接受范围内。 |
| `preisa_h8_hd1024` | `state + return + prev_action` | 无 | 8 / 1024 | 4.5228 | 0.9064 | -0.0103 | 1019.6 | 0.0533 | 0.0678 | 早期 `preisa` 中效果最好的一版。整体比基线更保守：码率更低、重缓冲更低、平滑惩罚更低。 |
| `preisa_h8_hd1024_mask` | `state + return + prev_action` | `context_readonly` | 8 / 1024 | 4.5229 | 0.9057 | -0.0111 | 1018.2 | 0.0533 | 0.0672 | 相比无 mask，训练更稳定，但最终 QoE 没有提升。平滑性略有改善，重缓冲几乎不变。 |
| `preisa_h8_hd2048` | `state + return + prev_action` | 无 | 8 / 2048 | 4.5207 | 0.8872 | -0.0295 | 1075.9 | 0.0655 | 0.0908 | 增大 FFN 容量后，策略变得更激进。码率回升了，但重缓冲和平滑性明显恶化。 |
| `preisa_h4_hd1024` | `state + return + prev_action` | 无 | 4 / 1024 | 4.5208 | 0.8897 | -0.0270 | 971.4 | 0.0443 | 0.0748 | 减少头数后，策略明显转向保守。重缓冲降得很多，但码率损失太大。 |
| `preisa_prev_ar_h8_hd1024` | `state + prev_action + prev_reward` | 无 | 8 / 1024 | 4.5225 | 0.9051 | -0.0116 | 1053.6 | 0.0619 | 0.0663 | 用 `prev_reward` 替换 `return` 后，码率更高、平滑性更好，但重缓冲也随之上升。 |
| `preisa_prev_ar_h8_hd1024_maskprevaction` | `state + prev_action + prev_reward` | `state_to_prev_action` | 8 / 1024 | 4.5194 | 0.9039 | -0.0128 | 1065.7 | 0.0616 | 0.0809 | 单独强调 `prev_action` 效果不理想。码率更高，但平滑性明显变差。 |
| `preisa_prev_ar_h8_hd1024_maskprevreward` | `state + prev_action + prev_reward` | `state_to_prev_reward` | 8 / 1024 | 4.5235 | **0.9175** | **+0.0008** | 1039.3 | **0.0528** | 0.0791 | 非 TimeMixer 路线里的最优版本。 |

## 本次新增：TimeMixer 版本分析

本次新增版本为：

- `sr_sfd256_h4_hmix_preisa_prev_ar_h8_hd1024_d0p1_maskprevreward_isa_off_tsa_off_r128_w20_g1_lr0p0001_wd0p0001_wu2000_e60_s100003`

它和此前最强的 `preisa_prev_ar_h8_hd1024_maskprevreward` 相比，最关键的变化是加入了：

- `--use-history-multiscale-mixer`

从 `seed=100003` 的结果看，`TimeMixer` 并不是简单把已有策略继续往“更保守、更低重缓冲”方向推，而是带来了更好的整体折中：

1. `mean_reward` 从 `0.9175` 提升到 `0.9212`。
2. `bitrate` 从 `1055.0` 回升到 `1066.5`。
3. `smooth` 从 `0.0807` 明显下降到 `0.0744`。
4. `rebuf` 虽然比非 `hmix` 版略高，但仍然显著优于大多数其它版本。

这说明在好的 seed 上，`TimeMixer` 更像是在改善历史时序信息的组织方式，让模型不必再通过过度保守的选码策略来换取低重缓冲，而是能在码率、平滑性和重缓冲之间找到更优的平衡点。

训练侧也有一个积极信号：

- `hmix` 版的 `best_return = 4.5224`
- 它非常接近历史最强 `preisa_prev_ar_h8_hd1024_maskprevreward` 的 `4.5235`
- 并且在后期多个 eval 点都稳定保持在 `4.51 ~ 4.52`

相比之下，旧版 `maskprevreward` 更像是“best checkpoint 选得很好”，而 `hmix seed=100003` 则更像是后期整体都比较稳，泛化可信度更高。

不过，后续补做的 `seed=100001` 和 `seed=100002` 结果也说明，这条线目前还存在明显 seed 敏感性：

- `hmix seed=100001` 的 `best_return = 4.5210`，训练验证回报并不低，最优点甚至出现在 `epoch 58`
- 但其最终测试 `mean_reward = 0.8206`
- 对应 `bitrate = 1089.0`、`rebuf = 0.0426`、`smooth = 0.0854`

这说明 `seed=100001` 这次虽然训练看起来正常收敛，但泛化明显掉了，主要问题不是码率不够，而是重缓冲和波动控制没有压住。换句话说，`TimeMixer` 当前体现出来的是“更高的上限”，但还没有表现出“比原主线更稳”的一致性。

`seed=100002` 的情况又是另一种风格：

- `hmix seed=100002` 的 `best_return = 4.5203`，最优点出现在 `epoch 34`
- 最终测试 `mean_reward = 0.8631`
- 对应 `bitrate = 1014.0`、`rebuf = 0.0110`、`smooth = 0.1035`

这个 seed 在测试时把重缓冲压得比较低，但代价是码率偏低、平滑性明显变差，最终 QoE 依然明显低于不加 `TimeMixer` 的 `seed=100002` 基线（`0.9076`）。这说明 `TimeMixer` 当前并不是稳定把原有最优折中往前推，而是在不同 seed 下会落到很不一样的策略风格上。

## 其它语义建模消融结果

下面这些结果是后续补做的相关语义实验，均直接从当前结果目录重新统计得到。

| 版本 | test mean_reward | bitrate | rebuf | smooth | 简评 |
|---|---:|---:|---:|---:|---|
| `gisa_h4_d0p1` | 0.8463 | 1000.5 | 0.0150 | 0.0898 | 比普通纯语义更保守，重缓冲较低，但整体 QoE 不足。 |
| `isa_h4_d0p1` | 0.8233 | 939.5 | 0.0098 | 0.0739 | 很稳，但码率损失太多。 |
| `isa_h4_d0p2` | 0.8644 | 1117.3 | 0.0355 | 0.1003 | 提高 dropout 后变得更激进，重缓冲和平滑性都恶化。 |
| `tsa_h4_d0p1_cm` | 0.8414 | 952.5 | 0.0016 | 0.1043 | 因果时间注意力极度抑制重缓冲，但代价是码率和波动控制都不理想。 |
| `tsa_h4_d0p1` | 0.6537 | 720.3 | 0.0022 | 0.0570 | 明显过于保守，整体不具竞争力。 |
| `legacy` | 0.5907 | 1116.8 | 0.0664 | 0.2408 | 作为旧编码器基线，重缓冲和平滑性都很差，已被新方案全面超越。 |

这些结果说明：

1. 单独堆 `ISA` / `GISA` / `TSA` 都还没有形成比 `pre-align + prev_reward` 更强的主线。
2. 它们大多是在不同方向上重排权衡：
   - 有的极度保守，重缓冲低但码率损失大
   - 有的重回高码率，但平滑性或重缓冲又恶化
3. 当前最值得继续投入的仍然是 `maskprevreward` 主线，尤其是其 `TimeMixer` 扩展版。

## Seed 敏感性

### 纯语义基线

| 版本 | seed | test mean_reward | bitrate | rebuf | smooth |
|---|---:|---:|---:|---:|---:|
| `semantic_only` | 100001 | 0.7923 | 922.5 | 0.0143 | 0.0686 |
| `semantic_only` | 100002 | 0.8552 | 1054.7 | 0.0285 | 0.0769 |
| `semantic_only` | 100003 | **0.9167** | 1097.7 | 0.0241 | 0.0772 |

### `maskprevreward` 主线

| 版本 | seed | test mean_reward | bitrate | rebuf | smooth |
|---|---:|---:|---:|---:|---:|
| `preisa_prev_ar_h8_hd1024_maskprevreward` | 100001 | 0.8902 | 1050.1 | 0.0203 | 0.0728 |
| `preisa_prev_ar_h8_hd1024_maskprevreward` | 100002 | 0.9076 | 1017.8 | 0.0092 | 0.0708 |
| `preisa_prev_ar_h8_hd1024_maskprevreward` | 100003 | **0.9175** | 1055.0 | 0.0132 | 0.0807 |

### `TimeMixer + maskprevreward` 主线

| 版本 | seed | test mean_reward | bitrate | rebuf | smooth |
|---|---:|---:|---:|---:|---:|
| `hmix_preisa_prev_ar_h8_hd1024_maskprevreward` | 100001 | 0.8206 | 1089.0 | 0.0426 | 0.0854 |
| `hmix_preisa_prev_ar_h8_hd1024_maskprevreward` | 100002 | 0.8631 | 1014.0 | 0.0110 | 0.1035 |
| `hmix_preisa_prev_ar_h8_hd1024_maskprevreward` | 100003 | **0.9212** | 1066.5 | 0.0165 | 0.0744 |

这里能看出两个现象：

1. 纯语义基线的 seed 波动其实不小，`100003` 明显优于 `100001 / 100002`。
2. `maskprevreward` 主线也有 seed 波动，但整体更稳，且在三个 seed 上都没有出现特别离谱的崩盘。
3. `TimeMixer` 目前呈现出“高上限但高波动”的特点：`seed=100003` 是当前最优，但 `seed=100001` 和 `seed=100002` 都明显掉点，且都不如不加 `TimeMixer` 的对应 seed 版本。

如果看单次最优结果，`TimeMixer` 仍然是当前总榜第一；但如果看目前已经完成的三 seed 均值：

- `TimeMixer + maskprevreward`：约 `0.8683`
- 非 `TimeMixer` 的 `maskprevreward` 主线：约 `0.9051`

那么 `TimeMixer` 现在还不能被认为“整体优于原主线”，更准确的说法应当是：

- 它有更高的单点上限
- 但目前三 seed 平均表现和稳定性都还不如原来的 `maskprevreward` 主线

因此，下一步如果还要继续推这条线，更合理的目标应该是“降低 seed 敏感性”，而不是直接把它当成已经稳定更强的方法。

## 当前排名

按当前已经落盘的主要完成版本排序：

1. `hmix_preisa_prev_ar_h8_hd1024_maskprevreward`：`0.9212`
2. `preisa_prev_ar_h8_hd1024_maskprevreward`：`0.9175`
3. `semantic_only`：`0.9167`
4. `preisa_h8_hd1024`：`0.9064`
5. `preisa_h8_hd1024_mask`：`0.9057`
6. `preisa_prev_ar_h8_hd1024`：`0.9051`
7. `preisa_prev_ar_h8_hd1024_maskprevaction`：`0.9039`
8. `preisa_h4_hd1024`：`0.8897`
9. `preisa_h8_hd2048`：`0.8872`

如果把其它探索性语义消融也算上，它们都还没有进入前三。

需要强调的是，这个排名按“单次已完成实验的最终 `mean_reward`”排序。若按当前三 seed 均值排序，`TimeMixer` 还不能排在 `maskprevreward` 主线之前。

## 实用结论

如果现在要选一条最值得继续跟进的路线，优先顺序是：

1. `hmix_preisa_prev_ar_h8_hd1024_maskprevreward`
2. `preisa_prev_ar_h8_hd1024_maskprevreward`
3. `semantic_only`

如果要写论文或报告，当前最自然的叙述线是：

1. 先以纯语义重编程作为强基线。
2. 再引入 `pre-align intra-step attention`。
3. 说明 `prev_reward` 比 `prev_action` 更适合作为上下文条件。
4. 说明 `state_to_prev_reward` 是第一个稳定超过纯语义的掩码设计。
5. 最后说明加入 `TimeMixer` 后，模型可以达到更高的单点上限，但当前还伴随较强的 seed 敏感性，因此它更像是“有潜力但尚未稳定”的增强方向，而不是已经稳定优于原主线的最终方案。
