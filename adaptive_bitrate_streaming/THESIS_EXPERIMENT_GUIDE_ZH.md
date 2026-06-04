# ABR 毕业论文实验整理与补实验指南

最后更新：`2026-05-07`

## 1. 文档用途

这份文档用于指导当前 `adaptive_bitrate_streaming` 目录下的 ABR 实验结果整理、论文主表设计、消融安排和后续补实验优先级。

建议把它当作后续执行清单来用，而不是只当作结论总结。

## 2. 当前已有结果概览

当前已经可以支撑论文实验章节的结果分成两类：

1. 外部基线
2. 自己方法的主线与消融

当前已经确认存在的外部基线结果目录：

- `artifacts/results/fcc-test_video1/trace_num_100_fixed_True/bba`
- `artifacts/results/fcc-test_video1/trace_num_100_fixed_True/mpc`
- `artifacts/results/fcc-test_video1/trace_num_100_fixed_True/genet`
- `artifacts/results/fcc-test_video1/trace_num_100_fixed_True/udr_3`

当前已经确认存在的自己方法结果目录位于：

- `artifacts/results/fcc-test_video1/trace_num_100_fixed_True/llama_small`

当前方法主线记录见：

- `PREISA_EXPERIMENT_COMPARISON.md`

## 3. 当前可直接写进论文的结果

### 3.1 主结果快照

下表中的 `mean_reward` 是当前最适合放入论文主比较表的核心指标。

| 类别 | 方法 | seed 数 | mean_reward | bitrate | rebuf | smooth |
|---|---:|---:|---:|---:|---:|---:|
| 传统方法 | `BBA` | 1 | `0.7642 ± 0.0000` | `1183.3021 ± 0.0000` | `0.0472 ± 0.0000` | `0.3974 ± 0.0000` |
| 传统方法 | `MPC` | 1 | `0.8485 ± 0.0000` | `1212.9479 ± 0.0000` | `0.0953 ± 0.0000` | `0.1375 ± 0.0000` |
| 强化学习 | `UDR_3` | 3 | `0.7299 ± 0.0221` | `1097.3125 ± 3.8095` | `0.0882 ± 0.0044` | `0.1684 ± 0.0017` |
| 强化学习 | `Genet` | 3 | `0.8673 ± 0.0285` | `1105.1945 ± 5.0840` | `0.0775 ± 0.0051` | `0.0880 ± 0.0030` |
| 自己方法 | `semantic_only` | 3 | `0.8547 ± 0.0622` | `1024.9667 ± 91.3062` | `0.0223 ± 0.0073` | `0.0742 ± 0.0049` |
| 自己方法 | `maskprevreward` | 3 | `0.9051 ± 0.0138` | `1040.9667 ± 20.2120` | `0.0142 ± 0.0056` | `0.0748 ± 0.0052` |
| 自己方法 | `hmix_v2lite` | 3 | `0.9248 ± 0.0229` | `1068.2333 ± 29.4274` | `0.0151 ± 0.0067` | `0.0783 ± 0.0052` |

### 3.2 当前最重要的论文结论

当前最适合写入论文正文的结论如下：

1. `hmix_v2lite` 是当前最强的完整方案，`3-seed mean_reward = 0.9248`。
2. `hmix_v2lite` 相比 `maskprevreward` 的 `0.9051` 提升约 `2.18%`。
3. `maskprevreward` 已经明显优于当前外部强化学习基线 `Genet` 的 `0.8673`。
4. `hmix_v2lite` 不仅超过传统方法和已有强化学习基线，也超过自己当前最强的非 TimeMixer 主线。

## 4. 论文里应该怎么放这些方法

### 4.1 正文主比较表建议

正文主比较表建议只放下面这些方法：

- `BBA`
- `MPC`
- `UDR_3`
- `Genet`
- `semantic_only`
- `maskprevreward`
- `hmix_v2lite`

这样做的好处是：

1. 有传统方法
2. 有强化学习方法
3. 有自己方法内部的强基线
4. 有最终最佳方法

这样主表故事会最完整，也最像标准论文写法。

### 4.2 `hmix` 该怎么处理

`hmix` 建议保留，但不要放在主结果表的核心位置。

推荐定位如下：

- 放在消融实验表中
- 或放在正文消融小节
- 如果篇幅很紧，可以挪到附录

不建议把 `hmix` 当成主结果方法，因为它没有带来稳定的整体提升。

当前对 `hmix` 的推荐表述是：

1. `hmix` 的最佳 seed 说明多尺度历史混合方向有潜力。
2. 但 `hmix` 的 3-seed 均值只有 `0.8683`，低于 `maskprevreward` 的 `0.9051`。
3. 同时它的波动更大，因此不能作为最终方案。
4. 不过它为后续 `hmix_v2lite` 的结构收缩与稳定性优化提供了直接动机。

### 4.3 消融实验表建议

消融表建议按下面顺序放：

| 顺序 | 方法 | 作用 |
|---|---|---|
| 1 | `semantic_only` | 纯语义重编程基线 |
| 2 | `maskprevreward` | 证明 `prev_action + prev_reward + pre-align` 有效 |
| 3 | `hmix` | 证明原始 multiscale mixer 有上限，但不稳定 |
| 4 | `hmix_v2lite` | 证明结构收缩后同时保留收益并改善稳定性 |

这个顺序的好处是：实验故事自然，动机链条完整，不需要额外解释为什么突然引入 `v2lite`。

## 5. 建议的论文实验章节结构

建议实验章节按下面顺序组织：

### 5.1 实验设置

这一节写清楚：

- 数据集与 trace 设置
- `fcc-train / fcc-valid / fcc-test`
- `video1`
- 统一 QoE 奖励公式
- 测试设置是 `trace_num_100_fixed_True`
- 结果指标为 `mean_reward / bitrate / rebuf / smooth`
- 除 `BBA` 和 `MPC` 外，主要学习型方法默认报告多 seed 结果

### 5.2 与传统方法和强化学习方法对比

这一节使用正文主表，重点讲：

- 自己方法是否优于传统方法
- 自己方法是否优于 RL 基线
- 最终最优方法是谁

### 5.3 自己方法内部消融

这一节使用消融表，重点讲：

- `semantic_only -> maskprevreward` 的收益
- 原始 `hmix` 的问题
- `hmix_v2lite` 为什么合理

### 5.4 稳定性分析

这一节重点讲 seed 波动。

当前已知的 `mean_reward` 标准差如下：

| 方法 | std |
|---|---:|
| `semantic_only` | `0.0622` |
| `maskprevreward` | `0.0138` |
| `hmix` | `0.0505` |
| `hmix_v2lite` | `0.0229` |

推荐解释方式：

1. `semantic_only` 有一定效果，但 seed 波动较大。
2. `maskprevreward` 是当前最稳的非 TimeMixer 主线。
3. 原始 `hmix` 波动明显放大，说明直接引入 multiscale mixer 会带来稳定性问题。
4. `hmix_v2lite` 显著优于 `hmix`，说明结构收缩是有效的。

### 5.5 单条 trace 或分布级分析

如果时间允许，建议补一组图：

- QoE CDF
- `hmix_v2lite` 对 `maskprevreward` 的 per-trace win-rate

这组图的收益很高，因为它能补强“不是只靠少数 trace 拉高均值”的论证。

## 6. 当前最推荐的后续工作顺序

### P0：必须先做

1. 把现有所有方法整理成一个总表。
2. 先完成正文主比较表。
3. 再完成消融表。
4. 确认论文里统一只用 `test mean_reward` 作为最终性能指标。

### P1：强烈建议补

1. 给 `maskprevreward` 和 `hmix_v2lite` 各补到 5-seed。
2. 做一张稳定性图，至少是误差棒图或箱线图。
3. 做一张 QoE CDF 或 win-rate 图。

### P2：如果时间够再做

1. 再补一个关键小消融。
2. 可优先考虑 `mask mode` 对比，或者 `prev_reward` 是否有效。
3. 如果还够时间，可以做 `video2` 泛化测试。

## 7. 不建议优先投入的方向

当前不建议优先做的事情：

1. 不建议再开很多新模型分支。
2. 不建议把大量时间花在 `hmix` 原版继续调参上。
3. 不建议先做复杂可视化而不先整理总表。
4. 不建议在正文主表里塞太多内部试验版本。

原因很简单：论文现在最缺的是规范整理和关键证据闭环，而不是更多探索性分支。

## 8. 推荐的数据整理方式

建议后续自己维护 3 份表：

### 8.1 `thesis_results_master.csv`

每一行对应一个 `method + seed`，建议列：

- `category`
- `method`
- `seed`
- `result_dir`
- `best_return`
- `best_epoch`
- `test_mean_reward`
- `bitrate`
- `rebuf`
- `smooth`
- `notes`

### 8.2 `thesis_main_table.csv`

每一行对应一个方法聚合结果，建议列：

- `method`
- `seed_count`
- `mean_reward_mean`
- `mean_reward_std`
- `bitrate_mean`
- `bitrate_std`
- `rebuf_mean`
- `rebuf_std`
- `smooth_mean`
- `smooth_std`

### 8.3 `thesis_ablation_table.csv`

专门用于方法内部比较，建议列：

- `method`
- `module_change`
- `mean_reward_mean`
- `mean_reward_std`
- `short_conclusion`

## 9. 推荐补图清单

建议最终至少准备下面这些图：

1. 主比较柱状图
2. 消融柱状图
3. 稳定性误差棒图
4. `maskprevreward` vs `hmix_v2lite` 的 QoE CDF
5. `maskprevreward` vs `hmix_v2lite` 的 per-trace win-rate 图

如果时间很紧，至少保住前 3 张。

## 10. 可直接照着执行的清单

下面这份清单建议后续逐项打勾：

- [ ] 从 `artifacts/results` 提取所有方法的最终指标
- [ ] 生成 `thesis_results_master.csv`
- [ ] 生成正文主比较表
- [ ] 生成消融表
- [ ] 生成稳定性图
- [ ] 生成主比较柱状图
- [ ] 生成消融柱状图
- [ ] 生成 `maskprevreward` vs `hmix_v2lite` 的 CDF 或 win-rate 图
- [ ] 决定是否补到 5-seed
- [ ] 决定是否补一个关键小消融
- [ ] 将 `hmix` 放入正文消融或附录

## 11. 最终建议

当前最稳妥的论文实验策略不是继续大规模探索，而是：

1. 用现有外部基线撑起主比较表
2. 用 `semantic_only / maskprevreward / hmix / hmix_v2lite` 撑起消融链条
3. 把 `hmix` 定位成“中间失败但有信息量的实验”
4. 把 `hmix_v2lite` 定位成“最终最优方案”

只要后续把表格、图和少量补实验补齐，这一套已经足够支撑毕业论文实验章节。
