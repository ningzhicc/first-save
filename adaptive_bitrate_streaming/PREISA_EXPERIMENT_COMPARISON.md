# PREISA 实验对比总结

本文档汇总了当前 `llama_small + semantic_reprogram` 设置下所有已完成的 `preisa` 实验，并与纯语义基线进行对比。

## 指标说明

- `best_return`：训练过程中保存到的最佳验证回报
- `test mean_reward`：在 `fcc-test` 上的平均奖励
- `bitrate`：在 `fcc-test` 上的平均码率
- `rebuf`：在 `fcc-test` 上的平均重缓冲时间
- `smooth`：在 `fcc-test` 上的平均平滑性惩罚
- 参考基线：`semantic_only`

## 总体结论

1. 最新的 `preisa_prev_ar + maskprevreward` 是当前效果最好的 `preisa` 变体。
2. 它也是第一个在最终测试 QoE 上轻微超过纯语义基线的 `preisa` 版本。
3. 之前的大多数 `preisa` 版本，本质上都在重新平衡码率、重缓冲和平滑性，但还不足以整体超过基线。
4. 各实验在训练阶段的 `best_return` 非常接近，但测试 QoE 的差异更明显，这说明不同 `preisa` 版本的主要区别不在于“能不能训起来”，而在于泛化出来的策略风格不同。

## 对比表

| 版本 | 上下文设计 | 时刻内掩码 | Heads / Hidden | best_return | test mean_reward | 相比纯语义 | bitrate | rebuf | smooth | 特点与变化 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `semantic_only` | 无 pre-align 时刻内注意力 | 无 | - | 4.5243 | 0.9167 | 0.0000 | 1081.0 | 0.0635 | 0.0756 | 强基线。码率较高，同时重缓冲和平滑性仍保持在可接受范围内。 |
| `preisa_h8_hd1024` | `state + return + prev_action` | 无 | 8 / 1024 | 4.5228 | 0.9064 | -0.0103 | 1019.6 | 0.0533 | 0.0678 | 早期 `preisa` 中效果最好的一版。整体比基线更保守：码率更低、重缓冲更低、平滑惩罚更低。 |
| `preisa_h8_hd1024_mask` | `state + return + prev_action` | `context_readonly` | 8 / 1024 | 4.5229 | 0.9057 | -0.0111 | 1018.2 | 0.0533 | 0.0672 | 相比无 mask，训练更稳定，但最终 QoE 没有提升。平滑性略有改善，重缓冲几乎不变。 |
| `preisa_h8_hd2048` | `state + return + prev_action` | 无 | 8 / 2048 | 4.5207 | 0.8872 | -0.0295 | 1075.9 | 0.0655 | 0.0908 | 增大 FFN 容量后，策略变得更激进。码率回升了，但重缓冲和平滑性明显恶化。 |
| `preisa_h4_hd1024` | `state + return + prev_action` | 无 | 4 / 1024 | 4.5208 | 0.8897 | -0.0270 | 971.4 | 0.0443 | 0.0748 | 减少头数后，策略明显转向保守。重缓冲最低，但码率下降过多。 |
| `preisa_prev_ar_h8_hd1024` | `state + prev_action + prev_reward` | 无 | 8 / 1024 | 4.5225 | 0.9051 | -0.0116 | 1053.6 | 0.0619 | 0.0663 | 用 `prev_reward` 替换 `return` 后，码率更高、平滑性更好，但重缓冲也随之增加，总体 QoE 仍未超过旧版最佳 `preisa`。 |
| `preisa_prev_ar_h8_hd1024_maskprevaction` | `state + prev_action + prev_reward` | `state_to_prev_action` | 8 / 1024 | 4.5194 | 0.9039 | -0.0128 | 1065.7 | 0.0616 | 0.0809 | 让 state 主要依赖 `prev_action` 的效果不够理想。虽然码率更高，但平滑性明显变差。 |
| `preisa_prev_ar_h8_hd1024_maskprevreward` | `state + prev_action + prev_reward` | `state_to_prev_reward` | 8 / 1024 | 4.5235 | **0.9175** | **+0.0008** | 1039.3 | **0.0528** | 0.0791 | 当前最优 `preisa`。虽然码率低于基线，但更好的重缓冲控制抵消了这部分损失。平滑惩罚仍略高于基线。 |

## 各实验逐项分析

### 1. `preisa_h8_hd1024`

这是第一个表现较强的 `preisa` 版本。它说明 pre-alignment 的时刻内 self-attention 是可行的，但它的主要作用是让策略变得更保守。整体 QoE 已经比较接近基线，但还没有真正超过基线。

### 2. `preisa_h8_hd1024_mask`

原始的 `context_readonly` 掩码确实改善了训练稳定性，也降低了 loss，但并没有带来更高的最终 QoE。这说明仅仅把两个上下文 token 固定成“只读条件”还不足以显著优化 state 和上下文之间的交互。

### 3. `preisa_h8_hd2048`

把 hidden dimension 从 `1024` 提到 `2048` 后，模块表达能力过强，模型明显变得更激进。平均码率提升了，但重缓冲和平滑性恶化得更多，最终 QoE 反而下降。这是“加大容量不一定更好”的典型例子。

### 4. `preisa_h4_hd1024`

把头数从 `8` 降到 `4` 会把策略推向另一个方向：更安全、更保守。重缓冲降得很明显，但码率损失太大，所以整体收益依然不理想。这一版说明 head 数会显著影响策略风格。

### 5. `preisa_prev_ar_h8_hd1024`

把 `return` 改成 `prev_reward` 后，上下文信息更贴近真实历史交互，也更符合部署时可获得的信息。相比旧的 `return + prev_action` 方案，它提升了码率，也改善了平滑性，但丢掉了低重缓冲优势。

### 6. `preisa_prev_ar_h8_hd1024_maskprevaction`

这一版验证了“单独强调 `prev_action` 是否足够”。结论是不够。模型对动作历史变得过于敏感，虽然码率更高了，但平滑性明显恶化。

### 7. `preisa_prev_ar_h8_hd1024_maskprevreward`

这是目前最有希望的一版。它说明在当前 `preisa` 设计下，`prev_reward` 比 `prev_action` 更适合作为上下文条件。最终测试 `mean_reward` 略微超过纯语义基线，主要原因是重缓冲降低得足够多，足以弥补一部分码率下降和稍差的平滑性。

不过也要注意，这一版后期训练评估有明显回落，所以它更像是“`best_model` 选点成功”，而不是“最后阶段稳定全面领先”。换句话说，这一版对模型保存时机比较敏感。

## 当前排名

按最终测试 `mean_reward` 排序：

1. `preisa_prev_ar_h8_hd1024_maskprevreward`：`0.9175`
2. `semantic_only`：`0.9167`
3. `preisa_h8_hd1024`：`0.9064`
4. `preisa_h8_hd1024_mask`：`0.9057`
5. `preisa_prev_ar_h8_hd1024`：`0.9051`
6. `preisa_prev_ar_h8_hd1024_maskprevaction`：`0.9039`
7. `preisa_h4_hd1024`：`0.8897`
8. `preisa_h8_hd2048`：`0.8872`

## 实用结论

如果现在要选一条最强的已完成 `preisa` 实验结果，优先使用：

- `preisa_prev_ar_h8_hd1024_maskprevreward`

如果要在论文里把整个探索过程讲清楚，最自然的叙述线是：

1. 先以纯语义重编程作为强基线。
2. 在 alignment 前加入时刻内 self-attention。
3. 说明容量变化（`heads`、`hidden_dim`）主要是在重新平衡码率与重缓冲。
4. 再说明上下文设计很关键：
   - `return + prev_action` 更稳定
   - `prev_action + prev_reward` 更贴近真实历史信息
5. 最后说明精细化掩码设计很重要。
6. 得出结论：`state_to_prev_reward` 是目前第一个在最终 QoE 上轻微超过纯语义基线的 `preisa` 掩码变体。
