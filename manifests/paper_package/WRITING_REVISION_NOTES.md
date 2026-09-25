# 本轮分页修订

当前版正文完整占用第 1–9 页，参考文献从第 10 页另起。完整选项向量的 sibling-removal 表从附录前移为正文 Table 3；详见 `NINE_PAGE_REVISION_NOTES.md`。以下保留前一轮叙事修改的历史记录，表号以当前编译版为准。

# 本轮修改：让证据推进论点，而不是让正文逐条防守

## 1. 主线变为“覆盖的程度”与“覆盖的来源”

开篇从模型在集合中的不同作用进入问题。相近的重构误差可能来自一个关键 peer，也可能需要多个 peers。读者因此先理解为什么只报告残差不够，再理解 group gain 和 removal effects 的作用。

摘要和方法直接说明 primary LLM response 是 cyclic-option-balanced reference-answer probability，而不在大范围的 behavioral-equivalence 叙述后连续补限制。一个固定权重向量跨问题和干预重构整条响应序列；不会每道题单独拟合。

## 2. 三条贡献替代四条

Contributions 使用粗体引导句、简短斜体标签和完整论断：
- Collective response coverage：定义与诊断量；估计性质作为支撑。
- Directional and distributed relations：Qwen 的方向性关系与组合覆盖。
- Stress and localization：相同压力的不同方向，以及共享高残差输入。

投影唯一性和一致性没有删除，也没有新增证明；它们不再被包装成与核心实证发现等量齐观的突破。

## 3. 已有完整选项向量证据正式进入正文

新增 Section 4.4 与正文 Table 2，使用已完成的 V2.2 全选项向量审计，不增加推理或重新拟合。四个展示对象是 Qwen2.5、General Reasoner、Mistral 和 Gemma。完整八模型、两种权重路径以及 sibling-removal 结果保留在附录 C.3 的 Tables 7–8。

数字从原始 `trackwise_vector_results.parquet` 的 1,600 行结果导出。每个 overall 值在五个 dose 行中重复保存，因此先按 target/family/split/weight-source 去重，再平均十个 split，避免重复计数。CSV 使用 17 位浮点有效数字；其全部 23 个标量列与源 Parquet 解码结果逐值相同，16 个有记录的数值列也通过 footer extrema 检查。

该分析有独立的观测/拟合定义：canonical option order、五剂量、vector squared-loss group fit、fitting-TV selected single peer。没有把它冒充 balanced endpoint、loss-matched 或新 bootstrap 结果。正文保留 Gemma 的不同符号，说明两个已完成设置支持的是有选择的对应，而非全部结论无条件不变。

## 4. 让必要的选择自然进入定义

fixed convex class 被解释为在原响应尺度上组合现有 peer traces；权重非负、和为一、跨输入固定。signed 或 input-dependent 类对应另一个可表达性问题，放在讨论中明确其后续作用，而非在方法后堆叠免责声明。

逐 realization 的绝对残差平均直接写入方法。0.18964 对 0.17849 的聚合比较和 Jensen 区别保留于附录，移出正文估计部分，不再出现实验修订记录式中断。

## 5. 保留图，但调整其证据角色

Vision 的六面板主图和全部六个独立 PCA 附录图保留。几何、权重和样本级 residual 的来源不变，不新造区间或重复拟合 PCA。

Traffic 的完整散点图移入附录 D，正文保留简短、定义清楚的 coverage-versus-utility 比较。原始 negative coefficients、归一化敏感性、closest-peer selection 口径都仍可查。这不是删除不便结果，也不将有限案例升格为下游价值证明。

图形数值与上一份已修复版本保持一致；只有概念图拟合点标签和交通 Luzern 标签的文字位置做了微调，未移动数据点。

## 6. 没有以写作替代尚不存在的实验

本轮没有添加 unconstrained-linear baseline、小型非线性 baseline、下游 ensemble/pruning 实验、Vision 新 bootstrap 或 balanced-vector inference。对应问题没有被宣称为“已经实验解决”。通过正确定义主要对象、恢复已有向量证据、重排贡献与案例来提高论证质量。

## 7. 复现与版式

提供可编译 LaTeX、原始输入、数值导出、逐图来源及算术重放。所有主图与附录图仍由独立 standalone panel 构成，组合使用 `subfigure`。最终检查记录见 `qa/PAGE_BY_PAGE_QA.md` 与 `qa/final_build_checks.json`。
