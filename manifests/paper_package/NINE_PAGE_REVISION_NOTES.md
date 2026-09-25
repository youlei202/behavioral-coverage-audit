# 正文九页修订

## 版面

- 修改基底：`ICLR2027_Behavioral_Coverage_ClaimDriven_Source.zip`，未回退到早期重建图版本。
- 科学正文占用第 1–9 页；第 9 页最后一段是 Discussion，参考文献从第 10 页开始。
- 没有改变会议样式、字号、行距、字距、页边距或已有面板尺寸。

## 用于补足正文的内容

1. **Phi sibling 对照。** 在 Section 4.2 具体给出 Phi sibling removal 与 non-sibling removal 的对照，说明 shared lineage 和实际重构依赖并非同一个概念。Section 4.3 补充 phi-4 本身的已有 group gains，与 Qwen 两方向的差别相呼应。
2. **完整选项向量的 removal 证据。** 将已存在的四行向量 sibling-removal 表移到 Section 4.4，作为正文 Table 3；表内所有数值不变。附录保留完整协议、全部目标结果和统计范围。
3. **标量系数到完整选项分布的迁移。** 展开既有 transferred-weight 结果，和 vector-refit 结果区分；保留 canonical vector 与 balanced scalar 设计不同的事实。
4. **其他视觉目标的 support 变化。** 展开 SIN+IN 与 SIN+IN→IN 两个已存档目标的上下文拟合变化，不再只描述 SIN 一个例子。其数值来自已有 geometry support 表。
5. **结尾精炼。** 将 Discussion 收束为一个连续结论段，避免重复罗列前文已定义的限定。

## 数据范围

没有新推理、新优化、重新拟合、新 bootstrap 或新生成的数据点。所有 `data/` 文件、26 个 standalone `.tex` 面板和会议样式均保持输入身份；只修改正文、表的位置/说明和对应文档。完整数据恒等检查与页面检查见 `qa/`。

## 复现

```bash
make replay
bash compile.sh
```

所有表格仍可由已有脚本从打包的结果重新导出；组合图仍使用 `subfigure`。
