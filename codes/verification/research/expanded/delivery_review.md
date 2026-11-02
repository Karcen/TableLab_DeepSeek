# 扩展实验交付复核

2026-11-02。实验未因图形或文字检查而重新选择参数。原始数据、冻结实现和旧实验记录分别保留。

- 1,791 项主实验组合完成；6,129 项外层指标独立复算一致。279 对网格/TPE 搜索轨迹、78 份预处理缓存及内外层分折通过核对。
- Adult 后续扩展测试的 105 项指标复算一致，另以独立实现核对 147 项重抽样 AUC。所有模型先锁定，再预测；既有测试文件的使用历史已披露。
- 108 次拟合诊断的 756 项指标复算一致；260 个选中求解器组件没有触及迭代上限。四十轮网络不据此声称已收敛。
- 旧研究的 546 项指标重新复算一致，原有六项协议测试通过；未改旧拟合对象。
- SHAP 覆盖 43,893 条外层验证记录，加和最大差 4.66e-10，分类解释对数几率，回归使用目标单位。
- 论文 44 页、Madrid 展示稿 20 页。XeLaTeX/BibTeX 编译通过，无缺字、未定义引用或溢出。最终嵌入字号最低分别为 5.98pt、5.14pt，未发现页外文本。
- 通读摘要、结果、方法、结论、中英文 README 与汇总说明，保留试验范围和确定程度，删去重复或含混句子。房价与学生的接近均值没有写成稳定胜出；NBA 两种任务没有跨样本比较网络收益；Adult 未将测试最高者替换为预选模型。

## Nature 图形检查

使用用户指定 Nature Skills，Python/matplotlib 绘图，PDF/SVG 文本可编辑，PNG 为 600dpi。源代码检查无 FAIL；唯一 WARN 是未导出 TIFF。本次交付为课程论文和展示稿，使用矢量 PDF，PNG 仅作预览，未声称符合期刊投稿上传格式。

17 张最终图均重新导出并通过 PDF 字号、碰撞检查；没有 FAIL 或 WARN 碰撞。全部可比较面板的最终绘图区按 1.5pt 容差检查。色条不作为数据面板参与等宽等高比较；单轴矩阵的多面板对齐为不适用。字号还在最终论文和展示稿中重新检查，修复了缩放后低于 5pt 的一处图。

每个面板与组图均已查看。网格/TPE、拟合曲线及 SHAP 使用三折均值与标准差；三个折的训练数据重合，误差线不作为独立重复的置信区间。消融矩阵保留三个折的配对变化，不只显示均值。局部解释展示实际中位误差记录，不是多次重复的均值。Adult 测试使用 2,000 次配对指纹组 bootstrap 的百分位区间，未作多重比较校正，区间条件于已拟合模型。

展示图选择规则见各页图注：排行榜只显示网格前七项及常数，特征消融展示六项指定参照，Adult 展示预选模型和固定参照。论文、表格及 CSV 保留全部条目。对称线性色标包含全部逐折值，未截断不利结果。

| 图 | 面板 | 本面板内容 | 中心与变异 | 对齐 | 碰撞及视觉 |
| --- | --- | --- | --- | --- | --- |
| ablation | a | 房价字段组依赖 | 三个配对折原值 | PASS | 通过 |
| ablation | b | 收入字段组依赖 | 三个配对折原值 | PASS | 通过 |
| ablation | c | NBA 字段组依赖 | 三个配对折原值 | PASS | 通过 |
| ablation | d | 学生字段组依赖 | 三个配对折原值 | PASS | 通过 |
| architecture | a | 五种部件控制与四任务的配对变化 | 三个配对折原值 | NOT APPLICABLE | 通过 |
| benchmark | a | 房价模型与两类搜索 | 三折均值 ± 标准差 | PASS | 通过 |
| benchmark | b | 收入模型与两类搜索 | 三折均值 ± 标准差 | PASS | 通过 |
| benchmark | c | NBA 原任务模型与两类搜索 | 三折均值 ± 标准差 | PASS | 通过 |
| benchmark | d | 学生模型与两类搜索 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_ablation | a | 房价字段组依赖 | 三个配对折原值 | PASS | 通过 |
| deck_ablation | b | 收入字段组依赖 | 三个配对折原值 | PASS | 通过 |
| deck_architecture | a | 五种部件控制与四任务的配对变化 | 三个配对折原值 | NOT APPLICABLE | 通过 |
| deck_benchmark | a | 房价模型与两类搜索 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_benchmark | b | 收入模型与两类搜索 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_benchmark | c | NBA 原任务模型与两类搜索 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_benchmark | d | 学生模型与两类搜索 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_holdout | a | CV 预选与全部测试条目 | 95% 配对分组 bootstrap 区间 | PASS | 通过 |
| deck_holdout | b | 预选减去固定参照的配对差值 | 95% 配对分组 bootstrap 区间 | PASS | 通过 |
| deck_local | a | 房价单条预测分解 | 单条记录，无重复误差线 | PASS | 通过 |
| deck_local | b | 收入单条对数几率分解 | 单条记录，无重复误差线 | PASS | 通过 |
| deck_neural_training | a | 房价固定轮数训练与验证 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_neural_training | b | 收入固定轮数训练与验证 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_neural_training | c | NBA 固定轮数训练与验证 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_neural_training | d | 学生固定轮数训练与验证 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_sequence | a | 相同历史上的十五项比较 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_sequence | b | 相同目标记录上的历史与字段删减 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_shap | a | 房价字段贡献 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_shap | b | 收入字段贡献 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_shap | c | NBA 字段贡献 | 三折均值 ± 标准差 | PASS | 通过 |
| deck_shap | d | 学生字段贡献 | 三折均值 ± 标准差 | PASS | 通过 |
| fit_diagnostics | a | 房价样本量 | 三折均值 ± 标准差 | PASS | 通过 |
| fit_diagnostics | b | 房价树深度 | 三折均值 ± 标准差 | PASS | 通过 |
| fit_diagnostics | c | 收入样本量 | 三折均值 ± 标准差 | PASS | 通过 |
| fit_diagnostics | d | 收入树深度 | 三折均值 ± 标准差 | PASS | 通过 |
| fit_diagnostics | e | NBA 样本量 | 三折均值 ± 标准差 | PASS | 通过 |
| fit_diagnostics | f | NBA 树深度 | 三折均值 ± 标准差 | PASS | 通过 |
| fit_diagnostics | g | 学生样本量 | 三折均值 ± 标准差 | PASS | 通过 |
| fit_diagnostics | h | 学生树深度 | 三折均值 ± 标准差 | PASS | 通过 |
| holdout_extension | a | CV 预选与全部测试条目 | 95% 配对分组 bootstrap 区间 | PASS | 通过 |
| holdout_extension | b | 预选减去固定参照的配对差值 | 95% 配对分组 bootstrap 区间 | PASS | 通过 |
| local | a | 房价单条预测分解 | 单条记录，无重复误差线 | PASS | 通过 |
| local | b | 收入单条对数几率分解 | 单条记录，无重复误差线 | PASS | 通过 |
| neural_training | a | 房价固定轮数训练与验证 | 三折均值 ± 标准差 | PASS | 通过 |
| neural_training | b | 收入固定轮数训练与验证 | 三折均值 ± 标准差 | PASS | 通过 |
| neural_training | c | NBA 固定轮数训练与验证 | 三折均值 ± 标准差 | PASS | 通过 |
| neural_training | d | 学生固定轮数训练与验证 | 三折均值 ± 标准差 | PASS | 通过 |
| sequence | a | 相同历史上的十五项比较 | 三折均值 ± 标准差 | PASS | 通过 |
| sequence | b | 相同目标记录上的历史与字段删减 | 三折均值 ± 标准差 | PASS | 通过 |
| shap | a | 房价字段贡献 | 三折均值 ± 标准差 | PASS | 通过 |
| shap | b | 收入字段贡献 | 三折均值 ± 标准差 | PASS | 通过 |
| shap | c | NBA 字段贡献 | 三折均值 ± 标准差 | PASS | 通过 |
| shap | d | 学生字段贡献 | 三折均值 ± 标准差 | PASS | 通过 |
