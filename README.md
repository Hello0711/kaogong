# 中国公务员考试职位竞争分析与「性价比」职位推荐

基于 **2019–2026 年国考 + 31 省省考共 83 万条真实职位数据**，完成从数据清洗到机器学习建模落地的端到端数据分析项目。

## 项目亮点

| 模块 | 内容 | 体现能力 |
|------|------|----------|
| 1. 数据工程 | 合并 32 个异构 CSV（83 万行），清洗脏数据、解析 `3:1` 比例字段、规范化学历文本，工程化构造核心指标**报录比** | pandas、数据管道、特征工程 |
| 2. 多维分析 | 时间趋势、地域、学历、专业、部门五维竞争格局刻画 | 透视分析、Matplotlib/Seaborn 可视化 |
| 3. NLP 文本挖掘 | jieba 分词 + 专业要求结构化，构建专业「机会度」排名 | 中文文本处理 |
| 4. 机器学习 | 无泄漏建模预测报录比（梯度提升回归），排列重要性解释，落地「性价比职位推荐器」 | scikit-learn 建模、模型解释、应用落地 |

## 核心结果

- 数据规模：**831,407** 条职位记录，覆盖 **2019–2026** 年、**31** 个省份。
- 核心指标 **报录比 = 报名人数 / 招录人数**，全样本中位数约 **15**。
- 建模效果：HistGradientBoosting（log-R² **0.453**）显著优于线性回归基线（0.267）。
- 结论：招录人数、省份、部门系统、专业限制是决定岗位竞争强度的关键因素；「不限专业」岗位竞争远高于限定专业岗位。

## 目录结构

```
gongkao_analysis/
├── src/
│   ├── build_dataset.py      # 模块1：解压+合并+清洗+特征工程 -> data/clean/positions.parquet
│   └── build_notebook.py     # 用 nbformat 脚本化组装分析报告 notebook
├── notebooks/
│   └── gongkao_analysis.ipynb  # 主交付物：完整分析报告（含图表与建模）
├── data/
│   ├── raw/                  # 解压后的原始 CSV（自动生成）
│   └── clean/positions.parquet
├── outputs/
│   ├── figures/             # 8 张分析图表
│   └── tables/              # 导出的表格（如最卷岗位 TOP20）
├── requirements.txt
├── 简历要点.md               # 单独整理的简历可写能力关键词与成果
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建清洗数据集（需把 gongkao.7z 放在本项目上一级目录）
python src/build_dataset.py

# 3.（可选）重新组装 notebook
python src/build_notebook.py

# 4. 执行 notebook 生成带输出的报告
python -m nbconvert --to notebook --execute --inplace notebooks/gongkao_analysis.ipynb
# 或直接: jupyter notebook notebooks/gongkao_analysis.ipynb
```

## 数据字段说明（部分）

| 字段 | 含义 |
|------|------|
| `year` / `exam_type` | 年份 / 国考·省考 |
| `province` / `city` | 工作省份 / 城市 |
| `department_system` / `department` | 部门系统 / 招录部门 |
| `major_requirement` / `degree_requirement` | 专业要求（自由文本）/ 学历要求 |
| `employ_count` / `enrollment_count` | 招录人数 / 报名人数 |
| `report_ratio`（工程构造） | **报录比 = 报名人数 / 招录人数** |
| `min_degree_level` / `major_unlimited`（工程构造） | 最低学历等级 / 是否不限专业 |
