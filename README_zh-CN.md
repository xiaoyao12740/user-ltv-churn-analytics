# 用户生命周期价值（LTV）预测与流失预警分析

[English](README.md)

这是一个基于**可复现合成数据**的端到端数据分析与机器学习求职展示项目，用于展示方法与工程流程，不代表任何真实公司或客户数据。

![项目流水线](reports/figures/01_project_pipeline.png)

## 项目概览

项目将 30,000 名模拟用户的历史行为转化为通过校验、可写入 MySQL 的数据表，并完成业务 KPI、留存 Cohort、防泄漏月度快照、LTV/流失预测、可执行用户分层和 Power BI 数据输出。下文所有模型数字均来自 seed=42 的真实流水线产物。

## 业务问题

增长团队需要同时回答：用户参与度是否健康、哪些用户可能流失、有限的挽留资源应投向哪里。本项目将描述性分析、预测模型与价值×风险运营矩阵连接为完整链路。

## 架构

`模拟数据 → 校验 → 可选 MySQL → KPI/留存 → 时间快照 → LTV/流失模型 → 分层 → Power BI CSV + 图片`

所有特征窗口严格结束于 `snapshot_date` 之前；流失观察之后 30 天，LTV 观察之后 90 天。

## 数据集

正式运行包含 30,000 用户、671,174 条事件、44,960 笔订单，日期范围为 2024-01-01 至 2025-06-30。内部用户类型仅影响生成时的活跃度、衰减、购买频率、客单价和退款，绝不提供给模型。数据还包含渠道、注册初期、周末、月份、噪声及长尾收入规律。

原始 CSV 被 Git 忽略，可通过以下命令重建：

```bash
python -m src.data.generate_data --users 30000 --seed 42
```

## 数据质量

校验会拒绝用户 ID 重复/空值、未知外键、注册前活动、负会话时长、非正订单金额，以及不在 `[0, amount]` 内的退款；严重问题会抛出清晰的 `ValueError`，并由测试覆盖。

## KPI 框架

Python 实现 DAU、WAU、MAU、DAU/MAU、净收入、ARPU、ARPPU、AOV 和付费率；`sql/04_analysis_queries.sql` 提供等价 MySQL 查询。正式结果：总净收入 **$4,141,910.05**，平均 MAU **16,583**，最新月 MAU **19,486**，最新 ARPU **$12.00**，最新付费率 **11.55%**。

![KPI 趋势](reports/figures/02_kpi_trends.png)

## 留存分析

严格按注册后第 N 天计算：D1 **15.52%**、D7 **15.24%**、D30 **7.09%**；同时输出月度 Cohort 以及渠道、设备留存数据。

![Cohort 留存](reports/figures/03_retention_cohort.png)

## 特征工程

月度快照包含 tenure、7/30/90 天活跃天数、事件与购买次数、近期收入、最近活跃/购买间隔、平均会话/订单、购买频率和历史 LTV。`feature_columns()` 明确排除两个目标字段。

## 基于时间的切分

快照覆盖 2024-07-01 至 2025-03-01：早期月份训练，倒数第二期验证和选模，最新期作为测试。禁止随机切分，快照之后的行为不会进入特征。

## LTV 预测

目标是未来 90 天净收入。Linear Regression 与 Random Forest 均用 `log1p(y)` 训练，再用 `expm1` 还原；生产候选由验证集 MAE 选择。

![LTV 分布](reports/figures/04_ltv_distribution.png)
![LTV 实际与预测](reports/figures/05_ltv_actual_vs_predicted.png)

## 流失预测

`churn_30d = 1` 表示快照后 30 天无有效事件。带类别权重的 Logistic Regression 提供可解释基线，Random Forest 捕获非线性。由于漏掉流失用户代价高且类别不平衡，业务重点是 **Recall 与 PR-AUC**，不能只看 Accuracy。

![ROC](reports/figures/06_churn_roc.png)
![PR 曲线](reports/figures/07_churn_pr_curve.png)
![混淆矩阵](reports/figures/08_confusion_matrix.png)
![特征重要性](reports/figures/09_feature_importance.png)

## 模型评估

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 25.894 | 86.940 | -0.073 |
| Random Forest | **25.281** | **82.324** | **0.038** |

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.477 | 0.333 | **0.842** | **0.477** | 0.638 | **0.385** |
| Random Forest | **0.556** | **0.356** | 0.699 | 0.472 | **0.638** | 0.381 |

项目保留了不漂亮但可信的结果：未来收入大量为零，行为有噪声，不同潜在人群存在真实重叠，因此 LTV R² 和流失 AUC 较温和，没有可疑的接近 1.0 指标。

## 用户分层

价值阈值来自所选模型在**训练期预测**的 75% 分位数（**$2.74**）；风险采用预先声明的 0.50 概率阈值，均未使用测试结果调参。

| 分层 | 推荐动作 | 用户数 |
|---|---|---:|
| Priority Retention | 人工触达并提供挽留优惠 | 1,529 |
| VIP Maintenance | VIP 权益与交叉销售 | 4,646 |
| Automated Reactivation | 自动化召回活动 | 20,656 |
| Growth/Nurture | 教育与下一最佳行动培育 | 3,169 |

![价值风险矩阵](reports/figures/10_value_risk_matrix.png)

## Power BI

`powerbi/` 提供 CSV 导入、关系、DAX 和四页设计：Executive Overview、Retention、Customer Value、Churn Risk。Python 不伪造 `.pbix` 或截图；完成报表后应导出为 `reports/figures/11_powerbi_dashboard.png`。

## 关键洞察

- 用户基数推动活跃增长，但 DAU/MAU 与 Cohort 更能揭示 MAU 背后的质量。
- D30 明显低于 D1/D7，长期召回是实际运营重点。
- Logistic 默认阈值召回 84.2% 的流失用户，同时付出预期的 Precision 代价。
- 最大人群为低价值/高风险；昂贵的人工挽留仅投向 1,529 名高价值/高风险用户。

## 仓库结构

```text
config/                 可复现配置
data/{raw,interim,processed}/  生成产物（CSV 被忽略）
notebooks/              EDA、KPI/留存、模型分析
src/                    生成、校验、数据库、分析、特征、模型、分层、制图
sql/                    MySQL 8 数据库、表、索引、分析查询
powerbi/                导入指南、DAX、Dashboard 设计
reports/{figures,metrics}/     提交的真实图片与 JSON 指标
tests/                  校验、分析、防泄漏、模型、分层测试
.github/workflows/      不依赖 MySQL 的 CI
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline --users 30000 --seed 42
pytest -q
```

macOS/Linux 使用 `source .venv/bin/activate`；开发调试可用 `--users 5000`。

## MySQL 设置

复制 `.env.example` 为 `.env`，仅在本地填写凭据，再依次运行 `sql/01_create_database.sql`、`02_create_tables.sql`、`03_indexes.sql`。本机没有可连接的 MySQL 服务，因此**未虚假声称完成数据库集成测试**；SQL 与客户端代码完整保留，CI 不依赖数据库。

## 可复现性

所有随机过程使用 seed/random_state 42，路径由 `pathlib` 推导，核心逻辑位于 `src/`。30k 正式流水线在 Windows CPU 环境耗时 **126.71 秒**。

## 测试

`pytest -q`：**11 passed**。覆盖完整性、时间逻辑、KPI/留存、快照边界、目标排除、两类训练流程和合法分层；GitHub Actions 会安装依赖并执行相同测试，不需要 MySQL 或 Power BI。

## 局限

合成数据无法还原所有真实混杂因素；收入高度零膨胀；尚无生产级校准和漂移监控；获客属性尚未进入快照模型；MySQL 写入未在本机集成验证；Power BI 需依据文档手工搭建。

## 未来工作

增加概率校准和成本敏感阈值、更丰富的季节/促销机制、生存分析、挽留 uplift 建模、漂移监控、可解释性、MySQL 集成测试及完整 Power BI 成品。

## 技术栈

Python 3.10+、pandas、NumPy、scikit-learn、matplotlib、SQLAlchemy、PyMySQL、pytest、Jupyter、MySQL 8+、Power BI、GitHub Actions。

## 许可证

MIT，详见 [LICENSE](LICENSE)。
