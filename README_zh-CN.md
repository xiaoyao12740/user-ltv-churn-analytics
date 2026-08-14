# 用户生命周期价值（LTV）预测与流失预警分析

[![CI](https://github.com/xiaoyao12740/user-ltv-churn-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/xiaoyao12740/user-ltv-churn-analytics/actions/workflows/ci.yml)

[English](README.md)

这是一个基于**可复现合成数据**的端到端数据分析与机器学习求职展示项目，用于展示方法与工程流程，不代表任何真实公司或客户数据。

![项目流水线](reports/figures/01_project_pipeline.png)

## 项目概览

项目将 30,000 名模拟用户历史转化为经过验证的 MySQL 8 数据表，并完成 KPI、右删失 Cohort、标签成熟度快照、LTV/校准流失预测、用户分层和 Power BI 输出。下文所有数字均来自 seed=42 的正式流水线产物。

## 业务问题

增长团队需要同时回答：用户参与度是否健康、哪些用户可能流失、有限的挽留资源应投向哪里。本项目将描述性分析、预测模型与价值×风险运营矩阵连接为完整链路。

## 架构

`模拟数据 → 校验 → KPI/右删失留存 → 标签成熟度快照 → LTV/校准流失 → 分层 → Docker MySQL 对账 → Power BI CSV + 图片`

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

严格按注册后第 N 天计算：D1 **15.52%**、D7 **15.24%**、D30 **7.09%**。尚未达到观察期的 Cohort 单元格为 `NaN`，不会被错误显示为 0%；已经成熟但无人返回的单元格仍保留 0%。

![Cohort 留存](reports/figures/03_retention_cohort.png)

## 特征工程

月度快照包含 tenure、7/30/90 天活跃天数、事件与购买次数、近期收入、最近活跃/购买间隔、平均会话/订单、购买频率和历史 LTV。`feature_columns()` 明确排除两个目标字段。

## 基于时间的切分

快照覆盖 2024-07-01 至 2025-03-01，两类目标均只用 **2025-03-01** 作为最终测试快照。LTV 使用 2024-07 至 09 训练、2024-12-01 验证和 90 天 embargo；Churn 训练至 2024-12-01、2025-01-01 验证，并 embargo 2025-02。训练样本必须满足 `snapshot_date + label_horizon <= model_as_of_date`，同时防止特征泄漏和“当时尚未成熟的未来标签”泄漏。

## LTV 预测

目标是未来 90 天净收入。新增 Zero/Mean/Median baseline，并保留 Linear Regression、Random Forest；two-stage hurdle 模型计算 `P(正收入) × E(收入|正收入)`。生产模型只由验证集 MAE 选择，测试集不参与选模。

![LTV 分布](reports/figures/04_ltv_distribution.png)
![LTV 实际与预测](reports/figures/05_ltv_actual_vs_predicted.png)

## 流失预测

`churn_30d = 1` 表示快照后 30 天无有效事件。Logistic 与 RF 的分数使用验证集 Platt scaling 校准，业务阈值也只在验证集选择；JSON 同时保留固定 0.5 baseline。新增 prevalence、no-skill PR-AUC、Brier、Top10% Precision/Recall/Lift。

![ROC](reports/figures/06_churn_roc.png)
![PR 曲线](reports/figures/07_churn_pr_curve.png)
![混淆矩阵](reports/figures/08_confusion_matrix.png)
![特征重要性](reports/figures/09_feature_importance.png)
![概率校准](reports/figures/11_churn_calibration.png)
![提升图](reports/figures/12_churn_lift.png)

## 模型评估

| LTV 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Zero baseline | 25.341 | 87.729 | -0.091 |
| Mean baseline | 45.576 | 84.278 | -0.007 |
| Median baseline | 25.341 | 87.729 | -0.091 |
| Linear Regression | 25.673 | 87.636 | -0.089 |
| Random Forest（选中） | **25.060** | 82.517 | 0.035 |
| Two-stage hurdle | 35.009 | **73.604** | **0.232** |

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic（阈值 0.220） | 0.463 | 0.337 | **0.867** | 0.486 | 0.634 | **0.393** |
| Random Forest（阈值 0.215） | **0.469** | **0.339** | 0.863 | **0.487** | **0.635** | 0.389 |

测试集流失率/no-skill PR-AUC 为 **0.292**；Logistic PR-AUC 为 **0.393**（1.346 倍），Brier 为 **0.197**，Top10% lift 为 **1.519 倍**。Hurdle 改善 RMSE/R²，但 MAE 更差，因此验证集仍选择 RF。

## 用户分层

价值阈值为开发期预测 75% 分位数（**$2.79**）；校准风险阈值 **0.220** 只由验证集 F1 选择，未使用测试结果调参。

| 分层 | 推荐动作 | 用户数 |
|---|---|---:|
| Priority Retention | 人工触达并提供挽留优惠 | 1,517 |
| VIP Maintenance | VIP 权益与交叉销售 | 4,565 |
| Automated Reactivation | 自动化召回活动 | 21,014 |
| Growth/Nurture | 教育与下一最佳行动培育 | 2,904 |

![价值风险矩阵](reports/figures/10_value_risk_matrix.png)

## Power BI

`powerbi/` 提供 CSV、关系、DAX 和四页设计。Python 不伪造 `.pbix` 或截图；完成后应导出为 `reports/figures/13_powerbi_dashboard.png`。

## 关键洞察

- 用户基数推动活跃增长，但 DAU/MAU 与 Cohort 更能揭示 MAU 背后的质量。
- D30 明显低于 D1/D7，长期召回是实际运营重点。
- 校准 Logistic 在验证阈值下召回 86.7% 流失用户；Top10% 风险名单比随机目标浓度高 1.52 倍。
- 最大人群为低价值/高风险；昂贵人工挽留仅投向 1,517 名高价值/高风险用户。

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

Docker MySQL 8.4 已真实实现并验证：

```bash
docker compose up -d
python -m src.pipeline --users 30000 --seed 42 --mysql
set RUN_MYSQL_TESTS=1 && python -m pytest -q -m integration
```

正式运行写入 30,000 users、671,174 events、44,960 transactions、228,350 snapshots；SQL 与 pandas 净收入均为 **$4,141,910.05**，行数完全一致且外键孤儿数为 0。默认 CI 仍不依赖 MySQL。

## 可复现性

所有随机过程使用 seed/random_state 42，路径由 `pathlib` 推导。30k + MySQL 正式流水线在 Windows CPU 环境耗时 **274.85 秒**。

## 测试

`pytest -q`：**17 passed，1 个 integration 默认跳过**；显式 MySQL 测试：**1 passed**。CI 在 Python 3.10/3.11/3.12 上运行单元测试，不依赖 MySQL 或 Power BI。

## 局限

合成数据无法还原所有真实混杂因素；收入高度零膨胀；Platt 校准尚无生产监控；当前只报告一个最终评估快照；Docker integration 未进入默认 CI；Power BI 仍需依据文档手工搭建。

## 未来工作

完成 Power BI 成品，增加 rolling-origin 多轮回测、成本敏感阈值模拟、校准/漂移监控及更丰富的季节/促销机制。没有 treatment/control 结果前不做虚假的 uplift。

## 技术栈

Python 3.10+、pandas、NumPy、scikit-learn、matplotlib、SQLAlchemy、PyMySQL、pytest、Jupyter、MySQL 8+、Power BI、GitHub Actions。

## 许可证

MIT，详见 [LICENSE](LICENSE)。
