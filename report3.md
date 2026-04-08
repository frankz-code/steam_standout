# Report v4 — 框架与实现细节

## 一、报告概览

| 项目 | 内容 |
|------|------|
| 标题 | 「慧眼识珠」：Steam爆款是否有迹可循？ |
| 副标题 | 从 Steam 真实数据出发，寻找下一个爆款的早期信号 |
| 输出文件 | `output/report.html`（~1MB，自包含交互式网页）+ `output/report_text.md`（文本稿） |
| 生成脚本 | `src/10_generate_report.py`（~2100行） |
| 数据来源 | Gamalytic API → 本地 parquet/json 文件 |
| 样本 | 30 款爆款 vs **3,783 款 backdrop 样本**（从 61,696 款 Steam 游戏中分层抽样，2023–2026） |
| 语言 | 全中文（游戏名有中文名时用中文，部分保留英文原名） |
| 设计风格 | 白底 + **深蓝色主色调**（Steam色盘 #1b2838/#2a475e/#171a21），卡片式排版 |

### v3 → v4 关键变更摘要

| 变更项 | v3 | v4 |
|--------|-----|-----|
| 筛选指标 | 巅峰周增速 (peak only) | **巅峰周增速 × 日常增速（双指标）** |
| 评估样本 | 300 对照组 + 部分 backdrop (742) | **3,810 游戏**（27 success + 3,783 backdrop） |
| 策略档位 | 4 档 (撒网/甜点/精选/高确信) | **3 档 (广角/精选/高确信)** |
| PR 曲线 | 单指标 sweep | 双指标 sweep (peak sweep + avg floor) |
| 策略评估 | 仅命中率/不漏球率 | **新增 Revenue-based 评估**（收入中位数、Lift、Cost-Efficiency） |
| "覆盖率"用词 | 覆盖率 | **不漏球率** |
| "误判"用词 | 误判 | **看走眼？** |
| 散点图样本 | 300 对照组 | **全量 backdrop 样本** |
| "深口袋假说" | 存在 | **删除** |
| "误判分析" | 存在 | **删除**（被策略质量分析替代） |
| KDE 分布图 | 存在 | **删除**（被双指标散点图替代） |

---

## 二、章节结构与对应图表

### Section 0：缘起——我们想找什么

**内容**（与 v3 相同）：
- 研究动机、「潜力爆款」定义、30 款标的列表
- 对照组逻辑（标签相似度 60% + 日期 20% + 价格 20%）
- 数据可靠性 callout

**图表**：无（纯表格 + 文字）

---

### Section 1：后视镜——数据里的初步发现

**内容**（与 v3 基本相同）：
- Cohen's d 方法论、核心发现（关注者增速 30d 最强）
- 「平平无奇」的指标、开发者历史 d=0.01
- 3 款低调爆款轨迹

**图表**：

| 图表 ID | 类型 | 内容 |
|---------|------|------|
| `chart_separation` | 横向柱状图 | 13 个指标的 Cohen's d（v3 不变） |
| `chart_trajectories` | 折线图 | Schedule I / R.E.P.O. / 小丑牌 轨迹（v3 不变） |

**v4 变更**：avg vs peak 散点图从 Section 1 移至 Section 2。

---

### Section 2：车前灯——提前识别爆款

**v4 大幅重构。** 叙事从"Peak > Avg → 用 Peak"改为"双指标各有区分力 → 组合更强"。

**内容结构**：

1. **前向指标：巅峰周增速 × 日常增速** — 介绍两个指标定义，Cohen's d 对比表（log 尺度），引入双指标散点图
2. **实例看看巅峰周增速的区分度** — 4 款游戏的 7d 滚动均增轨迹（ARC Raiders、漫威争锋、Manor Lords、Schedule I；v3 的幻兽帕鲁已移除）
3. **实例看看日常增速的区分度** — 4 个时间窗口的中位增速对比
4. **前向预判与后视镜场景的不同**（v4 新增）— 讨论 300 对照组 vs 3,810 真实样本的差异，说明分层抽样逻辑
5. **门槛定多少？命中率和不漏球率的取舍** — PR 曲线（双指标 sweep）
6. **三种策略，各取所需** — 3 档策略表 + 交互名单
7. **最高19%命中率，这个策略很棒吗？**（v4 新增）— Revenue-based 策略评估
8. **标准越高，发现越晚** — 检测提前量表

**图表**：

| 图表 ID | 类型 | 内容 | 数据来源 |
|---------|------|------|---------|
| `chart_avg_vs_peak` | 散点图 + 双椭圆 + 阈值线 | 3,810 游戏的日常增速 vs 巅峰周增速 | `peak_data_full` (histories/) |
| `chart_peak_annotated` | 折线图 + pin 标注 | 4 款游戏 7d 增速轨迹 | `forward_observations.parquet` |
| `chart_windows` | 分组柱状图 | 4 时间窗口中位增速 | `forward_observations.parquet` |
| `chart_threshold` | 双折线图 + 策略线 | 命中率 / 不漏球率随门槛变化 | `extract_threshold_sweep(avg_floor=20)` |
| `chart_strategy_quality` | 分组柱状图 | 3 策略的 >$10M / >$50M / Top-100 占比 | `extract_strategy_quality()` |

**已删除的 v3 图表**：`chart_peak_distribution`（KDE 分布图，被双指标散点图替代）

---

### Section 3：其他发现

**内容**：
- 标签画像（与 v3 相同）
- ~20% 爆款上线前完全隐身（致命公司等 4 款示例）
- 2×2 框架：增速 × 游戏质量
- 关注者 vs 营收散点图（**v4: 改用全量 backdrop 样本，移除 25K markLine，新增解读段落**）

**v4 删除**："所谓的误判，真的是误判吗？"（被 Section 2 的策略质量分析替代）、"深口袋假说被否定"

---

## 三、三档策略设计

### 方法论：阶梯百分位法 + 双指标

用 backdrop 数据集的百分位来设定阈值。Peak 使用稍高百分位，Avg 使用稍低百分位。

### 最终策略（3,810 游戏数据集）

| 策略 | Peak ≥ | Avg ≥ | Bkdrp P | 关注数 | 命中率 | 不漏球率 | Lift |
|------|--------|-------|---------|--------|--------|---------|------|
| 广角策略 | 400 | 20 | P86/P82 | 335 | 6.3% | 78% | 9x |
| 精选策略 | 900 | 40 | P93/P90 | 135 | 12.6% | 63% | 18x |
| 高确信策略 | 1,900 | 70 | P97/P95 | 58 | 19.0% | 41% | 27x |

### Revenue-based 策略评估

| 策略 | 收入>$10M | 收入>$50M | 收入中位数 | Top-100 | Review/Hit | Rev Lift |
|------|-----------|-----------|-----------|---------|------------|----------|
| 广角 | 45% | 17% | $10.7M | 25% | 2.2款 | 30x |
| 精选 | 73% | 35% | $36.4M | 53% | 1.4款 | 101x |
| 高确信 | 86% | 57% | $83.3M | 74% | 1.2款 | 231x |

随机选取中位数: $0.36M

---

## 四、数据处理管线

### 输入文件

| 文件 | 内容 | 用途 |
|------|------|------|
| `data/analysis_features.parquet` | 330 款游戏的上线前特征 | Cohen's d、标签分析 |
| `data/forward_observations.parquet` | 逐日观测（330 游戏） | 轨迹图、velocity windows |
| `data/backdrop_games.parquet` | 61,696 款 Steam 游戏元数据 | 游戏名、发行商、销量 |
| `data/histories/*.json` | **4,872 个历史文件**（3,819 有上线前数据） | 全量 peak/avg 速度计算 |
| `config/success_games.json` | 30 款爆款列表 | 分组标记 |

### 关键处理函数（v4 新增/修改）

| 函数 | 变更 | 输出 |
|------|------|------|
| `extract_peak_week_from_histories()` | **新增 revenue 字段** | 每游戏 peak_week / avg_daily / revenue |
| `extract_threshold_sweep(data, avg_floor)` | **新增 avg_floor 参数** | 双指标 PR 曲线数据 |
| `extract_strategy_tiers()` | **双阈值 (peak+avg)，3 档** | 策略命中/遗漏/看走眼？列表 |
| `extract_peak_cohens_d(data, full)` | **改用 log 尺度，全量 backdrop** | peak_d / avg_d + 中位数 + 倍率 |
| `extract_strategy_quality()` | **v4 新增** | 收入占比、中位数、Top-100、Cost-Efficiency |
| `extract_scatter_data(af, names, full)` | **改用全量 backdrop** | followers vs revenue 散点 |
| `extract_detection_leadtime()` | **阈值改为 400/900/1900** | 检测提前量 |

### 两套数据源

| 变量 | 来源 | 用途 |
|------|------|------|
| `peak_data` | `forward_observations.parquet`（330 游戏） | Section 1 轨迹图、peak annotated |
| `peak_data_full` | `histories/*.json`（3,810 游戏） | **策略评估、散点图、PR 曲线、质量分析** |

---

## 五、ECharts 图表配置

### 图表索引（v4）

| 序号 | 图表 ID | 类型 | 所属 Section |
|------|---------|------|-------------|
| 1 | `chart_separation` | 横向柱状图 | S1 后视镜 |
| 2 | `chart_trajectories` | 折线图 | S1 后视镜 |
| 2b | `chart_avg_vs_peak` | 散点图 + 双椭圆 + 阈值线 | **S2 车前灯**（v3 在 S1） |
| 4 | `chart_peak_annotated` | 折线图 + pin 标注 | S2 车前灯 |
| 7 | `chart_windows` | 分组柱状图 | S2 车前灯 |
| 5 | `chart_threshold` | 双折线图 + 策略线 | S2 车前灯 |
| — | 策略交互表 | HTML tab (3 tabs) | S2 车前灯 |
| Q | `chart_strategy_quality` | 分组柱状图 **(新)** | S2 车前灯 |
| — | 检测提前期 | HTML 表格 | S2 车前灯 |
| 6 | `chart_tags` | 横向柱状图 | S3 其他发现 |
| 8 | `chart_scatter` | 散点图（全量 backdrop） | S3 其他发现 |

### JS 变量映射（v4）

```javascript
const sepData           = d["sep_power"]
const trajectoryData    = d["trajectories"]
const controlEnvelope   = d["control_envelope"]
const peakDistribution  = d["peak_distribution"]       // success+control (330)
const peakDistributionFull = d["peak_distribution_full"] // full backdrop (3810)
const peakAnnotated     = d["peak_annotated"]
const thresholdData     = d["threshold_sweep"]
const tagData           = d["tags"]
const windowData        = d["velocity_windows"]
const scatterData       = d["scatter"]                  // v4: full backdrop
const stratQuality      = d["strategy_quality"]         // v4 新增
```

---

## 六、文件清单（v4）

```
src/10_generate_report.py      ← 报告生成主脚本 (~2100行)
src/11_dual_metric_analysis.py ← 双指标验证分析脚本 (v4 新增)
src/03c_refetch_backdrop_prerelease.py ← Backdrop 数据恢复 (v4 新增)
output/report.html             ← 交互式 HTML 报告 (~1MB)
output/report_text.md          ← 文本稿
output/dual_scatter.png        ← matplotlib 验证图
output/dual_pr_curves.png      ← PR 曲线验证图
output/spike_ratio_hist.png    ← Spike ratio 验证图
report3.md                     ← 本文件（v4 框架文档）
report2.md                     ← v2/v3 框架文档（含 v3→v4 变更日志）
```

---

## 七、术语对照

| 报告中文 | 英文 | 定义 |
|---------|------|------|
| 巅峰周增速 | Peak Week Velocity | 7d 滚动均增的最大值 |
| 日常增速 | Median Weekly Velocity | 7d 滚动均增的**中位数** |
| 命中率 | Hit Rate / Precision | 标记游戏中真正爆款的比例 |
| 不漏球率 | Catch Rate / Recall | 所有爆款中被标记到的比例 |
| 看走眼？ | False Positive | 被标记但非爆款的游戏（实际多为商业成功作品） |
| Lift | — | 命中率 / 基准率（随机选取的提升倍数） |
| Rev Lift | — | 策略收入中位数 / 随机收入中位数 |

---

## 八、已知问题

### 数据层
1. **Gray Zone Warfare steamId 已修复**（config 层），pipeline 未重跑（`forward_observations.parquet` 仍含旧 ID）
2. **PEAK (3527290)** 仅 3 天上线前数据（Steam 页面创建极晚），属于"shadow drop"类型，无法前向检测
3. **`avg_daily` 变量名不准确**：实际计算的是中位数（`rolling.median()`），展示标签已改为"日常增速（中位数）"

### 展示层
1. **report.html 体积 ~1MB**：嵌入了 3,810 游戏的 peak_data_full JSON，可考虑压缩或分离
2. **散点图标签重叠**：当多个爆款聚集时，名称标签可能遮挡
