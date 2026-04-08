# Report v2 — 框架与实现细节

## 一、报告概览

| 项目 | 内容 |
|------|------|
| 标题 | 「慧眼识珠」：Steam爆款是否有迹可循？ |
| 副标题 | 从 Steam 真实数据出发，寻找下一个爆款的早期信号 |
| 输出文件 | `output/report.html`（~200KB，自包含交互式网页）+ `output/report_text.md`（文本稿） |
| 生成脚本 | `src/10_generate_report.py`（1704行） |
| 数据来源 | Gamalytic API → 本地 parquet/json 文件 |
| 样本 | 30 款爆款 vs ~300 款对照（2023–2026，均为 >5万份销量） |
| 语言 | 全中文（游戏名有中文名时用中文，如"黑神话：悟空"、"幻兽帕鲁"） |
| 设计风格 | 白底 + **深蓝色主色调**（Steam色盘 #1b2838/#2a475e/#171a21），卡片式排版 |

### v1 → v2 关键变更

| 变更项 | v1 | v2 |
|--------|-----|-----|
| 标题 | 那些「骨骼清奇」的游戏 | 「慧眼识珠」：Steam爆款是否有迹可循？ |
| 主色调 | 绿色 (#065f46 hero gradient) | Steam深蓝 (#1b2838 hero gradient) |
| 成功表排序列 | 首月营收 ($M) | 累计销量 (万份) |
| Separation标签 | "关注者增速 30天" | "关注者增速 - 上线前30天" |
| Separation图表 | 按d值排序，无标记线 | 按类型分组(follower→wishlist→其他)，0.2/0.5/0.8 标记线 |
| Highlight文字 | 引用`sep_power[0]`（可能是wishlist 180d） | 明确引用`fol_30d`（follower 30d velocity） |
| 轨迹图上线后范围 | T+30天 | T+7天（压缩上线后，突出上线前形态差异） |
| Peak Week说明 | 无Cohen's d对比 | 展示Peak Week d vs Avg Daily d |
| 巅峰标注图游戏 | 全为晚期peak（庄园领主/缉私风云/Palworld/THE FINALS） | 混合早晚：ARC Raiders(T-178)/漫威争锋(T-245)/庄园领主/缉私风云/帕鲁 |
| 检测提前量 | 无 | **新图表**: 各策略≥90天/<90天/miss/无数据 堆叠柱状图 |
| 散点图 | 视口可能截断 | 固定min/max + containLabel修复 |
| 轴标签 | 可能与legend重叠或被裁剪 | grid padding + nameGap调整 |

---

## 二、技术栈

| 组件 | 方案 |
|------|------|
| 图表 | ECharts v5.5.0（CDN），支持中文渲染、交互 tooltip、响应式 |
| 样式 | 纯 CSS（CSS Custom Properties + Grid + Media Queries） |
| 布局 | 单列居中（max-width 900px），sticky 导航栏 |
| 响应式 | 3 个断点：>1024px（桌面）、768px（平板）、480px（手机） |
| 数据嵌入 | Python 提取后以 JSON 内嵌到 `<script>` 标签 |
| 字体 | 系统字体栈（PingFang SC → Microsoft YaHei → Noto Sans CJK） |

---

## 三、章节结构与对应图表

### Section 0：缘起——我们想找什么

**内容**：
- 研究动机：PEAK、R.E.P.O.、幻兽帕鲁引出核心问题
- 「潜力爆款」定义：超预期、小团队、新IP、63%自发行
- 30 款爆款标的列表（HTML 表格：游戏名 / **累计销量（万份）** / 发行商 / 发售日）
- 对照组逻辑：标签相似度 60% + 日期 20% + 价格 20%，每款爆款 ~10 个对照
- 数据可靠性 callout：follower = Steam 真实数据，wishlist/营收 = Gamalytic 估算

**图表**：无（纯表格 + 文字）

**数据来源**：
- `config/success_games.json` → 30 款爆款列表
- `data/backdrop_games.parquet` → `copiesSold` 累计销量、发行商信息
- `data/analysis_features.parquet` → 发售日

---

### Section 1：后视镜——数据里的初步发现

**内容**：
- Gamalytic 可用数据概述
- Cohen's d 方法论简介（大/中/小效应阈值）
- 核心发现：关注者增速-上线前30天是区分力最强的**真实数据**指标，明确引用 `fol_30d` 数据（d值、中位数、倍率）
- 增速 > 绝对数量的洞察
- 「平平无奇」的指标：售价（0.09）、开发者历史（0.01）、挂页天数（0.08）、标签数（0.25）
- 开发者历史 d=0.01 作为关键意外发现
- 3 款低调爆款的轨迹展示

**图表**：

| 图表 ID | 类型 | 内容 | 数据来源 |
|---------|------|------|---------|
| `chart_separation` | ECharts 横向柱状图 | 13 个指标的 Cohen's d 排名 | `analysis_features.parquet` → `extract_separation_power()` |
| `chart_trajectories` | ECharts 折线图 | 3 款低调爆款的关注者增长轨迹 + 对照组中位数灰线 | `forward_observations.parquet` → `extract_trajectories()` + `extract_control_envelope()` |

**Separation Power 图表细节**：
- 特征按类型分组显示：follower 类在一起（绿色），wishlist 类在一起（橙色），其他灰色
- 在图表内用 `markLine` 标注三条参考线：
  - x=0.2 → "小效应 0.2"（灰色虚线）
  - x=0.5 → "中效应 0.5"（琥珀色虚线）
  - x=0.8 → "大效应 0.8"（红色虚线）
- 特征列表（sorted by d descending within groups）：
  - 关注者增速 - 上线前30天 (follower, 绿色)
  - 关注者增速 - 上线前90天 (follower, 绿色)
  - 关注者增速 - 上线前180天 (follower, 绿色)
  - 上线时关注者数 (follower, 绿色)
  - 愿望单增速 - 上线前180天 (wishlist, 橙色)
  - 愿望单增速 - 上线前90天 (wishlist, 橙色)
  - 愿望单增速 - 上线前30天 (wishlist, 橙色)
  - 上线时愿望单数 (wishlist, 橙色)
  - 其他：开发者作品数/标签数/售价/挂页天数/开发者历史营收 (灰色)
- grid: left 35% 给标签留足空间
- tooltip 显示：指标名、d 值、爆款中位数、对照组中位数、样本量

**轨迹图表细节**：
- 展示游戏（steamId）：
  - `3164500` 缉私风云（Schedule I）
  - `3241660` R.E.P.O.
  - `2379780` 小丑牌（Balatro）
- 对照组灰线：所有对照组游戏关注者轨迹的中位数
- X 轴：距上线天数（倒序），上线后**截断到 7 天**（v1 是 30 天，v2 改为 7 天以突出上线前轨迹形态差异）
- X 轴 formatter: `T-N` / `上线` / `T+N`
- Y 轴：去掉 name 属性避免与 legend 重叠，用 nameLocation='end'
- grid: top 30, bottom 55, 给 legend 留空间

---

### Section 2：车前灯——提前通过"速度"识别爆款

**内容**：
- 后视镜指标的局限性（30d velocity 只能事后算）
- 前向指标对比：Peak Week Velocity vs Avg Daily Velocity
- **Peak Week 胜出 + Cohen's d 对比**：展示 Peak Week d 和 Avg Daily d 的具体数值
- 巅峰周时刻的可视化（混合早期 + 晚期 peak）
- 命中率 vs 覆盖率的永恒取舍
- 4 种策略框架（撒网/甜点/精选/高确信）
- 交互式策略表：4 个 tab 切换，每个 tab 显示命中/遗漏/误判/完整关注列表
- **新增：检测提前量图表** — 各策略下 ≥90天提前检测 / <90天检测 / 未检测 / 无数据
- 推荐甜点策略 callout

**图表**：

| 图表 ID | 类型 | 内容 | 数据来源 |
|---------|------|------|---------|
| `chart_peak_distribution` | ECharts 柱状图（直方图） | 巅峰周增速的分布：爆款（绿）vs 对照组（灰） | `forward_observations.parquet` → `extract_peak_week_data()` |
| `chart_peak_annotated` | ECharts 折线图 + 标注 | 5 款游戏的 7 日滚动均增轨迹，pin 标注巅峰周 | `forward_observations.parquet` → `extract_trajectory_with_peak()` |
| `chart_threshold` | ECharts 双折线图 | 命中率（绿）和覆盖率（蓝）随门槛变化曲线，X 轴对数坐标 | `extract_threshold_sweep()` |
| `chart_detection_lead` | **ECharts 堆叠柱状图（新）** | 4 策略 × 4 类别（≥90天/90天内/miss/无数据）的爆款检测分布 | `forward_observations.parquet` → `extract_detection_leadtime()` |

**Peak Annotated 细节（v2 更新）**：
- 展示游戏（混合早期和晚期 peak）：
  - `1808500` ARC Raiders（peak ~T-178，**极早**）
  - `2767030` 漫威争锋（peak ~T-245，**早**）
  - `1363080` 庄园领主（peak ~T-7，晚）
  - `3164500` 缉私风云（peak ~T-12，晚）
  - `1623730` 幻兽帕鲁（peak ~T-1，极晚）
- X 轴 formatter: `T-N` 格式

**Detection Lead Time 图表细节（v2 新增）**：
- 堆叠柱状图，X 轴 = 4 个策略标签
- 4 层堆叠：
  - "提前≥90天检测"（绿色 #10b981）
  - "<90天检测"（蓝色 #60a5fa）
  - "未达标"（琥珀色 #fbbf24）
  - "无数据"（浅灰 #cbd5e1）
- 每层在 bar 内显示数量
- 数据来源：`extract_detection_leadtime()` 函数
  - 遍历每个 success game 的 pre-launch follower 7d velocity
  - 找到首次 ≥ threshold 的 days_to_launch
  - ≥90 → early, 0-89 → late, never → missed, no data → nodata

**4 策略数据**：

| 策略 | 门槛 | 关注数 | 命中 | 命中率 | 覆盖率 | 适用场景 |
|------|------|--------|------|--------|--------|---------|
| 撒网 | ≥448/天 | ~92 | ~20 | ~22% | ~80% | 追踪成本低（看仪表盘） |
| 甜点 | ≥845/天 | ~50 | ~15 | ~30% | ~60% | 中等精力（调研、沟通） |
| 精选 | ≥1626/天 | ~29 | ~10 | ~34% | ~40% | 重投入（投资、合作） |
| 高确信 | ≥4140/天 | ~8 | ~4 | ~50% | ~16% | 只关注最有把握的 |

**交互策略表**：
- 4 个 tab 按钮（pill 样式），active = **深蓝描边 + 浅蓝底**（v2 从绿色改为深蓝主色调）
- 默认选中「甜点策略」
- 顶部 stat cards：关注列表总数 / 命中率 / 覆盖率 / 误判数
- 4 列网格（桌面）→ 2 列（平板）→ 1 列（手机）
- 列可滚动（max-height: 350px, overflow-y: auto）

---

### Section 3：其他发现

**内容**：
- 标签画像分析：什么类型更容易爆？
- 增速差距从什么时候开始存在？（velocity windows 4窗口对比）
- "误判"重新定义：大部分是 AAA 大作
- 约 20% 爆款上线前完全隐身（致命公司、恶魔轮盘、小丑牌、Megabonk）
- 2x2 框架：增速 × 游戏质量
- 关注者 vs 营收散点图
- 「深口袋」假说被否定

**图表**：

| 图表 ID | 类型 | 内容 | 数据来源 |
|---------|------|------|---------|
| `chart_tags` | ECharts 横向柱状图 | 标签频率差异（百分点），正=爆款更常见 | `analysis_features.parquet` → `extract_tag_analysis()` |
| `chart_windows` | ECharts 分组柱状图 | 4 个时间窗口的关注者中位增速，爆款 vs 对照组 | `forward_observations.parquet` → `extract_velocity_windows()` |
| `chart_scatter` | ECharts 散点图 | 上线时关注者数 vs 首月营收（对数坐标），爆款标名 | `analysis_features.parquet` → `extract_scatter_data()` |

**散点图修复（v2）**：
- 加入 `containLabel: true` 确保轴标签不被裁剪
- 固定 `xAxis min:100, max:1500000` 和 `yAxis min:0.1` 确保两组数据全部可见
- grid right 加至 40px

---

### Section: 研究局限性

5 条有序列表：样本量有限 / Gamalytic追踪缺口 / 生存者偏差 / 无因果推断 / 营收估算

---

## 四、数据处理管线

### 输入文件

| 文件 | 内容 | 用途 |
|------|------|------|
| `data/analysis_features.parquet` | 330 款游戏的上线前特征（velocity、counts、tags 等） | Cohen's d、标签分析、散点图 |
| `data/forward_observations.parquet` | 438K 条逐日观测（关注者/愿望单 × 330 游戏） | 轨迹图、巅峰周、velocity windows、检测提前量 |
| `data/backdrop_games.parquet` | 6.2 万款 Steam 游戏元数据 | 游戏名、发行商、**累计销量 (`copiesSold`)** |
| `config/success_games.json` | 30 款爆款列表（steamId + name） | 分组标记 |

### 关键处理函数

| 函数 | 输出 | 给哪个图表/组件 |
|------|------|---------------|
| `extract_separation_power(af)` | Cohen's d + 中位数 + 样本量 per feature | `chart_separation` |
| `extract_success_games_table(af, bd, ...)` | 30 款游戏表格数据（累计销量） | HTML 表格 |
| `extract_trajectories(fwd, ids, ...)` | 指定游戏的 [days_to_launch, followers] 序列 | `chart_trajectories` |
| `extract_control_envelope(fwd)` | 对照组中位数轨迹 | `chart_trajectories` 灰线 |
| `extract_peak_week_data(fwd, ...)` | 每游戏的 peak_week / avg_daily / peak_dtl | `chart_peak_distribution` + 策略表 |
| `extract_peak_cohens_d(peak_data)` | Peak Week d vs Avg Daily d **（v2 新增）** | Section 2 文本 |
| `extract_threshold_sweep(peak_data)` | 门槛 → precision/recall 曲线点 | `chart_threshold` |
| `extract_strategy_tiers(peak_data, ...)` | 4 个策略的命中/遗漏/误判/watchlist 游戏列表 | 交互策略表 |
| `extract_detection_leadtime(fwd, ...)` | 4 策略 × ≥90天/90天内/miss/无数据 **（v2 新增）** | `chart_detection_lead` |
| `extract_tag_analysis(af)` | 标签频率差异 | `chart_tags` |
| `extract_velocity_windows(fwd)` | 4 个时间窗口的中位增速 | `chart_windows` |
| `extract_scatter_data(af, ...)` | followers_at_launch vs first_month_revenue | `chart_scatter` |
| `extract_trajectory_with_peak(fwd, ids, ...)` | 指定游戏的 7d 滚动均增序列 + 巅峰标注 | `chart_peak_annotated` |

### 关键数据处理逻辑

- **Onset-artifact 处理**：`onset_artifact` 是**游戏级**标志（如果 Gamalytic 首次追踪时游戏已有大量 followers，则该游戏所有行均标为 True）。实际修复仅 null 掉第一行的 `daily_rate`。**不应用 onset_artifact 做整行过滤**——否则会丢失 23/30 款成功游戏的全部数据。7d rolling mean 自然处理了第一行的 NaN。
- **Per-feature eligibility**：velocity 计算在数据覆盖不足时返回 `None`（不是 0），不同窗口有不同的样本量
- **中文名映射**：`cn()` 函数通过 `CN` 字典查询，无中文名时保持英文原名
- **`fol_30d` 引用**：highlight 文本明确用 `next((f for f in sep_power if f["col"] == "follower_velocity_30d"), ...)` 而非 `sep_power[0]`，避免 wishlist 特征排在前面时引用错误

---

## 五、视觉设计规范

### 色板（v2 Steam 深蓝主色调）

| 用途 | 色值 | 变量名 |
|------|------|--------|
| **主色（Steam深蓝）** | `#1b2838` | `--primary` |
| **主色浅** | `#2a475e` | `--primary-light` |
| **主色最浅** | `#c7d5e0` | `--primary-lighter` |
| 爆款/成功标记 | `#10b981` | `--green` |
| 爆款浅底 | `#d1fae5` | `--green-light` |
| 对照组 | `#94a3b8` | `--gray` |
| 强调/蓝 | `#3b82f6` | `--blue` |
| 警告/琥珀 | `#f59e0b` | `--amber` |
| 危险/红 | `#ef4444` | `--red` |
| 页面背景 | `#f0f4f8` | `--bg` |
| 卡片背景 | `#ffffff` | `--card` |
| 正文 | `#1e293b` | `--text` |
| 标题 | `#0f172a` | `--heading` |
| 边框 | `#e2e8f0` | `--border` |
| Highlight块底色 | `#e8edf2` | (inline) |

### 组件样式

- **Hero**：**Steam深蓝渐变**（#1b2838 → #2a475e → #171a21），白色大标题
- **Sticky Nav**：白底，hover/active 状态用 `--primary` 深蓝色
- **Section 编号**：**深蓝色**圆形 badge（32px），`background: var(--primary)`
- **Highlight 块**：左边框 `--primary` 深蓝 4px + 浅蓝灰底 #e8edf2
- **Highlight-blue 块**：左边框 `--blue` 4px + 浅蓝底
- **Callout 块**：黄色背景 + 黄色边框
- **图表容器**：白色卡片 + 1px border + box-shadow
- **策略 Tab**：pill 按钮，active = **深蓝描边 + `--primary-lighter` 底**
- **2x2 Grid**：4 格 CSS Grid，每格不同颜色背景

### 响应式断点

```
> 768px:  4 列策略表、2x2 grid、标准图表高度 420px
≤ 768px: 2 列策略表、单列 2x2、图表高度缩至 320px、hero 缩小
≤ 480px: 单列策略表、更小字体
```

---

## 六、文本风格规范

- **语气**：中文日常口语、自然、娓娓道来，不过于正式和学究气
- **串联**：用启发式提问衔接章节
- **术语处理**：
  - Cohen's d → 先用通俗描述（"两组数据的距离"），再给出阈值
  - Precision/Recall → 翻译为"命中率"/"覆盖率"
  - Peak Week Velocity → 保留英文但加中文注释"巅峰周增速"
  - False Positive → "误判"
- **游戏名**：有中文名用中文，无中文名保留英文（通过 `CN` 字典和 `cn()` 函数）
- **数据呈现**：关键数字加粗，比值用"X倍"格式，百分比用 X% 格式

---

## 七、ECharts 图表配置备忘

### 轴标签防重叠方案

| 问题 | 解决方案 |
|------|---------|
| Y 轴 name 与 legend 重叠 | 使用 `nameLocation:'end'` + `nameTextStyle:{padding:[0,0,0,N]}` |
| X 轴 name 被裁剪 | 加大 `nameGap` (30→35) + grid bottom 增加到 55 |
| grid 不适配轴标签 | 使用 `containLabel:true`（如散点图） |
| 对数坐标轴范围截断 | 手动设 `min`/`max`（如散点图 xAxis min:100, max:1500000） |

### 图表 JS 变量名 → 数据来源映射

```javascript
const sepData        = d["sep_power"]         // Chart 1: Separation Power
const trajectoryData = d["trajectories"]      // Chart 2: Trajectories
const controlEnvelope= d["control_envelope"]  // Chart 2: Control gray line
const peakDistribution= d["peak_distribution"]// Chart 3: Peak Distribution
const peakAnnotated  = d["peak_annotated"]    // Chart 4: Peak Annotated
const thresholdData  = d["threshold_sweep"]   // Chart 5: Threshold Sweep
const tagData        = d["tags"]              // Chart 6: Tags
const windowData     = d["velocity_windows"]  // Chart 7: Velocity Windows
const scatterData    = d["scatter"]           // Chart 8: Scatter
const detectionData  = d["detection_leadtime"]// Chart 9: Detection Lead Time (v2 新增)
```

### Chart colors 对象

```javascript
const colors = {
  green:'#10b981',   // 爆款/成功
  gray:'#94a3b8',    // 对照组
  blue:'#3b82f6',    // 覆盖率线
  amber:'#f59e0b',   // 警告/wishlist
  red:'#ef4444',     // 危险
  primary:'#1b2838'  // Steam深蓝 (v2 新增)
};
```

---

## 八、已知问题与待修复

### 数据层

1. **Gray Zone Warfare steamId 已修复**（config层），但 pipeline 未重跑。`forward_observations.parquet` 仍含旧 ID `2009100`（实为 Immortals of Aveum）。需重跑 step 04→06→10。
2. **Detection leadtime 函数跳过旧 GZW ID**：暂未硬编码跳过，因为新 ID `2479810` 在 forward_observations 中还不存在（需重跑 pipeline）。

### 展示层

1. **Separation Power 分组**：当前实现通过在 features list 中手动排列 follower→wishlist→other 顺序，然后 sort by d descending。如果 d 值分布导致组内排序异常，可能需要更显式的分组逻辑。
2. **巅峰标注图**：pin 标注在部分游戏上重叠时可能遮挡。
3. **标签图仅显示正向差异**：只展示了 |diff| > 5 的标签，未区分正负方向的视觉分隔。

### 后续可改进

- 散点图加入 trend line（log-linear fit）
- 策略表加入搜索/过滤功能
- 标签图用中文翻译（目前是英文标签名）
- 巅峰标注图用事件名（"预告片发布""试玩节"）替代纯日期
- 检测提前量图表加 tooltip 显示具体游戏名

---

## 九、文件清单

```
src/10_generate_report.py     ← 报告生成主脚本 (1704行)
output/report.html            ← 交互式 HTML 报告 (~200KB)
output/report_text.md         ← 文本稿（用于润色）
report2.md                    ← 本文件（v2 框架与实现文档）
report1.md                    ← v1 框架文档（含用户反馈 # 注释，已执行完毕）
Report.md                     ← 用户原始 storyline 大纲
```

### 上游依赖（需先运行，按顺序）

```bash
python3 src/04_build_analysis_set.py   # → data/analysis_features.parquet
python3 src/05_analyze_signals.py      # → output/01-06_*.png
python3 src/06_forward_analysis.py     # → data/forward_observations.parquet
python3 src/07_threshold_screening.py  # → output/07-08_*.png（参考）
python3 src/08_strategy_table.py       # → output/09_strategy_table.png（参考）
python3 src/10_generate_report.py      # → output/report.html + report_text.md
```

### 重跑顺序（修复 GZW 后）

```bash
# 1. config 已修复（success_games.json + control_group.json）
# 2. 重跑 pipeline
python3 src/04_build_analysis_set.py
python3 src/06_forward_analysis.py
python3 src/10_generate_report.py
```

---

## 十、v2 → v3 变更日志（2026-03-19）

### A. 游戏名称统一（CN 字典 + 全文硬编码文本）

| 原名 | 改为 | 原因 |
|------|------|------|
| 缉私风云 | Schedule I | 读者更常用英文名 |
| 暗与暗 | Dark and Darker | 同上 |
| 雾锁王国 | Enshrouded | 同上 |
| 逃离鸭堡 | 逃离鸭科夫 | 更接近社区通用译名 |
| 光暗交错：远征33 | 光与影：远征队33 | 更准确的翻译 |
| 庄园领主 | Manor Lords | 读者更常用英文名 |
| 匹诺曹的谎言 | Lies of P | 同上 |
| 灰区战事 | Gray Zone Warfare | 同上 |

变更位置：`CN` 字典（~line 29-96）+ 所有 HTML/MD 模板中的硬编码引用。

### B. Cohen's d 区分力图表（Chart 1: `chart_separation`）

| 变更 | 旧 | 新 |
|------|-----|-----|
| 排序逻辑 | 全局按 d 值降序 | 分组排序：关注者组(绿) → 愿望单组(橙) → 其他(灰)，组内按 d 降序 |
| markLine 标签位置 | `position:'start'`（底部，被裁切） | `position:'end'`（顶部） |
| markLine 标签字号 | `fontSize:9` | `fontSize:11`（与 yAxis label 一致） |
| grid.top | 10 | 30（避免顶部标签被裁切） |
| grid.bottom | 20 | 45 |
| xAxis nameGap | 25 | 35（xAxis tick 与 markLine 标签分离） |

排序函数 `_group_order`：`is_follower=True → (0, -d)`, `False → (1, -d)`, `None → (2, -d)`。

### C. 轨迹图（Chart 2: `chart_trajectories`）

| 变更 | 旧 | 新 |
|------|-----|-----|
| Legend 颜色匹配 | 仅 `lineStyle.color`（legend 可能不同步） | 每个 series 额外加 `itemStyle:{color:...}` |
| X 轴范围 | 无限制（数据到 T-600） | `max:300`（截断到 T-300） |
| grid.bottom | 55 | 75（legend 与 xAxis name 分离） |

### D. 新增图表：平均周增速 vs 巅峰周增速散点图（Chart 2b: `chart_avg_vs_peak`）

- **位置**：Section 1 "爆款游戏的增长轨迹长什么样？" 之后、Section 2 之前
- **类型**：ECharts 散点图 + 2-sigma 置信椭圆
- **X 轴**：平均周增速（人/天），对数坐标（数据来自 `peak_data[].avg_daily`，即 7d 滚动均增的中位数）
- **Y 轴**：巅峰周增速（人/天），对数坐标（`peak_data[].peak_week`）
- **系列**：
  - 对照组灰色椭圆（虚线边框 + 10% 透明填充）+ 灰色散点
  - 爆款绿色椭圆 + 绿色散点（带游戏名标签）
- **椭圆算法**：对 log10(x), log10(y) 计算协方差矩阵 → 特征值分解 → 2-sigma 参数化椭圆 → 转换回线性坐标
- **数据复用**：使用已有的 `peakDistribution` JS 变量（即 `peak_data`），无需新增数据提取函数
- grid.top: 40（避免 Y 轴标题被裁切）

### E. 巅峰周增速分布图（Chart 3: `chart_peak_distribution`）

| 变更 | 旧 | 新 |
|------|-----|-----|
| 图表类型 | 离散柱状图（histogram, binSize=200） | **KDE 面积图**（连续密度分布曲线） |
| grid.bottom | 50 | 70（legend 与 xAxis name 分离） |
| legend.bottom | 0 | 0（保持底部） |
| xAxis nameGap | 30 | 28 |

KDE 实现：JS 端高斯核密度估计，bandwidth=180，200个采样点，上限 `min(max*1.3, 6000)`。

### F. 巅峰时刻轨迹图（Chart 4: `chart_peak_annotated`）

| 变更 | 旧 | 新 |
|------|-----|-----|
| X 轴范围 | 无限制（数据到 T-800） | `max:400`（截断到 T-400） |
| grid.bottom | 55 | 70（legend 与 xAxis 说明分离） |
| legend.bottom | 0 | 5 |

### G. 命中率/覆盖率曲线（Chart 5: `chart_threshold`）

| 变更 | 旧 | 新 |
|------|-----|-----|
| Y 轴标题 | `name:'比率'`（被裁切） | `nameLocation:'end', nameGap:15` |
| grid.top | 20 | 40（Y 轴标题不再被裁切） |
| grid.bottom | 50 | 70（legend 与 xAxis 分离） |
| xAxis nameGap | 30 | 28 |
| markLine 位置 | 在 `series` 同级（**错误，不生效**） | 移入第一个 series 内部 |
| markLine 内容 | 仅策略名+门槛值 | **新增命中率/覆盖率标注**（从 `thresholdData` 查找对应值） |
| series | 无 itemStyle | 每个 series 加 `itemStyle:{color:...}` |

### H. 检测提前期（原 Chart 9: `chart_detection_lead` → 静态表格）

| 变更 | 旧 | 新 |
|------|-----|-----|
| 展示形式 | ECharts 堆叠柱状图 | **HTML 静态表格**（`data-table` 样式） |
| 数据列 | 4 列：≥90天 / <90天 / 未达标 / 无数据 | **3 列**：≥90天 / <90天 / 未发现（去掉"无数据"） |
| JS 代码 | Chart 9 初始化 + `detectionData` const | **已删除**（不再需要 ECharts） |
| 模板 bug | `html += """` 导致 `{d["detection_leadtime"]...}` 未插值 | 改为 `f"""` + loop 生成表格行 |

### I. 策略交互表顶部统计卡片

| 变更 | 旧 | 新 |
|------|-----|-----|
| 卡片顺序 | 关注列表 → 命中率 → 覆盖率 → 误判 | **命中率 → 覆盖率 → 误判 → 关注列表**（与下方4列对应） |
| 覆盖率颜色 | `var(--blue)` | `var(--red)`（与下方红色表头对应） |
| 误判颜色 | `var(--red)` | `var(--amber)`（与下方黄色表头对应） |

### J. Section 结构调整

**"这个信号从什么时候开始存在？"**（含 `chart_windows` 及 highlight 文本）：
- **旧位置**：Section 3（其他发现）
- **新位置**：Section 2（车前灯），紧接"这些巅峰时刻发生在什么时候？"之后
- 同步更新了 MD 文本（`generate_md()`）

### K. 散点图 markLine 修复（Chart 8: `chart_scatter`）

`markLine` 从 `setOption` 顶层移入对照组 series 内部（原位置在 `series` 同级，ECharts 不识别）。

### L. 当前图表索引（v3）

| 序号 | 图表 ID | 类型 | 所属 Section |
|------|---------|------|-------------|
| 1 | `chart_separation` | 横向柱状图 | S1 后视镜 |
| 2 | `chart_trajectories` | 折线图 | S1 后视镜 |
| 2b | `chart_avg_vs_peak` | 散点图+椭圆 **(新)** | S1 后视镜 |
| 3 | `chart_peak_distribution` | **KDE 面积图**（原柱状图） | S2 车前灯 |
| 4 | `chart_peak_annotated` | 折线图+标注 | S2 车前灯 |
| 7 | `chart_windows` | 分组柱状图 | **S2 车前灯**（原 S3） |
| 5 | `chart_threshold` | 双折线图+策略线 | S2 车前灯 |
| — | 策略交互表 | HTML tab | S2 车前灯 |
| — | 检测提前期 | **HTML 静态表格**（原 ECharts） | S2 车前灯 |
| 6 | `chart_tags` | 横向柱状图 | S3 其他发现 |
| 8 | `chart_scatter` | 散点图 | S3 其他发现 |

### M. JS 变量映射更新（v3）

```javascript
const sepData         = d["sep_power"]          // Chart 1
const trajectoryData  = d["trajectories"]       // Chart 2
const controlEnvelope = d["control_envelope"]   // Chart 2 gray line
const peakDistribution= d["peak_distribution"]  // Chart 2b + Chart 3
const peakAnnotated   = d["peak_annotated"]     // Chart 4
const thresholdData   = d["threshold_sweep"]    // Chart 5
const tagData         = d["tags"]               // Chart 6
const windowData      = d["velocity_windows"]   // Chart 7
const scatterData     = d["scatter"]            // Chart 8
// detectionData 已删除（改为静态表格，数据在 Python 端渲染）
```

---

## 十一、v3 → v4 变更日志（2026-03-20）：双指标筛选 + Backdrop 数据恢复

### A. 核心问题：单指标筛选的局限

v3 使用**巅峰周增速（Peak Week Velocity）**作为唯一筛选指标。当评估样本从 300 个标签匹配的对照组扩展到全量 backdrop 样本后，命中率急剧下降：

| 问题 | 原因 |
|------|------|
| 命中率从 ~30% 降至 ~5-7% | 对照组是精选的"还不错"的游戏，backdrop 包含所有类型 |
| 大量"一次性爆发"假阳性 | 游戏有一次大事件（预告片/展会）带来巨峰，但日常增速接近零 |
| 原"甜点"阈值 845 基于 PR 交叉点 | 该交叉点在 300 对照上算出，在 3800 游戏上已不适用 |

### B. 关键发现：日常增速的独立区分力

分析发现 `avg_daily`（周滚动增速的中位数）在全量 backdrop 上的 Cohen's d **高于** `peak_week`：

| 指标 | Cohen's d (log10) | 爆款中位数 | Backdrop 中位数 | 倍率 |
|------|-------------------|-----------|----------------|------|
| Peak Week Velocity | 1.85 | 1,451/天 | 78/天 | 18.7x |
| Avg Daily Velocity (中位数) | **1.89** | 111/天 | 3.8/天 | **29.2x** |

这颠覆了 v3 的结论（"巅峰周 > 日均"）。原结论是在 300 个对照组上算出的，对照组本身已有一定持续关注度，所以 peak 更能区分。但在真实全量中，大多数游戏日常增速接近零，avg 的区分力更强。

**两个指标捕捉不同维度**：Peak = 天花板潜力（最火的一周），Avg = 持续热度（典型一周）。爆款游戏两者都高（中位 spike ratio = 14.8x），而假阳性往往只有高 peak 但极低 avg（spike ratio > 100x）。

### C. Backdrop 数据恢复

**发现 bug**：`src/03b_collect_backdrop_sample.py` 第 85 行获取 backdrop 历史时遗漏了 `include_pre_release_history=True` 参数（`03_collect_histories.py` 有此参数）。导致 4,872 个 backdrop 历史文件中仅 748 个（15.4%）有上线前数据。

**修复**：创建 `src/03c_refetch_backdrop_prerelease.py`，对所有 followers ≥ 500 的 backdrop 游戏重新获取历史数据（带 `include_pre_release_history=True`）。

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 有上线前数据的文件 | 748 (15.4%) | **3,819 (78.4%)** |
| 重新获取成功 | — | 3,095 个文件更新 |
| 失败 | — | 0 |
| 可用于速度分析 | 742 | **3,810** |

### D. avg_daily 实际是中位数，而非均值

尽管变量名为 `avg_daily`，实际计算的是 7 日滚动均增值的**中位数**（`rolling.median()`）。对于筛选目的这是正确选择：

- **中位数**：抗异常值，代表"典型一周"的增速。spike-only 游戏的中位数 ≈ 0（大部分周没增长）
- **均值**：会被极端值拉高，让 spike-only 游戏看起来也有持续热度

展示标签应改为"日常增速（周中位数）"或 "Median Weekly Velocity"。

### E. avg ≈ 0 的游戏不是数据错误

散点图上 avg ≈ 0 的 280+ 个游戏有充分数据（中位 307 条记录，跨 322 天），它们的确在大部分时间没有增长，只靠一次事件冲到高峰。这恰好是 avg 过滤器要捕捉的"一次性爆发"模式。

### F. 双指标筛选策略设计

**方法论：阶梯百分位法（Stepped Percentiles）**

用 backdrop 数据集的百分位来设定阈值。Peak 使用稍高的百分位（更严格选择天花板），Avg 使用稍低的百分位（更宽容地接受持续热度）。这确保两个维度都独立地将游戏置于 backdrop 的顶部。

**最终四档策略（3,810 游戏数据集，27 success + 3,783 backdrop）**：

| 策略 | Peak ≥ | Avg ≥ | Backdrop P | 关注数 | 命中 | 命中率 | 覆盖率 | Lift |
|------|--------|-------|------------|--------|------|--------|--------|------|
| 撒网策略 | 400 | 20 | P86/P82 | 335 | 21 | 6.3% | 77.8% | 9x |
| 甜点策略 | 600 | 30 | P90/P88 | 203 | 17 | 8.4% | 63.0% | 12x |
| 精选策略 | 1,100 | 50 | P95/P93 | 104 | 13 | 12.5% | 48.1% | 18x |
| 高确信策略 | 1,900 | 70 | P97/P95 | 58 | 11 | 19.0% | 40.7% | 27x |

**vs 仅用 Peak 的旧方案**（同 peak 阈值）：

| 策略 | 旧命中率 (peak only) | 新命中率 (dual) | 提升 |
|------|---------------------|-----------------|------|
| 撒网 | 4.0% | 6.3% | **1.6x** |
| 甜点 | 5.0% | 8.4% | **1.7x** |
| 精选 | 8.0% | 12.5% | **1.6x** |
| 高确信 | 11.2% | 19.0% | **1.7x** |

所有档位的命中率提升约 **1.6-1.7 倍**，同时覆盖率仅下降 5-10 个百分点。

### G. "误判"仍然是好游戏

在甜点策略下的 186 个"误判"（backdrop 非 success）中：
- 中位营收 $18.0M
- 37% 营收 > $30M
- 24% 营收 > $50M

包括 Battlefield 6、Ghost of Tsushima、Slay the Spire 2、Hades II、S.T.A.L.K.E.R. 2 等。这些是"预期内的成功"或"知名续作"，没有进入我们的"惊喜爆款"名单，但商业上表现优秀。

### H. PR 交叉点的变迁

| 数据集 | Peak-only 交叉点 | Dual-filter 交叉点 |
|--------|-----------------|-------------------|
| 300 对照组 (v3) | ~845/天 | — |
| 3,810 全量 (v4) | ~5,900/天 | P98 (peak≥2,271, avg≥168) |

原"甜点"阈值 845 在全量数据上早已偏离 PR 交叉点。v4 的阶梯百分位法不依赖单一交叉点，而是在整个 PR 曲线上选择 4 个有意义的点。

### I. 新增脚本

| 脚本 | 用途 |
|------|------|
| `src/03c_refetch_backdrop_prerelease.py` | 重新获取 backdrop 历史（带 `include_pre_release_history=True`） |
| `src/11_dual_metric_analysis.py` | 独立验证脚本：Cohen's d、spike ratio、双指标网格、PR 曲线、策略设计 |

### J. 验证图表（matplotlib，存于 output/）

| 文件 | 内容 |
|------|------|
| `output/dual_scatter.png` | Peak vs Avg 散点图（3,810 游戏），阈值线标注 |
| `output/dual_pr_curves.png` | PR 曲线：单指标 vs 不同 avg 门槛的双指标 |
| `output/spike_ratio_hist.png` | Spike ratio 分布：success vs backdrop |

### K. 待实施（HTML 报告集成）

1. **Section 2 叙事修改**：从"Peak > Avg → 用 Peak"改为"Peak 和 Avg 各有独立区分力 → 双指标组合更强"
2. **散点图更新**：`chart_avg_vs_peak` 改用 3,810 游戏全量数据 + 阈值线
3. **策略表更新**：每档显示双阈值（Peak ≥ X AND Avg ≥ Y）
4. **PR 曲线更新**：`chart_threshold` 改为显示 dual-filter PR 曲线
5. **Lift 指标引入**：展示筛选相对随机选择的提升倍数（9x-27x）
6. **文案中的数字全部更新**：命中率、覆盖率、关注列表大小等


### L. Frank 的最新调整意见：
0. 如无必要，整体请基于上一版html去“迭代“而非“重写“，我没有明确提出一定要改的部分和章节尽量保持原样，除非你认为通过调整能更好呈现我们的分析发现，以及提供更顺的报告叙事
1. 在车前灯：提前识别爆款(Section2)中，“前向指标：Peak Week Velocity 胜出“ 章节整体调整，内容改为讨论：
  - 上面的分析用的“上线前30天增速“这些指标是后视镜指标，所以需要调整（可以用上一版原文不变）
  - 把上一个section的 平均周增速 vs 巅峰周增速的散点图移动到这里。然后提出假设：巅峰周增速与平均周增速结合起来，更能区分好的&一般的产品。然后分别计算这两个指标的cohen's d（是否有combined cohen‘s d? 有的话就计算，没有就不用了）
  - 原来这个章节中，巅峰周增速的单指标分布，在有了上面的联合分布图之后存在的意义不大了，所以可以整个取消掉
  - 紧接着，下面两个子章节迭代：
  - “这些巅峰时刻发生在什么时候？“章节调整为：“实例看看巅峰周增速的区分度“：整体还是用现在的图表内容和形式，Arc Raiders，漫威争锋，Schedule I，Manor Lords保留，幻兽帕鲁去掉。然后加一条对照组的中位数来作为对照
  - “这个信号从什么时候开始存在？“章节调整为：“实例看看周均增速的区分度“：整体还是用现在的图表内容和形式，呈现各个阶段窗口中，success组和对照组的周增速median的对比差距
  - 在“门槛定多少？Hit Rate 和 Catch Rate 的取舍“章节之前，加一个子章节：前向预判与后视镜场景的不同，讨论：
    - 在章节1中我们选择的300个对照组是筛选过，上线后销量>5万的产品，但是前向预判的时候无法做这个筛选
    - 真实场景里，所有需要对产品作出判断的人，确实也会面临更眼花缭乱的产品选择，在AI时代尤甚。另一方面，如果要对每年1万多款以上的steam产品进行全面监测和审视似乎也不现实；部分产品会被注意到，部分产品会从视线里漏掉
    - 所以在我们在模拟“时光倒流，一个人建立了一个监测的portfolio“的时候，假设他有精力在23-26年近6w款新品（数据请你核实）中的4000多款sample（请你说明sample的选取逻辑）
  - “门槛定多少？Hit Rate 和 Catch Rate 的取舍“章节中，现在的叙述逻辑是没问题的：
    - 引入和介绍“命中率“和“覆盖率“的概念，且说明这两个比例天然此消彼长：追求命中容易漏球，追求不漏球命中率就会下降。但“覆盖率“这个term我建议改为“不漏球率“，全文统改。
    - 然后下面的图表，其实主要作用是示意这两个比例之间此消彼长的关系。具体的数据其实可以用最终几千个历史样本的实际推演值来作图。对4种可能策略的位置选取可以根据最新的tier结果来计算
  - 四种策略、具体名单、标准越高，发现越晚 子章节提纲结构不变，但根据最新的数据重新调整图表中的内容（“推荐：甜点策略“部分删掉，目前不太能结合大家的处境去做推荐）
  - “深口袋假说被否定“这一段内容删掉
2. 其他未被提及的部分保留原样


