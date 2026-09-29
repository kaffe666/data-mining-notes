# Data Mining Notes / 数据挖掘笔记

Study notes for the *Data Mining* course (Politecnico di Milano, 2026/27).
本仓库是 Politecnico di Milano *Data Mining* 课程（2026/27）的学习笔记，中英双语，持续更新。

**Contents / 目录**

1. [Association Rules: support, confidence, lift / 关联规则的三个指标](#1-association-rules-support-confidence-lift--关联规则的三个指标)
2. [Apriori & Frequent Itemsets / Apriori 与频繁项集](#2-apriori--frequent-itemsets--apriori-与频繁项集)
3. [k-means](#3-k-means)
4. [Hierarchical Clustering / 层次聚类](#4-hierarchical-clustering--层次聚类)
5. [DBSCAN](#5-dbscan)
6. [Common Mistakes / 易错点](#6-common-mistakes--易错点)

---

## 1. Association Rules: support, confidence, lift / 关联规则的三个指标

An association rule A → B means "customers who buy A also buy B".
关联规则 A → B 表示"买了 A 的人也会买 B"。

| Metric / 指标 | Formula / 公式 | Meaning / 含义 |
| --- | --- | --- |
| support | #records containing the itemset ÷ #records | How common the itemset is / 这组东西有多常见 |
| confidence(A → B) | support(A ∪ B) ÷ support(A) | Share of A-buyers who also buy B / 买 A 的人里也买 B 的比例 |
| lift(A → B) | confidence(A → B) ÷ support(B) | > 1 positive, = 1 independent, < 1 negative / 正相关、独立、负相关 |

**Example (5 transactions) / 例子（5 笔记录）:** diapers appear 4 times, beer 3 times, both together 3 times.
尿布出现 4 次，啤酒 3 次，两者同时出现 3 次。

- confidence(diapers → beer) = 3/4 = 0.75
- confidence(beer → diapers) = 3/3 = 1 — same itemset, different direction, different confidence / 同一组东西，两个方向的 confidence 不同
- lift(diapers → beer) = 0.75 ÷ 0.6 = 1.25 → positive correlation / 正相关

**Exam trap (2025-06) / 真题陷阱:** the "support" of a rule means support(A ∪ B). If A → B and B → A are given different supports, they come from different datasets; without support(B) the lift cannot be computed, so the answer is "cannot be determined".
题目里规则的 support 指 support(A ∪ B)。若 A → B 和 B → A 的 support 不同，说明来自不同数据集；缺 support(B) 就算不出 lift，答案是"无法判断"。

---

## 2. Apriori & Frequent Itemsets / Apriori 与频繁项集

**Frequent / 频繁** = support ≥ minsup. minsup is a threshold given by the problem (count or ratio); there is no fixed value.
minsup 是题目给的门槛（次数或比例），没有固定值。

**Key principle (anti-monotonicity) / 核心原理:** if an itemset is infrequent, every superset is infrequent too.
一个组合不频繁，它的任何超集也不频繁。

**Apriori steps / 步骤:**

1. Count single items, keep the frequent ones / 数单品，留下频繁的
2. Build pairs from frequent items, scan the data, keep frequent pairs / 拼两两组合，回去数记录，留下频繁的
3. Build triples: **prune any candidate that contains an infrequent pair without counting it**; count only the rest / 拼三件组合：含有不频繁的一对就直接删掉，不用数；剩下的再回去数
4. Repeat until no frequent itemset can be built / 重复，直到拼不出频繁组合

Pruning only excludes: all pairs being frequent does **not** guarantee the triple is frequent.
剪枝只能排除：每一对都频繁，三个一起不一定频繁。

**Rule generation / 生成规则:** split each frequent itemset into X → Y, compute the confidence, keep rules with confidence ≥ minconf. Support is already above minsup, so only confidence needs checking.
把每个频繁项集拆成 X → Y，只保留 confidence ≥ minconf 的规则；support 已达标，只需检查 confidence。

**Compact representations / 压缩结果的两个概念:**

| Concept / 概念 | Definition / 定义 |
| --- | --- |
| Maximal | Frequent, and no superset is frequent / 频繁，且任何超集都不频繁 |
| Closed | Frequent, and no superset has the same support / 频繁，且没有超集和它 support 相同 |

Every maximal itemset is closed, not vice versa. Example: diapers (4) is closed but not maximal; beer (3) is neither, because {diapers, beer} also has support 3.
maximal 一定是 closed，反之不一定。例：尿布（4 次）是 closed 但不是 maximal；啤酒（3 次）两者都不是，因为 {尿布, 啤酒} 也是 3 次。

**Exam question / 真题:** Apriori runs slowly and outputs very long rules → minsup is set too low.
Apriori 跑得慢、出现很长的规则 → minsup 设得太低。

---

## 3. k-means

**k must be chosen in advance; there is no radius.** Each point joins the nearest centroid.
要事先定 k，没有半径。每个点离哪个中心近就归哪个。

1. **Initialisation / 初始化:** choose k starting centroids (given in the exam) / 选 k 个起始中心（考试会给）
2. **Assignment / 分配:** assign each point to its nearest centroid / 每个点归到最近的中心
3. **Update / 更新:** new centroid = mean of the points in the cluster (x and y separately) / 新中心 = 这一类所有点的平均（x、y 分开算）
4. **Repeat 2–3 until assignments stop changing.** If any point changed cluster, recompute centroids and do another round / 重复直到分类不再变化；只要有点换了类，就要再算中心、再做一轮

**Distance / 距离:** √((x₁ − x₂)² + (y₁ − y₂)²). To compare, skip the square root and compare squared distances.
只比大小时不用开根号，直接比平方和。

**Ties / 平局:** state your rule, e.g. "ties go to c1" / 写明规则，如"平局归入 c1"。

**WSS (Within Sum of Squares):** sum of squared distances from each point to its own centroid; smaller means tighter clusters.
每个点到自己簇中心的距离平方之和；越小簇越紧凑。

**The result depends on the starting centroids;** in practice run several times with different starts and keep the lowest WSS.
结果取决于初始中心，实际中换不同起点多跑几次，选 WSS 最小的。

**Exam format / 考试格式:** label each round (Iteration 1, …) and give a table:

| Point / 点 | d²(c1) | d²(c2) | Cluster / 归入 |
| --- | --- | --- | --- |
| A (1,1) | 0.25 | 16.5 | c1 |

Then write the new centroids (two decimals) and finish with "Assignments unchanged → converged, stop". Use a pen; work on the scratch pages first, then copy into the answer box.
表下写新中心（两位小数），最后写"Assignments unchanged → converged, stop"。用钢笔，先在草稿页算，再抄进答题框。

---

## 4. Hierarchical Clustering / 层次聚类

**No k needed.** Start with every point as its own cluster; at each step compute the distance between **every pair of clusters** and merge the closest pair, until one cluster remains.
不用定 k。一开始每个点自成一组；每一步算出所有组两两之间的距离，合并最小的一对，直到只剩一组。

| Linkage | Distance between two clusters / 两组之间的距离 |
| --- | --- |
| Single | **Closest** cross-cluster pair / 跨组最近的一对点 |
| Complete | **Farthest** cross-cluster pair / 跨组最远的一对点 |
| Average | Mean over all cross-cluster pairs / 跨组所有点对的平均 |

**Height = the distance used for that merge;** it is the vertical axis of the dendrogram. Heights need not be consecutive.
高度 = 合并时用的那个距离，就是树状图的纵轴；高度不必连续。

**Example / 例:** A=1, B=2, C=4, D=9, E=12

| Step / 步骤 | Merge / 合并 | Single height | Complete height |
| --- | --- | --- | --- |
| 1 | A, B | 1 | 1 |
| 2 | {A,B}, C | 2 | 3 |
| 3 | D, E | 3 | 3 |
| 4 | all / 全部 | 5 (C–D) | 11 (A–E) |

**Using the dendrogram / 树状图的用法:** cut horizontally at a height; the number of vertical lines cut = number of clusters.
在某个高度横切一刀，切到几根竖线就是几类。

**Distance-matrix questions / 距离矩阵题:** the entries **are** distances — read them directly, never subtract. Only coordinates need to be turned into distances first.
表里的数就是距离，直接查表，不要相减；给坐标才要先算距离。

---

## 5. DBSCAN

**Cluster = dense region.** No k needed; finds clusters of any shape and labels noise automatically.
簇 = 点很密的区域。不用定 k，能找任意形状，还能自动识别噪声。

**Parameters / 参数:** eps (radius) and MinPts (minimum number of points in the neighbourhood, **including the point itself**). In 1D the neighbourhood is the interval [x − eps, x + eps]; in 2D it is a circle.
eps（半径）和 MinPts（圈里至少几个点，包括自己）。一维的"圈"是区间 [x − eps, x + eps]，二维是圆。

| Type / 类型 | Condition / 条件 |
| --- | --- |
| Core / 核心点 | neighbourhood contains ≥ MinPts points / 圈里点数 ≥ MinPts |
| Border / 边界点 | not core, but inside a core point's neighbourhood / 不是核心点，但在某个核心点的圈里 |
| Noise / 噪声点 | neither / 两者都不是 |

Core points within each other's neighbourhood form one cluster; border points join a nearby core point.
互相在对方圈里的核心点连成一簇，边界点跟着附近的核心点。

**Example / 例:** 1, 2, 3, 6, 10, 11, 12, 20 with eps = 1, MinPts = 3

- Core / 核心点: 2, 11
- Border / 边界点: 1, 3, 10, 12
- Noise / 噪声: 6, 20
- Result / 结果: two clusters {1, 2, 3} and {10, 11, 12}

---

## 6. Common Mistakes / 易错点

- [ ] Re-check the conclusion after comparing numbers / 比较大小后再核对一遍结论
- [ ] Compute means carefully: (1+3)÷2 = 2 / 求平均别算错
- [ ] Report centroids with two decimals: 2.67, not 2.6 / 中心保留两位小数
- [ ] In k-means, if any point changed cluster, recompute centroids and run another round; state convergence explicitly / 有点换类就再算一轮，最后写明收敛
- [ ] In hierarchical clustering, compare **all** clusters at each step, including two single points / 每一步比较所有组，包括两个单独的点
- [ ] Between two clusters only cross-cluster pairs count / 两组之间只看跨组点对
- [ ] Read distance matrices directly; do not subtract / 距离矩阵直接查表
- [ ] k-means has no radius; eps belongs to DBSCAN / k-means 没有半径，半径是 DBSCAN 的
- [ ] Wrong answers lose points: leave blank what you don't know, and justify every answer ("adequately motivated") / 答错倒扣分，不会的宁可空着；每题要写理由
