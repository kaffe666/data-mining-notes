# Data Mining Notes / 数据挖掘笔记

Study notes for the *Data Mining* course (Politecnico di Milano, 2026/27).
本仓库是 Politecnico di Milano *Data Mining* 课程（2026/27）的学习笔记，中英双语，持续更新。

**Contents / 目录**

1. [Association Rules: support, confidence, lift / 关联规则的三个指标](#1-association-rules-support-confidence-lift--关联规则的三个指标)
2. [Apriori & Frequent Itemsets / Apriori 与频繁项集](#2-apriori--frequent-itemsets--apriori-与频繁项集)
3. [k-means](#3-k-means)
4. [Hierarchical Clustering / 层次聚类](#4-hierarchical-clustering--层次聚类)
5. [DBSCAN](#5-dbscan)
6. [Regression Basics / 线性回归基础](#6-regression-basics--线性回归基础)
7. [Model Evaluation: train / test / CV / 模型评估](#7-model-evaluation-train--test--cv--模型评估)
8. [Lasso, Ridge & Standardization / 正则化与标准化](#8-lasso-ridge--standardization--正则化与标准化)
9. [Pipeline Template (exam) / Pipeline 万能模板](#9-pipeline-template-exam--pipeline-万能模板)
10. [Decision Trees / 决策树](#10-decision-trees--决策树)
11. [Regression Tree Exam Problem (2025-06) / 回归树真题](#11-regression-tree-exam-problem-2025-06--回归树真题)
12. [Explainability / 可解释性](#12-explainability--可解释性)
13. [Clustering Evaluation / 聚类评估](#13-clustering-evaluation--聚类评估)
14. [Classification Metrics / 分类评估指标](#14-classification-metrics--分类评估指标)
15. [Common Mistakes / 易错点](#15-common-mistakes--易错点)

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

## 6. Regression Basics / 线性回归基础

**Least squares / 最小二乘:** fit the line y = a + b·x that minimises the sum of squared residuals.
找一条直线 y = a + b·x，让所有残差的平方和最小。

| Term / 术语 | Meaning / 含义 |
| --- | --- |
| intercept a | value of y when x = 0 / x = 0 时的 y |
| slope b | how much y changes when x increases by 1 / x 每增加 1，y 变多少 |
| residual | y − ŷ (true − predicted) / 真实值 − 预测值 |
| MSE | mean of squared residuals; **smaller is better** / 残差平方的平均，越小越好 |
| R² | share of variance explained (0–1); for simple regression R² = r² / 解释了多少变化，越接近 1 越好 |

**Calculator (fx-991CN X) / 计算器:** MENU → 6 → 2 (y=a+bx) → enter data → OPTN → 回归计算 gives a, b, r.
⚠️ Some screens show y = ax + b: **the number multiplied by x is always the slope**.
⚠️ 有的屏幕写 y = ax + b：**乘以 x 的那个数永远是斜率**。

---

## 7. Model Evaluation: train / test / CV / 模型评估

**Analogy / 比喻:** training set = textbook, cross-validation = mock exams, test set = the final exam (used **once**, at the end).
训练集 = 课本，交叉验证 = 模拟题，测试集 = 期末考（只能在最后用一次）。

| Situation / 情况 | Training error | Test error |
| --- | --- | --- |
| Underfitting / 欠拟合 (model too simple) | high / 高 | high / 高 |
| Overfitting / 过拟合 (memorised the training data) | low / 低 | **much higher** / 高很多 |

**10-fold cross-validation / 10 折交叉验证:** split the training set into 10 parts; train on 9, validate on 1; repeat 10 times and average the error. Use it to **choose parameters**.
把训练集分成 10 份，用 9 份训练、1 份验证，轮 10 次取平均误差。用来**选参数**。

---

## 8. Lasso, Ridge & Standardization / 正则化与标准化

Both add a penalty on large coefficients to reduce overfitting; α controls the strength.
两者都惩罚过大的系数来防止过拟合，α 控制惩罚力度。

| | Lasso (L1) | Ridge (L2) |
| --- | --- | --- |
| Effect / 效果 | can set coefficients **exactly to 0** → feature selection / 能把系数压成 0，自动挑特征 | shrinks coefficients but **not to 0** / 只压小，不归零 |

| α | Result / 结果 |
| --- | --- |
| too small / 太小 | almost no penalty → **overfitting** / 几乎不惩罚 → 过拟合 |
| too large / 太大 | everything shrunk → **underfitting** / 全被压没 → 欠拟合 |
| best / 最佳 | chosen by **cross-validation on the training set**, never by the test set / 在训练集上用 CV 选，绝不用测试集选 |

**Standardization (StandardScaler) / 标准化:** z = (x − mean) ÷ std.
Without it, features with small numbers (e.g. number of rooms 1–5) need big coefficients and get punished unfairly by Lasso/Ridge.
不标准化的话，数值小的特征（如房间数 1–5）需要很大的系数，会被 Lasso/Ridge 不公平地惩罚。

- Example / 例: rooms mean 3, std 1 → 5 rooms → (5 − 3) ÷ 1 = **2** ("2 std above average" / 比平均高 2 个标准差)
- Mean and std are computed on the **training set only**; otherwise it is **data leakage** / 平均值和标准差只能用训练集算，否则就是数据泄露
- **Decision trees do not need scaling:** they split on one feature at a time ("area > 100?"), and scaling does not change the order of values / 决策树不需要标准化：每次只问一个特征大于还是小于某个数，标准化不改变大小顺序

---

## 9. Pipeline Template (exam) / Pipeline 万能模板

Pipeline design is the most frequent exam problem (10 of 30 problems in recent exams). The answer is **python-like pseudo-code**: syntax errors are not penalised; the **steps** are graded.
Pipeline 设计是最常考的题型（近年 30 题里 10 题）。答案写伪代码，语法错不扣分，看的是**步骤**对不对。

**Mnemonic / 口诀:** lock the test set → scale → choose the parameter by CV → score on test once
先锁 test → 标准化 → CV 选参数 → test 打分（只用一次）

```python
# hold out test set, used only once at the end
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# standardize features, fit on training data only
pipeline = Pipeline([StandardScaler(), MODEL()])
# choose parameter with 10-fold cross-validation
grid = GridSearchCV(pipeline, param_grid={'PARAM': [...]}, cv=10)
grid.fit(X_train, y_train)
# final assessment on test set
y_pred = grid.predict(X_test)
print(METRIC(y_test, y_pred))
```

Only three blanks change / 只换三个空:

| Task / 任务 | MODEL | PARAM | METRIC |
| --- | --- | --- | --- |
| Predict a number (regression) / 预测数字 | Lasso / Ridge | alpha | MSE, R² |
| Predict a class (classification) / 预测类别 | DecisionTreeClassifier | max_depth | accuracy |

- `fit` = train the model on training data / 用训练集训练
- `predict` = use the trained model to predict; **never fit on the test set** / 用训练好的模型预测，绝不在 test set 上 fit
- If test MSE ≫ CV MSE → overfitting / test 的 MSE 比 CV 大很多 → 过拟合
- Trees: drop StandardScaler and write `# no scaling needed for trees` / 树模型可以去掉标准化并写一句理由

---

## 10. Decision Trees / 决策树

**Idea / 思路:** keep asking yes/no questions that split the data into purer and purer groups.
一直问"是/否"问题，把数据分成越来越纯的小组。

**Impurity measures / 不纯度（越小越纯）:**

| Measure | Formula | Pure / 最纯 | 50:50 (two classes) |
| --- | --- | --- | --- |
| Gini | 1 − Σ p² | 0 | 0.5 |
| Entropy | − Σ p · log₂ p | 0 | 1 |

- Example / 例 8:2 → Gini = 1 − (0.8² + 0.2²) = **0.32**; Entropy = −(0.8 log₂0.8 + 0.2 log₂0.2) = **0.722**
- A class with p = 0 is skipped (0 · log₂0 = 0); the calculator gives Math ERROR for log₂0 / 比例为 0 的类直接不算
- Calculator / 计算器: use the `log□□` key for log₂

**Choosing a split / 选分裂:**

1. Impurity before the split (parent) / 分裂前的不纯度
2. Impurity of each child, then the **weighted average** by number of samples / 两边各算，再按人数加权平均
3. **Gain = before − after** (Gini gain or Information Gain); choose the split with the **largest gain** / 下降最多的问题最好

**Example (10 emails, 5 spam / 5 normal) / 例:**

| Question / 问题 | Children (spam:normal) | Gini after | Entropy after | Information Gain |
| --- | --- | --- | --- | --- |
| "free" / 免费 | 5:1 and 0:4 | 0.167 | 0.39 | **0.61** ✅ |
| "link" / 链接 | 3:2 and 2:3 | 0.48 | 0.97 | 0.03 |

- Gain = 0 → the question is useless / 问了等于没问
- Perfect split (5:0 and 0:5) → impurity 0, gain = 0.5 (Gini) / 完美分裂
- Trees need **no scaling**; if a tree is the best model, the data probably has **non-linear** relations / 树不需要标准化；树表现最好说明数据有非线性关系
- Recent exams (2025–26) do **not** ask for hand-computed Gini/entropy; `criterion = 'gini' / 'entropy'` appears as a hyperparameter / 近两年真题没考手算 Gini，考的是回归树

---

## 11. Regression Tree Exam Problem (2025-06) / 回归树真题

**Reading a node (scikit-learn plot) / 读节点:**

| Field | Meaning / 含义 |
| --- | --- |
| `MD <= -0.1` | condition: **True → left**, False → right / 满足往左，不满足往右 |
| `samples` | number of training samples in the node / 训练样本个数 |
| `value` | **mean** of those samples = the prediction / 平均值 = 预测值 |
| `squared_error` | **mean** squared error in the node (MSE) / 节点内误差²的平均 |

⚠️ Negative numbers: −0.06 > −0.1, so `−0.06 <= −0.1` is False → right / 小心负数

**Steps / 步骤:**

1. **Predict** each test sample by walking down the tree / 顺着树走到叶子
2. **Test RSS** = Σ(real − predicted)²
3. **TSS** = Σ(real − mean of real)² — the error of a "dumb" model that always predicts the mean / 每次都猜平均值的笨模型的误差
4. **R² = 1 − RSS ÷ TSS** — how much better than predicting the mean / 比猜平均值好多少
5. **Training RSS** = Σ over leaves of (samples × squared_error) / 每个叶子 samples × squared_error 再相加

| R² | Meaning / 含义 |
| --- | --- |
| 1 | perfect / 完美 |
| 0 | same as predicting the mean / 和猜平均一样 |
| < 0 | worse than predicting the mean (e.g. overfitting) / 比猜平均还差 |

**Calculator (fx-991CN X) / 计算器:** `MENU → 6 → 2` (two-variable; re-selecting the type clears old data / 重新选类型会清空旧数据)

| Goal / 要算 | x column | y column | Formula (insert variables with `OPTN`) |
| --- | --- | --- | --- |
| Test RSS | real | predicted | `Σx² − 2 × Σxy + Σy²` (求和计算) |
| TSS | real | predicted | `n × σx²` (双变量计算; σx, not sx) |
| R² in one line | real | predicted | `1 − (Σx² − 2Σxy + Σy²) ÷ (n × σx²)` |
| Training RSS | samples | squared_error | `Σxy` |

- Σ(x − y)² = Σx² − 2Σxy + Σy² (expand each row, then sum each column); it is **not** (Σx − Σy)²
- n × σx² = Σ(x − x̄)²: the calculator already knows the mean / 平均值已经藏在 σx 里
- σx, σx² divide by **N**; sx, sx² divide by **N − 1**. Both are reliable, pick the one the problem asks for (test with data 1, 5: σx² = 4, sx² = 8) / σ 组除以 N，s 组除以 N−1，都准，按题目选
- ❌ Do **not** use the regression r² from the calculator: it refits its own line, so it is not the R² of the tree (here r² = 0.35, true R² = −0.07) / 不能用计算器的 r²

**Answers / 答案:** predictions 22.80, 39.10, 39.10, 39.10, 53.90; test RSS = 1045.47; R² = −0.07; training RSS = 9223.75

---

## 12. Explainability / 可解释性

**Local contribution (one sample) / 局部贡献（一个样本）:** walk the sample down the tree; at every split, the **change of the node average** is credited to the feature asked at that split.
顺着树走，每一步平均值的变化，记在这一步问的那个特征上。

- Prediction = root average + sum of contributions / 预测值 = 起点 + 所有贡献（用来检查）
- A feature never asked on the path → contribution **0** / 路上没问到的特征贡献为 0
- A feature asked twice → add both changes / 同一特征问两次，两次变化相加
- Use the **averages (W)**, not the sample counts in parentheses / 用平均值，不是括号里的样本数

**Exam 2026-06:** root 250.0 W (10,000) → CPU_Load > 60% → Node A 340.0 W → Ambient_Temp > 25°C → Node B 390.0 W → Mem_Util > 80% → Leaf C 415.0 W

| | Answer |
| --- | --- |
| Prediction | 415 W |
| CPU_Load | +90 W (250 → 340) |
| Ambient_Temp | +50 W (340 → 390) |
| Memory_Utilization | +25 W (390 → 415) |

**Global importance, MDI (Mean Decrease in Impurity) / 全局重要性:** for every split, impurity decrease = before − after, with "impurity" = **samples × squared_error** (squared_error = node variance). Credit the decrease to the split's feature and sum per feature; the largest total is the most important feature.
每次分裂算"分裂前 − 分裂后"（乱的程度 = samples × squared_error），按特征加起来，最大的最重要。

- Example (2025-06 tree), root split on MD: 200×171.7 − (126×100.5 + 74×50.5) = 34340 − 16400 = **17940 → MD**
- Split HA ≤ 7.5: 126×100.5 − (31×64.0 + 95×65.8) = 12663 − 8235 = **4428 → HA**; CS never used → 0
- Exam 2026-06 Q2: nodes show only averages and counts → **MDI cannot be computed**: the variance (squared error) of each node is missing. Same average can hide very different spreads (50, 50, 50 vs 0, 50, 100) / 只有平均值和个数算不了 MDI，缺方差

**PDP (Partial Dependence Plot) / 部分依赖图 — global:** fix one feature to a value for **all** samples, keep the other features, predict and **average**; repeat for many values and plot the curve.
把一个特征固定成某个值，其他特征不变，所有样本预测取平均，换不同的值连成曲线。

- Example: price = area + 10 × rooms, rooms = 1, 2, 3 → area 50: (60 + 70 + 80) ÷ 3 = 70; area 80: 100
- **ICE (Individual Conditional Expectation):** one curve per sample, no averaging; **PDP = average of the ICE curves**
- **Exam 2024-06 trap:** a flat PDP does **not** mean the feature is useless; opposite effects can cancel out (Milan: price = area, Rome: price = 200 − area → average always 100). Check the ICE curves: crossing lines reveal it; parallel ICE lines mean the PDP is reliable / PDP 平不代表特征没用，效果可能抵消，看 ICE

**SHAP (SHapley Additive exPlanations) — local (and global via the summary plot):** split "prediction − average prediction" fairly among features: each feature gets its **average marginal contribution over all orders** in which features are added.
所有顺序都试一遍，取每个特征"加入前后的差"的平均。

- Example: average 100; only area 130; only rooms 110; both 150 → order area→rooms: +30, +20; order rooms→area: +10, +40 → **SHAP area = 35, rooms = 15**; check 100 + 35 + 15 = 150
- n features → n! orders (3 → 6), too many → **approximate by sampling (Monte Carlo)**:

```python
# approximate Shapley value of feature j for sample x
contributions = []
for m in range(M):                        # repeat M times
    order = random_permutation(features)  # random order
    before = predict(x, known = features before j in order)
    after  = predict(x, known = features before j in order + j)
    contributions.append(after - before)  # marginal contribution of j
shap_j = mean(contributions)
```

- Larger M → more accurate, but slower / M 越大越准但越慢
- **Summary plot:** each dot = a sample; x position = SHAP value (right raises the prediction, left lowers it); colour = feature value (red high, blue low); features sorted by importance (top = most important). Dots near 0 with mixed colours = little impact

**LIME (Local Interpretable Model-agnostic Explanations) — local:** around the sample x, approximate the black box with a simple **linear model** ("the earth looks flat when you stand in Milan").

1. Generate perturbed samples around x and predict them with the black box
2. **Kernel**: weight each sample by its distance to x (closer = larger weight), so the surrogate fits only the neighbourhood of x
3. Fit a weighted linear model; its **coefficients** are the explanation

- **Kernel width (exam 2023-06):** too wide → far points distort the line, low **fidelity**; too narrow → too few effective points, unstable results. Unstable explanations → **increase** the width / 结果不稳定就调大宽度

**Choosing a method (exam 2025-01) / 怎么选方法:**

| | Local (one prediction) | Global (whole model) |
| --- | --- | --- |
| Intrinsically interpretable model (tree, linear) | decision path contributions, coefficients | MDI, coefficients |
| Black box (post-hoc, model-agnostic) | **LIME, SHAP** | PDP, SHAP summary plot |

> Example answer: explaining why one loan was approved is a **local** problem; with a black-box model use post-hoc model-agnostic methods (LIME, SHAP); with a tree or linear model inspect the decision path or the coefficients directly.

---

## 13. Clustering Evaluation / 聚类评估

Example / 例: points on a line, A = {1, 2, 4}, B = {10, 12}.

**Internal measures (no true labels needed) / 内部评估（只看距离）:**

| Measure | Definition | Better |
| --- | --- | --- |
| Silhouette s(i) | a = mean distance to own cluster, b = mean distance to nearest other cluster, s = (b − a) ÷ max(a, b) | close to 1; ≈ 0 on the border; < 0 probably wrong cluster |
| Overall silhouette | average of s(i) over all points | larger |
| Dunn index | (min distance between points of different clusters) ÷ (max distance between points of the same cluster) | larger |
| WSS | Σ squared distance to own centroid | smaller (but always decreases with k) |
| BSS | Σ n_k × (centroid_k − overall centroid)² | larger |

- Silhouette example: P(1): a = 2, b = 10, s = 0.8; Q(2) 0.833; R(4) 0.643; S(10) 0.739; T(12) 0.793 → overall **0.76**
- Dunn example: 6 ÷ 3 = **2**; worse split A = {1, 2}, B = {4, 10, 12}: 2 ÷ 8 = 0.25 (recompute numerator and denominator for every split / 每种分法分子分母都要重找)
- WSS = 4.67 + 2 = **6.67**; BSS = 3(7/3 − 5.8)² + 2(11 − 5.8)² = **90.13**; **TSS = WSS + BSS = 96.8**
- Calculator: WSS of one cluster = n × σx² of that cluster; TSS = n × σx² of all points
- **Elbow method / 肘部法:** plot WSS vs k and choose the k after which WSS stops dropping a lot (e.g. 80, 60, 20, 17, 15 for k = 2…6 → k = 4)

**External measures (true labels needed) / 外部评估（需要真实类别）:**

- **Purity** = Σ (size of the majority class in each cluster) ÷ N. Example: clusters (3 cats, 2 dogs) and (1 cat, 4 dogs) → (3 + 4) ÷ 10 = **0.7**
- **Pairwise / 成对比较:** for every pair ask "same cluster?" and "same class?"
  - **T/F**: both answers the same → True, different → False / 两个答案一样是 T，不一样是 F
  - **P/N**: "same cluster?" yes → Positive, no → Negative / 同组是 P，不同组是 N
  - TP = same cluster & same class; FP = same cluster, different class; FN = different cluster, same class; TN = different cluster & different class
- **Rand** = (TP + TN) ÷ number of pairs; **Jaccard** = TP ÷ (TP + FP + FN) (ignores TN, stricter)
- Example: a, b cats; c, d dogs; clusters {a, b, c}, {d} → TP 1, FP 2, FN 1, TN 2 → Rand = 0.5, Jaccard = 0.25

---

## 14. Classification Metrics / 分类评估指标

**Confusion matrix / 混淆矩阵** (Positive = the model says "yes, found it", like a positive COVID test / 阳性 = 检测说"有"):

- **TP**: says positive, truly positive / 测对了
- **FP**: says positive, truly negative → **wrongly accused** / 冤枉
- **FN**: says negative, truly positive → **missed** / 漏掉
- **TN**: says negative, truly negative / 测对了

Example: 10 emails, 4 spam; the filter flags 5, of which 3 are spam → TP 3, FP 2, FN 1, TN 4 (missing numbers = total − known)

| Metric | Formula | Example |
| --- | --- | --- |
| Accuracy | (TP + TN) ÷ total | 0.7 |
| Precision | TP ÷ (TP + FP) | 0.6 |
| Recall (= TPR) | TP ÷ (TP + FN) | 0.75 |
| F1 | 2 × P × R ÷ (P + R) | 0.67 |

- Afraid of **missing** (FN) → **recall** (cancer screening): a missed patient is worse than an extra test
- Afraid of **wrongly accusing** (FP) → **precision** (spam filter deleting an important email)
- F1 lies between P and R, closer to the smaller one

**ROC & AUC:**

- The model outputs a score; **threshold**: score ≥ threshold → positive. Changing the threshold changes TP, FP, FN, TN
- **TPR** = TP ÷ (TP + FN) (recall); **FPR** = FP ÷ (FP + TN) (share of negatives wrongly flagged)
- ROC curve = (FPR, TPR) for every threshold; best point is the top-left corner **(0, 1)**
- **AUC** = area under the ROC curve: 1 perfect, 0.5 random guessing
- Example: scores 0.9 S, 0.8 S, 0.6 N, 0.4 S, 0.2 N; threshold 0.5 → TPR 2/3, FPR 1/2; threshold 0.7 → TPR 2/3, FPR 0 (better)
- Pipeline exam: for classification, METRIC can be accuracy, f1_score or roc_auc_score

---

## 15. Common Mistakes / 易错点

- [ ] Re-check the conclusion after comparing numbers / 比较大小后再核对一遍结论
- [ ] Compute means carefully: (1+3)÷2 = 2 / 求平均别算错
- [ ] Report centroids with two decimals: 2.67, not 2.6 / 中心保留两位小数
- [ ] In k-means, if any point changed cluster, recompute centroids and run another round; state convergence explicitly / 有点换类就再算一轮，最后写明收敛
- [ ] In hierarchical clustering, compare **all** clusters at each step, including two single points / 每一步比较所有组，包括两个单独的点
- [ ] Between two clusters only cross-cluster pairs count / 两组之间只看跨组点对
- [ ] Read distance matrices directly; do not subtract / 距离矩阵直接查表
- [ ] k-means has no radius; eps belongs to DBSCAN / k-means 没有半径，半径是 DBSCAN 的
- [ ] Wrong answers lose points: leave blank what you don't know, and justify every answer ("adequately motivated") / 答错倒扣分，不会的宁可空着；每题要写理由
- [ ] Calculator shows y = ax + b: the coefficient on x is the slope / 乘以 x 的是斜率，别把 a、b 搞反
- [ ] Choose α by CV on the training set, never by the test set / α 用训练集 CV 选，不用 test set
- [ ] Split first, then standardize (no data leakage) / 先分 train/test，再标准化
- [ ] Regression → MSE / R²; classification → accuracy / 回归用 MSE/R²，分类用 accuracy
- [ ] Copy numbers carefully, check the decimal point / 抄数字核对小数点
- [ ] Sanity-check R²: predictions close to real values → R² near 1 / R² 算完先看合不合理
- [ ] Negative thresholds: −0.06 > −0.1 / 负数比较大小要小心
- [ ] Single-variable mode: the 2nd column is FREQUENCY, not y / 单变量模式第二列是频数，不是 y
- [ ] Minus key `−`, not `+` and not the negative sign `(−)` / 减号别按成加号或负号
- [ ] Test RSS: Σx² − 2Σxy + Σy²; training RSS from leaves: Σxy / 两种 RSS 公式别混
- [ ] Tree contributions use node averages (W), not sample counts / 局部贡献用平均值，不是样本数
- [ ] Dunn: recompute numerator and denominator for every clustering / Dunn 每种分法都要重新找分子分母
- [ ] Jaccard denominator = TP + FP + FN / Jaccard 分母三个都要加
- [ ] Don't pick the k with the smallest WSS; use the elbow / 不能直接选 WSS 最小的 k
- [ ] When entering data, check digits are not swapped (0.793 vs 0.739) / 输入数字别颠倒
- [ ] FP = wrongly accused, FN = missed; a missed spam email is FN, not FP / 漏掉永远是 FN
- [ ] FPR uses FP of the current threshold; recompute everything when the threshold changes / 换阈值要重算
- [ ] Shapley values must add up: average + all SHAP values = prediction / SHAP 值加起来要等于预测
- [ ] LIME unstable → increase the kernel width / LIME 不稳定就调大 kernel 宽度
