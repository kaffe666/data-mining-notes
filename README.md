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
10. [Common Mistakes / 易错点](#10-common-mistakes--易错点)

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

## 10. Common Mistakes / 易错点

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
