# 模式识别与机器学习 – 实验报告1

姓名: 蔡亦扬| 学号: 23307130258| 专业：计算机科学与技术| 学院：计算与智能创新学院

本实验聚焦于课程第三讲决策树与最近邻问题的内容，包含以下三部分：

- 分类评测指标 (20%)
- 决策树 (40%)
- 最近邻问题 (40%)

占课程总分：20% | 提交截至日期：<u>10/13 23:59</u>

## <u>分类评测指标 – 20%</u>

1. **指标定义 – 5%**

   用你的话简单解释下列每个指标的含义以及使用场景

   | 指标          | 含义 | 使用场景 |
   | ------------- | ---- | -------- |
   | Accuracy      | 预测正确的样本数占全部样本数，衡量整体分类是否准确 | 类别分布相对均衡的分类任务，用于快速评估模型总体表现 |
   | MSE           | 预测值与真实值差异的平方平均，反映误差整体量级     | 回归或概率输出等连续预测场景，观察预测偏差大小         |
   | Precision     | 被判为正例的样本中，有多少是真正的正例             | 更关心“误报”代价的任务，如垃圾邮件过滤、疾病诊断初筛   |
   | Recall        | 所有真实正例中，被成功识别为正例的比例             | 更关心“漏报”风险的任务，如安全告警、疾病复诊筛查       |
   | F1            | Precision 与 Recall 的调和平均，兼顾二者平衡        | 正负样本不均衡或需兼顾误报与漏报的分类任务             |

2. **代码实现（见 Part1_ReadMe.md）– 10%**

   请在报告中贴出你实现的五个核心函数的代码并简单说明实现的逻辑：

   - `accuracy_score()`
   - `mean_squared_error()`
   - `precision_score()`
   - `recall_score()`
   - `f1_score()`

   ```python
   # classification/accuracy_error.py
   import numpy as np
   
   def accuracy_score(y_true, y_pred):
       accuracy = -1
       y_true = np.asarray(y_true).reshape(-1)
       y_pred = np.asarray(y_pred).reshape(-1)
       if y_true.shape[0] != y_pred.shape[0]:
           raise ValueError("y_true and y_pred must have the same length.")
       correct_predictions = np.sum(y_true == y_pred)
       accuracy = correct_predictions / y_true.shape[0]
       return accuracy
   
   def mean_squared_error(y_true, y_pred):
       error = -1
       y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
       y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
       if y_true.shape[0] != y_pred.shape[0]:
           raise ValueError("y_true and y_pred must have the same length.")
       squared_differences = (y_true - y_pred) ** 2
       error = np.mean(squared_differences)
       return error
   
   # classification/evaluation_metrics.py
   EPS = 1e-12
   
   def precision_score(y_true, y_pred):
       tp, fp, _, _ = _binary_counts(y_true, y_pred)
       precision = tp / (tp + fp + EPS)
       return precision
   
   def recall_score(y_true, y_pred):
       tp, _, fn, _ = _binary_counts(y_true, y_pred)
       recall = tp / (tp + fn + EPS)
       return recall
   
   def f1_score(y_true, y_pred):
       precision = precision_score(y_true, y_pred)
       recall = recall_score(y_true, y_pred)
       f1 = 2 * (precision * recall) / (precision + recall + EPS)
       return f1
   ```

   - `accuracy_score`: 统一把标签转成一维数组，校验长度一致后计算命中数量，最后除以样本量得到整体准确率。
   - `mean_squared_error`: 将输入转换成浮点向量，逐元素计算平方误差并取平均，衡量预测值的整体偏差。
   - `_binary_counts`: 依据二分类标签一次性统计 TP/FP/FN/TN，后续评估函数直接复用这些统计量。
   - `precision_score`: 在 `_binary_counts` 的输出基础上计算 `TP / (TP + FP)`，并通过 `EPS` 防止分母为零，关注“判成正例的准确性”。
   - `recall_score`: 计算 `TP / (TP + FN)`，同样加上 `EPS` 做数值保护，衡量“正例被找回的比例”。
   - `f1_score`: 先得到 Precision/Recall，再用调和平均综合两者，避免只优化其中一个指标导致另一项过低。

3. **测试与结果 – 5%**

   完成评测指标的函数后，运行 `test.py`，把输出 log 的截图粘贴在下面一行：

   > 
   >
   > ![test1](/home/caiyy/.config/Typora/typora-user-images/image-20251013175843922.png)

   使用以下新的输入测试，你可以在 `test.py` 中更改，或自己计算：

   | Y_true                         | Y_predict                      |
   | ------------------------------ | ------------------------------ |
   | `[0, 1, 0, 1, 1, 0, 1, 0, 0]` | `[1, 1, 1, 1, 1, 0, 0, 0, 0]` |

   **Confusion Matrix**

   | TP | FP | FN | TN |
   | -- | -- | -- | -- |
   | 3  | 2  | 1  | 3  |

   **指标计算**

   | Accuracy | MSE | Precision | Recall | F1  |
   | -------- | --- | --------- | ------ | --- |
   | 0.6667   | 0.3333 | 0.6000 | 0.7500 | 0.6667 |

## <u>决策树 – 40%</u>

1. 简要解释下决策树，以及其优缺点 – 2%

   | 决策树概述 | 优缺点 |
   | ---------- | ------ |
   | 通过递归地基于特征阈值划分数据，把特征空间分成若干“叶子”，每个叶子对应一个类别或回归值 | **优点**：可解释性强、能处理混合型特征、无需大量预处理；**缺点**：易过拟合、对噪声敏感、对连续特征切分较粗糙 |

2. **代码实现（见 Part2_ReadMe.md）– 15%**

   请在报告中贴出你实现的四个核心函数的代码并简单说明实现的逻辑：

   - `criterion.py` 中的 `__info_gain(...)`
   - `criterion.py` 中的 `__info_gain_ratio(...)`
   - `criterion.py` 中的 `__gini_index(...)`
   - `criterion.py` 中的 `__error_rate(...)`

   ```python
   # decision_tree/criterion.py
   def _entropy(counts):
       total = float(sum(counts.values()))
       if total == 0.0:
           return 0.0
       entropy = 0.0
       for cnt in counts.values():
           if cnt:
               p = cnt / total
               entropy -= p * math.log2(p)
       return entropy
   
   def _gini(counts):
       total = float(sum(counts.values()))
       if total == 0.0:
           return 0.0
       g = 1.0
       for cnt in counts.values():
           p = cnt / total
           g -= p * p
       return g
   
   def _error_rate(counts):
       total = float(sum(counts.values()))
       if total == 0.0:
           return 0.0
       majority = max(counts.values())
       return 1.0 - majority / total
   
   def __info_gain(y, l_y, r_y):
       all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
       total = float(len(y))
       left_total = float(len(l_y))
       right_total = float(len(r_y))
       parent_entropy = _entropy(all_labels)
       weighted_children = 0.0
       if total > 0.0:
           if left_total > 0.0:
               weighted_children += (left_total / total) * _entropy(left_labels)
           if right_total > 0.0:
               weighted_children += (right_total / total) * _entropy(right_labels)
       info_gain = parent_entropy - weighted_children
       return info_gain
   
   def __info_gain_ratio(y, l_y, r_y):
       info_gain = __info_gain(y, l_y, r_y)
       total = float(len(y))
       if total == 0.0:
           return 0.0
       p_l = len(l_y) / total
       p_r = len(r_y) / total
       split_info = 0.0
       if p_l > 0:
           split_info -= p_l * math.log2(p_l)
       if p_r > 0:
           split_info -= p_r * math.log2(p_r)
       if split_info <= 0.0:
           return 0.0
       info_gain = info_gain / split_info
       return info_gain
   
   def __gini_index(y, l_y, r_y):
       all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
       total = float(len(y))
       left_total = float(len(l_y))
       right_total = float(len(r_y))
       before = _gini(all_labels)
       after = 0.0
       if total > 0.0:
           if left_total > 0.0:
               after += (left_total / total) * _gini(left_labels)
           if right_total > 0.0:
               after += (right_total / total) * _gini(right_labels)
       return before - after
   
   def __error_rate(y, l_y, r_y):
       all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
       total = float(len(y))
       left_total = float(len(l_y))
       right_total = float(len(r_y))
       before = _error_rate(all_labels)
       after = 0.0
       if total > 0.0:
           if left_total > 0.0:
               after += (left_total / total) * _error_rate(left_labels)
           if right_total > 0.0:
               after += (right_total / total) * _error_rate(right_labels)
       return before - after
   ```

   - `_entropy`/`_gini`/`_error_rate`: 针对计数字典统一计算三种纯度指标，遇到空节点直接返回 0，避免重复代码。
   - `__info_gain`: 按父节点熵减去左右子树熵的加权和，信息增益越大说明该划分提升纯度越显著。
   - `__info_gain_ratio`: 在信息增益基础上除以划分信息量，仅当分裂两侧都有样本时才返回正值，抑制“只切出单个样本”的极端情况。
   - `__gini_index`: 使用“分裂前 Gini − 分裂后加权 Gini”的形式量化纯度改进，保持与熵类指标同向。
    - `__error_rate`: 以分类误差率为度量，统计最常见标签之外的样本比例，通过加权差值判断分裂是否减少错误。

3. **测试和可视化 – 15%**

   完成 `criterion.py` 的四个函数后，运行 `test_decision_tree.py`，将会输出对应的 accuracy 和四张图片。请把图片粘贴在下面，并在图注中写明 **Accuracy、树深度、叶子数**。

   > *iris_error_rate*
   >
   > ![iris_error_rate](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/decision_tree/output/iris_error_rate.png)
   >
   > _iris_error_rate_  
   > Accuracy = 0.93\| Tree_depth = 4\| Tree_leaf_num = 7

   > 
   >
   > _iris_gini_  
   >
   > ![iris_gini](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/decision_tree/output/iris_gini.png)
   >
   > Accuracy = 0.90\| Tree_depth = 6\| Tree_leaf_num = 9

   > 
   >
   > _iris_info_gain_  
   >
   > ![iris_info_gain](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/decision_tree/output/iris_info_gain.png)
   >
   > Accuracy = 0.93\| Tree_depth = 6\| Tree_leaf_num = 9

   > 
   >
   > _iris_info_gain_ratio_  
   >
   > ![iris_info_gain_ratio](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/decision_tree/output/iris_info_gain_ratio.png)
   >
   > Accuracy = 0.90\| Tree_depth = 6\| Tree_leaf_num = 9

4. **进一步探索 – 8%**

   本部分希望同学们在固定训练/验证/测试划分下，调参使测试集 Accuracy 尽可能高。

   下表给出 `decision_tree.py` 中的可调参数：

   | 可调参数            | 参数说明                                         | 可选值                              |
   | ------------------- | ------------------------------------------------ | ----------------------------------- |
   | criterion           | 分裂度量不同，偏好不同；可先粗选再细调           | {info_gain, info_gain_ratio, gini, error_rate} |
   | splitter            | `random` 随机阈值更具多样性，配合多次 seed 取较稳的方案 | {best, random}                      |
   | max_depth           | 控制树深，限制过拟合                             | {None, 2-10}                        |
   | min_samples_split   | 节点最小样本数，越大越保守                       | {2, 3, 4, 5, 10, 20, ...}           |
   | min_impurity_split  | 最小分裂增益阈值，越高越保守                     | {0, 1e-4, 1e-3, 1e-2, ...}          |
   | max_features        | 每次候选的特征子集大小，和随机森林思想类似       | {None, "sqrt", "log2", 1..d, 0.5..1.0} |

   **建议的搜索顺序**

   1. 先在 {criterion} × {splitter} 上做粗选。
   2. 固定上一步较优组合，再对 {max_depth, min_samples_split, min_impurity_split} 做粗到细的网格。
   3. 最后微调 max_features，在不降 Accuracy 的前提下简化树（更浅 / 更少叶）。

   你可以在下表中进行参数的尝试（至少 3 组），准确率越高越好：

   | 项目                | 1 | 2 | 3 | 4 |
   | ------------------- | - | - | - | - |
   | criterion           | <mark>gini</mark> | error_rate | info_gain | info_gain_ratio |
   | splitter            | <mark>random</mark> | best | best | random |
   | max_depth           | <mark>6</mark> | None | 4 | 6 |
   | min_samples_split   | <mark>2</mark> | 2 | 2 | 4 |
   | min_impurity_split  | <mark>0.001</mark> | 0.001 | 0.0 | 0.0 |
   | max_features        | <mark>None</mark> | sqrt | None | sqrt |
   | **accuracy**        | <mark>1.0000 (depth=6, leaves=13)</mark> | 0.9333 (depth=3, leaves=5) | 0.9333 (depth=4, leaves=7) | 0.9000 (depth=5, leaves=9) |

   <mark>高亮你选中的最佳参数组合</mark>，在下面粘贴输出的树图：

   > ![final_gini](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/decision_tree/output/iris_gini.png)


## <u>最近邻问题 – 40%</u>

1. **简要解释下 kNN 算法，以及其优缺点 – 2%**

   | kNN 概述 | 优缺点 |
   | -------- | ------ |
   | 基于距离度量找到测试样本最接近的 K 个训练样本，并通过投票或平均得到预测 | **优点**：实现简单、对多类别友好、天然适配增量数据；**缺点**：查询时成本高、对特征缩放和噪声敏感、需要保存全部训练集 |

2. **代码实现（见 Part3_ReadMe.md）– 15%**

   请在报告中贴出你实现的三个核心函数的代码并简单说明实现的逻辑：

   - `pairwise_dist()`（包含 L2 – two_loops、L2 – no_loops、Cosine）
   - `knn_predict()`
   - `select_k_by_validation()`

   ```python
   # k_nerest_neighbors/knn_student.py
   def pairwise_dist(X_test, X_train, metric, mode):
       X_test = np.asarray(X_test, dtype=np.float64)
       X_train = np.asarray(X_train, dtype=np.float64)
       Nte, D = X_test.shape
       Ntr, D2 = X_train.shape
       assert D == D2, "Dim mismatch between test and train."
   
       if metric == "l2":
           if mode == "two_loops":
               dists = np.zeros((Nte, Ntr), dtype=np.float64)
               for i in range(Nte):
                   for j in range(Ntr):
                       dists[i, j] = np.sqrt(np.sum((X_test[i] - X_train[j]) ** 2))
               return dists
           elif mode == "no_loops":
               dists = np.zeros((Nte, Ntr), dtype=np.float64)
               dists = np.sqrt(np.sum(X_test**2, axis=1, keepdims=True) + 
                               np.sum(X_train**2, axis=1) - 
                               2 * np.dot(X_test, X_train.T))
               return dists
           else:
               raise ValueError("Unknown mode for L2.")
       elif metric == "cosine":
           dists = np.zeros((Nte, Ntr), dtype=np.float64)
           X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
           X_train_norm = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
           cosine_similarity = np.dot(X_test_norm, X_train_norm.T)
           dists = 1 - cosine_similarity
           return dists
       else:
           raise ValueError("metric must be 'l2' or 'cosine'.")
   
   def knn_predict(X_test, X_train, y_train, k, metric, mode):
       dists = pairwise_dist(X_test, X_train, metric=metric, mode=mode)
       y_train = np.asarray(y_train).reshape(-1).astype(int)
       y_pred = np.zeros(dists.shape[0], dtype=int)
       for i in range(dists.shape[0]):
           idx = np.argsort(dists[i])[:k]
           neighbors = y_train[idx]
           counts = np.bincount(neighbors)
           y_pred[i] = np.argmax(counts)
       return y_pred
   
   def select_k_by_validation(X_train, y_train, X_val, y_val, ks, metric, mode):
       accs = []
       for k in ks:
           y_pred = knn_predict(X_val, X_train, y_train, k, metric, mode)
           accuracy = np.mean(y_pred == y_val)
           accs.append(accuracy)
       best_k = ks[np.argmax(accs)]
       return best_k, accs
   ```

   - `pairwise_dist (two_loops)`: 双重循环逐个计算欧氏距离，结构直观，便于验证公式是否正确。
   - `pairwise_dist (no_loops)`: 通过向量化展开平方和定理，避免显式循环，提升大规模样本时的效率。
   - `pairwise_dist (cosine)`: 先对样本做 L2 归一化，再利用点积求余弦相似度并转化为距离（1−相似度）。
   - `knn_predict`: 调用距离矩阵，对每个测试样本取最近 k 个标签，使用 `np.bincount` 做多数表决，`np.argmax` 自然返回票数最高且标签最小的类别。
   - `select_k_by_validation`: 枚举给定的 k 列表并调用预测函数，记录验证准确率曲线，同时挑选出表现最好的 k 供后续使用。

3. **测试与结果可视化 – 15%**

   下面是我们在固定测试中使用的数据集参数设定，你可以在 `data_generate.py` 中查看和改变这些参数：

   > ![knn_arg](/home/caiyy/.config/Typora/typora-user-images/image-20251013204121466.png)

   **3.1** 完成 `knn_student.py` 的代码块后，在 `data_generate.py` 中设置以上的参数，然后运行 `test_knn.py`，把输出的 log 粘贴在下面：

   > ![knntest](/home/caiyy/.config/Typora/typora-user-images/image-20251013204055256.png)

   

   **3.2** 当所有的测试都通过后，运行 `knn_student.py`，程序会调用 `viz_knn.py` 中的可视化函数，输出 `knn_k_curve.png` 和 `knn_boundary_grid.png` 两个图像。需要你在 `knn_student.py` 中修改 `metric` 参数，分别生成使用 “L2” 和 “cosine” 的图像，贴在下面。

   a. Matric = “L2”

   > ![knn_k_curve](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/knn_k_curve.png)
   >
   > ![knn_boundary_grid](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/knn_boundary_grid.png)

   b. Metric = “cosine”  
   ![cosine1](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/knn_k_curve.png)

   ![cosine2](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/knn_boundary_grid.png)

   注意示例图片仅供参考，不是要求的数据集结果，删掉上面的示例图片，改成你自己的结果。

   **3.3** 观察你得到的测试报告和图像，回答以下问题：

   a. 准确率最高的 k 值是多少？

   |        | Best_k | Accuracy |
   | ------ | ------ | -------- |
   | L2     | 11     | 0.8240   |
   | Cosine | 13     | 0.8240   |

   b. K 对边界的影响，K=1 时边界为何 “锯齿 / 细碎”？K 增大为何更平滑？

   当 k=1 时，每个测试点直接继承最近邻训练样本的标签，决策边界需要严格穿过所有训练样本，因此会紧贴训练集形成“锯齿状”界线；随着 k 增大，判定依据是多个邻居的多数票，局部噪声或离群点对结果影响被平均掉，边界自然变得平滑，呈现从高度拟合到更平滑泛化的过渡。

   c. 在相同的数据下 “L2” 和 “cosine” 有什么差异（结合图像解释）？它们各自测量的 “距离” 是什么？

   L2 距离关注的是欧式几何距离，适合尺度统一、坐标差异等价表示相似度的情形，其决策边界倾向于形成圆形或椭圆形的等距轮廓，边界呈现为曲线；余弦距离只看方向夹角，忽略向量的长度，擅长刻画“方向相近但幅度可不同”的数据，其决策边界呈直线状从原点辐射。实际可视化中，L2 边界主要由空间距离主导，而余弦边界更关注向量方向，对模长差异不敏感，因此同一数据集在两种距离下的分界形状和误分类区域通常不同

4. **进一步探索 – 8%**

   本部分希望同学们能够选择自己感兴趣的问题进行探索，并完成一份简单的实验报告。我们提供三个样例问题，同学们可以选择其中之一进行探索，更加鼓励自己寻找一个问题进行实验。

   - 样例问题 1：探索适合 “L2” 和 “cosine” 的数据场景。通过修改 `data_generate.py` 中的数据参数和 k 的选择，分别搜索能够在 “L2” 和 “cosine” 方法下达到高准确度（95% 以上）的数据集，分析两种距离计算方式适配的场景和数据结构。
   - 样例问题 2：类内方差（重叠程度）对 k 的影响。通过修改 `data_generate.py` 中的 `CLUSTER_STD` 参数，探索其和 k 的联系，可从以下问题展开：`CLUSTER_STD ↑`（更模糊）时，`best_k` 是否趋向更大？为什么从 “锯齿→平滑” 的边界有助于抗噪？对比 k=1 与 k=best_k 的误分类点分布（图表 ×），哪些区域最难？
   - 样例问题 3：探索数据结构与过拟合 / 欠拟合的关系。过拟合 / 欠拟合的定义以及它们呈现的结果是什么样的？什么样的参数会导致过拟合 / 欠拟合的发生？

   简易实验报告必须包含以下几个部分：

   1. 具体的探索问题  
   2. 探索的方法（修改了哪些参数，为什么需要改这些参数）  
   3. 输出的结果（包括图像和 log）
   4. 最终的结论
   
   ### 实验报告：类内方差（重叠程度）对 kNN 中最佳 k 值的影响
   
   **姓名**：蔡亦扬
   
   **学号**：23307130258
   
   #### 1. 探索的具体问题
   
   - 选择样例问题 2：研究类内方差（`CLUSTER_STD`）对 kNN 最优 `k` 的影响：当数据簇逐渐模糊时，`best_k` 是否会增大？边界会如何由“锯齿”变为更平滑？  
   - 重点观察不同噪声水平下验证/测试准确率、最佳 `k` 的变化，以及误分类点是否集中在类间重叠区域。

   #### 2. 探索的方法

   - 在 `data_generate.py` 中固定随机种子、类别数量（3 类）和各划分样本数（train/val/test ≈ 180/60/60），仅调节 `CLUSTER_STD ∈ {1.0, 2.0, 3.0, 4.0, 5.0}`，让类内方差逐步增大以模拟噪声增强。  
   - 每次调整后重新生成数据集，保持其他参数（类中心、先验概率等）不变，保证比较只受噪声影响。  
   - 编写 `explore_knn_cluster_std.py` 脚本自动化流程：对每个 `CLUSTER_STD` 调用 `generate_and_save` 生成数据，随后以 `ks=[1,3,5,7,9,11,13,15,17]`、`metric="l2"`、`mode="no_loops"` 运行 `select_k_by_validation`。  
   - 脚本在验证集上记录 `accs` 与 `best_k`，并对 train+val 合并数据评估测试准确率；同时调用 `plot_k_curve`、`plot_decision_boundary_multi` 输出 `knn_k_curve.png` 与 `knn_boundary_grid.png`，集合路径保存到 `summary.txt` 方便整理。  
   - 最后比较 `k=1` 与 `k=best_k` 的边界差异及误分类分布，分析较大 `k` 是否能缓解噪声带来的预测抖动。

#### 3. 输出的结果

| CLUSTER_STD | best_k | 验证准确率（best_k） | 测试准确率 |
| ----------- | ------ | ------------------- | ---------- |
| 1.00        | 1      | 1.0000              | 1.0000     |
| 2.00        | 9      | 0.9760              | 0.9760     |
| 3.00        | 11     | 0.9120              | 0.8960     |
| 4.00        | 11     | 0.8240              | 0.8080     |
| 5.00        | 15     | 0.7440              | 0.7200     |

**CLUSTER_STD = 1.00**

![curve_std_100](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_1.00/knn_k_curve.png)
![boundary_std_100](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_1.00/knn_boundary_grid.png)

**CLUSTER_STD = 2.00**

![curve_std_200](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_2.00/knn_k_curve.png)
![boundary_std_200](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_2.00/knn_boundary_grid.png)

**CLUSTER_STD = 3.00**

![curve_std_300](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_3.00/knn_k_curve.png)
![boundary_std_300](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_3.00/knn_boundary_grid.png)

**CLUSTER_STD = 4.00**

![curve_std_400](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_4.00/knn_k_curve.png)
![boundary_std_400](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_4.00/knn_boundary_grid.png)

**CLUSTER_STD = 5.00**

![curve_std_500](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_5.00/knn_k_curve.png)
![boundary_std_500](/home/caiyy/FDU/year3/PRML/FDU-PRML-2025Fall/Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_5.00/knn_boundary_grid.png)

- 验证曲线显示：`CLUSTER_STD` 越大，峰值逐渐从 `k=1` 向更大的 `k` 偏移（2.0/3.0 时峰值在 9/11，5.0 时升至 15），同时整体准确率下滑。  
- 决策边界图表明：低噪声下 `k=1` 能细致刻画分隔线；随着标准差增加，若仍使用 `k=1` 会产生零散的误判点，改用更大的 `k`（如 9/11/15）可以平滑边界并包容噪声。

#### 4. 最终的结论

- 当 `CLUSTER_STD` 从 1.0 增至 5.0 时，最佳 `k` 从 1 升到 15，验证/测试准确率逐步下降，表明噪声增大后需要更多邻居来抵御局部异常点。  
- 可视化显示：低噪声下 `k=1` 即可获得清晰边界；噪声增大时，原本的“锯齿”划分导致多处误判，而较大的 `k`（如 9/11/15）能产生更平滑且泛化更好的决策边界。  
- 综合来看，kNN 对类内方差高度敏感：适当调大 `k` 有助于提升鲁棒性，但过大可能牺牲细节。实际应用需借助验证曲线寻找精度与平滑度之间的平衡点。
## <u>提交</u>

- 完成后删除所有红色字体的提示部分，不要改动黑色字体的题干部分。
- 选择合适的字体和行间距，保证美观和可读性。
- 保证粘贴的图像大小合适，图中内容清晰可见。
- 完成后导出为 **pdf**，把文件名改为 **PRML-实验1-姓名**，提交到 elearning 上，不需要提交单独的图像、代码文件或压缩包。
- 截至日期见开头。
