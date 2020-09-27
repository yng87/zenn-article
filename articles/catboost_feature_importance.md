---
title: "CatBoost の feature importance"
emoji: "🐈" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

# 概要
CatBoost が計算してくれる feature importance の定義がドキュメントを見てもよくわからなかったので調べた。

いくつか importance の定義があるが、ここでは `PredictionValuesChange` で定義されるものについてまとめる。これはその feature が最終的な予測値にどれだけ影響を持つかという指標である。

# CatBoost で importance を出してみる

まずはシンプルなモデルを作って実際に feature importance を出力してみる。ここでは iris dataset を使い以下のようなパラメーターでモデルを訓練する。

```python
from catboost import CatBoostClassifier, Pool
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

model = CatBoostClassifier(iterations=1, depth=2, random_seed=42)

model.fit(X, y)
```

あやめを3クラスに分ける分類問題を、深さが2の tree 一本で解いている。

これを実行すると結果として以下のようなモデルが得られる

```python
model.plot_tree(tree_idx=0)
```
![](https://github.com/yng87/zenn-article/raw/master/images/tree_iris.png)

まず、特徴量2の値で分岐をし、その後特徴量3の値で分岐をしている様子が確認できる。CatBoost は他のGBDTライブラリと異なり、(デフォルトでは) oblivious trees を作成する。そのため、同じ深さでは全てのノードが同じ条件で分岐を行う。

ここで leaf に入っている `val` が何者か？ということだが、今解いているのは3クラスの分類であり、あるサンプルがクラス $j$ に属する確率はソフトマックス関数

$$p(class = j) = \frac{\exp(a_{j})}{\sum_j \exp(a_{j})}$$

で表される。CatBoostClassifier が出力するのはこの $a_{j}$ である。

このモデルの feature importance を見てみると

```python
model.get_feature_importance(Pool(X, y), type="PredictionValuesChange")
```

```bash
array([ 0.        ,  0.        , 44.49955092, 55.50044908])
```

となる。木の分岐に特徴量0と1が使われていないので、それらに関してはゼロになっている。これをみると特徴量3が特徴量2に比べて少しだけ重要であるらしい。
この定義による importance は負にならないので、合計100になるよう normalize された値が出力されている。

# CatBoost は何を計算しているか？


[公式ドキュメント](https://catboost.ai/docs/concepts/fstr.html#fstr__regular-feature-importance)によると、feature F の feature importance は leaf の ペアに関してそれぞれの予測値とその加重平均を比較して計算されている

> Leaf pairs that are compared have different split values in the node on the path to these leaves. If the split condition is met (this condition depends on the feature F), the object goes to the left subtree; otherwise it goes to the right one.

$$importance_F=\sum_{tree, leafs_F} c_1(v_1 - avr)^2 + c_2(v_2 - avr)^2 \tag{1}$$

$c_1$, $c_2$ は比較している二つの leaf node それぞれの持つサンプル数（木を辿ってその leaf に属することになるデータの数）、$v_1$, $v_2$ はそこでの予測値の値 (val) である。$avr$ は以下のような $v_1$ と $v_2$ の加重平均である

$$avr = \frac{c_1v_1 + c_2 v_2}{c_1 + c_2}.$$

つまり、PredictionValuesChange による feature importance はその特徴量を使って分岐した場合としなかった場合でどれくらい予測値が変わるかという指標になっている。

比較に使う leaf pair はどのように選べば良いのだろうか？まず leaf 直上の分岐に使われる特徴量に関してはわかりやすい。同じ parent を持つ leaf 二つを使って $c_i$ と $v_i$ を計算すれば良い。

では木のもっと上の方に現れる特徴量についてはどうすれば良いだろうか？一般の木では leaf 間のペアを定義することはできなさそうだが、CatBoost が生成する oblivious tree に限ってはうまく定義できる。
というのも、木の各深さで分岐の条件が同一ということは、結局CatBoostは単に分岐条件を一列に並べて Yes/No の判定を繰り返しているに過ぎない。なので、d 番目の分岐が異なり、その前後の分岐は全て同じような leaf node のペアを必ず作ることができる。そのようなペアを集めてきて上の (1) 式で深さ d 番目のfeature のimportance を計算することができそうだ。

以上のことはドキュメントからだとわかりづらいが、[ソースコードの該当箇所](https://github.com/catboost/catboost/blob/49e24bba3279ae1ac22146b8322e37d86e6049bf/catboost/libs/fstr/feature_str.h#L218-L256)を見ると多分そうだろうと想像できる。[^1]

# 実際に計算して確かめてみる

今の説明を実際に先ほど計算した iris の例で確かめてみる。[^2] leaf node を左から leaf0, leaf1, ..., leaf3 と呼ぶことにする。まず leaf 直上の分岐に使われている feature 3 については、(leaf0, leaf1) と (leaf2, leaf3) に関してそれぞれ (1) 式を計算する

```python
# get border values for each feature
borders = model.get_borders() 
# {0: [], 1: [], 2: [4.949999809265137], 3: [0.44999998807907104]}

# get samples in each leaf node
leaf_samples = []
leaf_samples.append(X[(X[:, 2] <= borders[2][0]) & (X[:, 3] <= borders[3][0])])
leaf_samples.append(X[(X[:, 2] <= borders[2][0]) & (X[:, 3] > borders[3][0])])
leaf_samples.append(X[(X[:, 2] > borders[2][0]) & (X[:, 3] <= borders[3][0])])
leaf_samples.append(X[(X[:, 2] > borders[2][0]) & (X[:, 3] > borders[3][0])])

[s.shape[0] for s in leaf_samples] # [48, 56, 0, 46]

# get leaf values
vals = model.get_leaf_values().reshape(4, 3)
#
# array([[ 0.84210526, -0.42105263, -0.42105263],
#        [-0.38461538,  0.67692308, -0.29230769],
#        [ 0.        ,  0.        ,  0.        ],
#        [-0.41818182, -0.36363636,  0.78181818]])

ftr_imp = np.zeros(4)
feature_idx = 3

for i1, i2 in [(0, 1), (2, 3)]:
  # weight
  c1 = leaf_samples[i1].shape[0]
  c2 = leaf_samples[i2].shape[0]
  c_sum = c1 + c2

  # compute values difference
  for j in range(3):
    v1 = vals[i1][j]
    v2 = vals[i2][j]

    avg_v = (c1*v1 + c2*v2) / c_sum
    diff = c1*(v1 - avg_v)**2+ c2*(v2 - avg_v)**2

    ftr_imp[feature_idx] += diff

ftr_imp
```

結果

```bash
array([ 0.        ,  0.        ,  0.        , 70.48167229])
```

次に、root node で使われている feature 2 について計算する。ここでペアを組むのは (leaf0, leaf2), (leaf1, leaf3) である。例えば、leaf0 と leaf2 は、faeture 3 に関する分岐は両方 No だが、root での feature 2 に関する分岐は異なる。

```python
feature_idx = 2

for i1, i2 in [(0, 2), (1, 3)]:
  c1 = leaf_samples[i1].shape[0]
  c2 = leaf_samples[i2].shape[0]
  c_sum = c1 + c2

  for j in range(3):
    v1 = vals[i1][j]
    v2 = vals[i2][j]

    avg_v = (c1*v1 + c2*v2) / c_sum
    diff = c1*(v1 - avg_v)**2+ c2*(v2 - avg_v)**2

    ftr_imp[feature_idx] += diff

ftr_imp
```

結果は

```bash
array([ 0.        ,  0.        , 56.51130428, 70.48167229])
```

最後にこれを合計100%になるよう normalize すると

```bash
array([ 0.        ,  0.        , 44.49955092, 55.50044908])
```

となる。これはCatBoost の APIを使って求めた結果に一致している。より複雑な木になった時も同様に計算できるはずである。

# Non-oblivious tree の場合

一般の対称ではない木の場合は上記の方法は使えない。CatBoost は oblivious の制限を付けずに学習することもできるが、その際の importance は

1. leaf node の parent に対して (1) 式で feature F の importance を計算
2. その leaf node を削除し、parent だった node を新たに leaf とする。それらのサンプル数は $c_1 + c_2$で、val は $avr$ で割り当てる

ということを繰り返して計算される。この方法による importance は oblivious tree に対して計算された値とは一般に一致しない。
[詳しくはソースコードの該当箇所](https://github.com/catboost/catboost/blob/49e24bba3279ae1ac22146b8322e37d86e6049bf/catboost/libs/fstr/feature_str.h#L132-L216)を見るとわかる。

上で使った iris のモデルで実際に計算してみよう。
まずは leaf について
```python
ftr_imp = np.zeros(4)

parent_vals = np.zeros((2, 3))
parent_weight = np.zeros(2)

for i in range(0, 4, 2):
  c1 = leaf_samples[i].shape[0]
  c2 = leaf_samples[i+1].shape[0]
  c_sum = c1 + c2

  parent_weight[i // 2] = c_sum

  for j in range(3):
    v1 = vals[i][j]
    v2 = vals[i+1][j]

    avg_v = (c1*v1 + c2*v2) / c_sum
    diff = c1*(v1 - avg_v)**2+ c2*(v2 - avg_v)**2

    ftr_imp[3] += diff

    parent_vals[i // 2][j] = avg_v
```
そして root について
```python
c1 = parent_weight[0]
c2 = parent_weight[1]
c_sum = c1 + c2

for j in range(3):
  v1 = parent_vals[0][j]
  v2 = parent_vals[1][j]
  avg_v = (c1*v1 + c2*v2) / c_sum
  diff = c1*(v1 - avg_v)**2 + c2*(v2 - avg_v)**2

  ftr_imp[2] += diff

ftr_imp / sum(ftr_imp) * 100
```
結果は
```bash
array([ 0.        ,  0.        , 46.61367922, 53.38632078])
```
となり、確かに oblivious の時とは異なる値になる。

[^1]: 部分的にソースを眺めただけなので勘違いしている可能性はあるが…
[^2]: ノートブック: https://colab.research.google.com/drive/1CiRm_vby9S4c_q5dWq_oY88hSehC17_U?usp=sharing