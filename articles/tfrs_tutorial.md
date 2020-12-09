---
title: "TFRSで2-stageレコメンドシステムを実装する"
emoji: "🐈" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["機械学習"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

少し前に、[Tensorflow Recommenders (TFRS)](https://www.tensorflow.org/recommenders) というライブラリがリリースされていました。これは、Keras を使用してレコメンドシステムを作るためのモジュール群のようです。


近年は深層学習を用いたレコメンドシステムが実応用されるようになってきました。特に、YouTube に代表されるような大規模なプロダクトでは、2-stage レコメンドと呼ばれる形での利用がスタンダードになっています((https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/45530.pdf))。このTFRSは、2-stage レコメンドシステムを念頭に作られているようです。

ということでTFRSを使って、2-stage の深層推薦システムの実装・評価をしてみようと思います。使うデータセットは、TFRSのチュートリアルにも出てきている、Movielens 100k です。これくらいだと Google colaboratory でも気軽に実行できます。

# 2-stage レコメンドとは
レコメンドの目的は、ユーザーに engagement が高くなるようなアイテムを提示することです。そのために、モデルを作ってユーザーの好みやアイテムの人気を学習させ、できたモデルを使ってアイテムにスコアをつけます。そのスコアが高い順にユーザーにアイテムが推薦されることになります。

このスコアリングを単一のモデルで行うのは大変です。ユーザーごとに全アイテムのスコアリングとスコアに応じたソートが必要なため、対象のアイテムが多くなるほど計算量が増えます。例えば YouTube や Twitter のような大規模なシステムでこれをリアルタイムに行うのは latency の観点で許容されないでしょう。
あらかじめバッチで処理してキャッシュをしておくという方法もありますが、特徴量の次元が多くなるほど（例えばユーザーIDの他に、検索クエリをコンテキストとして持つ場合など）膨大な容量が必要になりますし、リアルタイムな情報を使うことができなくなります。

これを解決する方法が 2-stage レコメンドです。名前の通り二段階のシステムになっていて、以下のような構成を取ります

1. Candidate generation: アイテムコーパス全体から、コンテクストに応じた関連アイテム (candidates) を数十から数百取得する。速度重視。
2. Re-ranking: Candidates をより精緻なモデルでランクづけする。質重視。

Candidate generation に用いるモデルは retrieval model と呼ばれ、数百万や数十億のアイテムコーパスから少数の関連アイテムを抽出するのに使われます。このアルゴリズムは高速である必要があり、その代償として多少質は落ちることを許容します。その後取得した candidates を後続の ranking model がランクづけします。Retrieval model は高速な代わりに無関係のアイテムも推薦してくる可能性がありますが、ranking model がそのようなものを落として、ユーザーの興味に関連するようなアイテムを上位に持ってくるようにします。

この2-stage レコメンドシステムは大規模プロダクトでリアルタイムな推論を行うには必須で、YouTube、Twitter((https://irsworkshop.github.io/2020/publications/paper_2_%20Virani_Twitter.pdf))、Pinterest((http://arxiv.org/abs/1702.07969)) など多くのプロダクトで採用されています。

# 実装

2-stage レコメンドと言っても必ずしもディープラーニングを使う必要はなく、実際Pinterestなどは異なるアプローチで取り組んでいますが、ここはGoogleに倣ってDNNでの実装を試します。
試すと言っても、[TFRSのチュートリアル群](https://www.tensorflow.org/recommenders/examples/quickstart)をほぼコピペしているだけです。詳しいことはそちらを見てください。後で書く評価の部分だけは新しいです。
コードは [Colaboratory](https://colab.research.google.com/drive/1fP43e95Uf-kQmvnJ4Mt3nWd1eYtXZ9M9?usp=sharing) にあります。

## 準備

まず必要なライブラリをインポートします。
```python
!pip install -q tensorflow-recommenders
!pip install -q --upgrade tensorflow-datasets

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
```

使うデータは Movielens 100k です。ユーザーが各映画につけた1-5点のレビューが100,000件入っています。
ここでは、1-3点のレビューを負例、4-5点のレビューを正例として扱うことにします。
また特徴量として、レビューの timestamp を利用します。
```python
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
    "user_pref": 1 if x["user_rating"]>=4 else 0,
})
movies = movies.map(lambda x: x["movie_title"])
```

データを train/test split します。
```python
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()
```

特徴量作成に必要な下準備をします。
この後で、ユーザーやアイテムの embedding を多用することになるので、そのためにユニークなIDのセットを作成します。
timestamp も単なる整数値ではなく 1000 個のバケットに分けて扱い、やはり embed します。この辺はチュートリアルそのままです。

```python
timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,
)

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))
```

## Retrieval model
まずは、candidate generation 用の retrieval model を作成します。
Retrieval model は two-tower model とも言われ、query（user_id と timestamp）と candidate (movie) それぞれで独立した、multi-layer perceptron (MLP) を作成します。
Query tower と candidate tower からそれぞれ同じ次元のベクトルを取得し、それらの similarity をベースにしたロス関数を計算します。
推論時にはロスを計算せず、query tower から得られたベクトルに類似している candidate を取得すれば良いので、近侍近傍探索などを使って高速に推論ができます。

まずクエリの情報を embed するモデルを作ります。普通の keras のモデルです。ここでは user_id と timestamp の情報を concat しています。

```python

class UserModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
        ])
        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
        ])
        self.normalized_timestamp = tf.keras.layers.experimental.preprocessing.Normalization()

        self.normalized_timestamp.adapt(timestamps)

    def call(self, inputs):
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.timestamp_embedding(inputs["timestamp"]),
            self.normalized_timestamp(inputs["timestamp"]),
        ], axis=1)
```
この例で見られるように、DNNでレコメンドモデルを作る際は、ユーザーやアイテムのIDの embedding を多用します。

上のモデルで得られた embedding ベクトルをMLPに通します。前結合レイヤーの構成は外から指定できるようにしてあります。これで query tower の完成です。

```python
class QueryModel(tf.keras.Model):

    def __init__(self, layer_sizes):
        super().__init__()

        self.embedding_model = UserModel()
        self.dense_layers = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)
```

同様のことをアイテム (candidate) 側でも行います。ここでは映画のタイトルの embed と テキストとしての vectorization を利用します。
```python
class MovieModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=unique_movie_titles,mask_token=None),
          tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
        ])

        self.title_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
          self.title_vectorizer,
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(movies)

    def call(self, titles):
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)
```

Candidate の embedding をMLPに通せば candidate tower の完成です。
```python
class CandidateModel(tf.keras.Model):

    def __init__(self, layer_sizes):
        super().__init__()

        self.embedding_model = MovieModel()
        self.dense_layers = tf.keras.Sequential()
        
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))
            
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)
```

以上二つの tower から取得したベクトルを使ってロスを計算して学習します。
```python
class MovielensRetrievalModel(tfrs.models.Model):

    def __init__(self, layer_sizes):
        super().__init__()
        self.query_model = QueryModel(layer_sizes)
        self.candidate_model = CandidateModel(layer_sizes)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "timestamp": features["timestamp"],
        })
        movie_embeddings = self.candidate_model(features["movie_title"])

        return self.task(
            query_embeddings, movie_embeddings, compute_metrics=not training)
```
ここでは `tfrs.models.Model` を継承しています。これはレコメンドモデル用に、keras のモデルを薄く拡張したものです。実際チュートリアルにもある通り、同じ処理を tfrs を使わずに書くことも簡単です。
また、ここでTFRS特有の `tfrs.tasks.Retrieval` というレイヤーが登場します。要はロス関数を定義しているわけですが、retrieval モデルを想定して、query と candidate の embedding を入力とするような形のものが使われています。Query と candidate のベクトルを concat してさらに全結合層に通したりなどはしていないことが重要です。内部では二つのベクトルの（デフォルトでは）コサイン類似度からロスが計算されます。

ここのロスの説明がチュートリアルにはないので、少し調べる必要がありました((例えば [この論文](https://dl.acm.org/doi/10.1145/3298689.3346996)の3章など))。正式名称があるのかわかりませんが in-batch softmax などど呼ばれているようです。
ミニバッチの各サンプル `(query, candidate)` について、query に対応する candidate が正解クラス、それ以外のミニバッチ内の candidate が不正解クラスになります。式で書くと
$$P(y_i|x_i) = \frac{\exp(x_i^Ty_i)}{\sum_j \exp(x_i^Ty_j)}\,,L = -\frac{1}{|B|} \sum_{i\in B} \log P(x_i|y_i)$$
となります。[ソースコード](https://github.com/tensorflow/recommenders/blob/v0.3.0/tensorflow_recommenders/tasks/retrieval.py#L89-L161)を読むと多分こんな感じに（デフォルトでは）なっています。$x_i$, $y_i$ はそれぞれ query と candidate のそれぞれの tower を通した後の embedding ベクトルです。関連のある query と candidate のコサイン類似度が高くなるように学習します。



以上のモデルを訓練します。注意点として、ここではユーザーの映画に対する rating の情報は使っていません。ユーザーがある映画を見た時点でその映画の relevance は見なかったものより高いはずです。見た後に気にいるかどうかは後続の ranking model に判断させることにして、ここではとにかくユーザーが見た映画は全てポジティブだと思うことにします。

```python
num_epochs = 100

ret_model = MovielensRetrievalModel([64, 32])
ret_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

two_layer_history = ret_model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=10,
    epochs=num_epochs,
    verbose=0,
)

accuracy = two_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")
>>> 0.28
```
トップ100のaccuracy は 0.28 になりました。100本映画を推薦したら、平均28本は実際にユーザーが見るものになります。

以上で retrieval model ができたので、近傍探索のインデックスを作成しておきましょう。これによって入力クエリの embedding ベクトルから、類似度の高い candidate を抽出することができます。

```python
cand_index = tfrs.layers.factorized_top_k.BruteForce(ret_model.query_model, k=100)
cand_index.index(movies.batch(100).map(ret_model.candidate_model), movies)
```
ここでは簡単のために `tfrs.layers.factorized_top_k.BruteForce` という厳密な近傍探索を使用しますが、実際にモデルを serving するときには高速な近似近傍探索を使うことになるはずです。

## Ranking model
続いてランキング用のモデルを定義します。ここでは retrieval model と違って、query と candidate の embedding ベクトルを concat して全結合層に通します。
Retrieval モデルでは単なるコサイン類似度の部分しか学べませんでしたが、ここではよりリッチな特徴を学習に使うことができるはずです。
また、ここでは rating を（初めに述べたように 0 or 1 に変換したもの) 学習します。

```python
class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        self.user_embeddings = UserModel()
        self.movie_embeddings = MovieModel()

        self.ratings = tf.keras.Sequential([
          tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.Dense(64, activation="relu"),
          tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):

        user_embedding = self.user_embeddings({
            "user_id": inputs["user_id"],
            "timestamp": inputs["timestamp"],
        })
        movie_embedding = self.movie_embeddings(inputs["movie_title"])

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))
```

上のモデルを元に、TFRSの ranking model を作成します。この部分は普通の binary classification です。
```python
class MovielensRankingModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model = RankingModel()
        self.task = tfrs.tasks.Ranking(
            loss = tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

    def compute_loss(self, features, training=False):
        rating_predictions = self.ranking_model({
            "user_id": features["user_id"],
            "timestamp": features["timestamp"],
            "movie_title": features["movie_title"]
        })

        return self.task(labels=features["user_pref"], predictions=rating_predictions)
```

学習をします。大体 validation セットの accuracy が 0.7 くらいになっていました。
```python
num_epochs = 100

ranking_model = MovielensRankingModel()
ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

ranking1_history = ranking_model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=10,
    epochs=num_epochs,
    verbose=1,
)
```

# 評価

以上で作成した 2-stage モデルが上手く機能していることを確かめます。以下のような手順で評価します

* `test` データセットで正例だけを取る。各 `(user_id, timestamp)` がクエリで対応する `movie_title` が ground truth
* 各クエリに対して、10件の candidate generation を行う
* それらを、ranking model を使って並び替える
* リランキング前後の推薦リストを nDCG で評価

もし serving に使うなら candidate generation は100件などもっと多くやった方が良いと思いますが、ここではあくまで評価用に10件に留めて、その10件の中で ranking model が正解をより上位に持って来れるかどうかを見ることにします。

まずは candidate generation です。ここでは事前に作成しておいた index を使って取得します。
```python
df_test = tfds.as_dataframe(test)

for x in test.batch(20_000):
    _, tmp = cand_index(x, k=10)
df_test['cand'] = tmp.numpy().tolist()
```

続いて re-ranking です。取得した candidate それぞれに対してスコアリングするために多少面倒なコードを書いています。単に ranking model でスコアを出してその順にソートしているだけです。
```python
df_test['orig_index'] = df_test.index
df_test = df_test.query('user_pref==1')

# リストの各要素を各行に
df_test_exp = df_test[['user_id', 'timestamp', 'cand', 'orig_index']].explode('cand')

# Inference
scores = ranking_model.ranking_model.predict({
    'user_id':df_test_exp.user_id,
    'timestamp': df_test_exp.timestamp,
    'movie_title': df_test_exp.cand
})
df_test_exp['ranking_score'] = scores[:, 0]

# リランキング
df_test = pd.merge(
    df_test,
    df_test_exp.groupby('orig_index').ranking_score.apply(np.array).reset_index(name='ranking_score'),
    on=['orig_index']
)
df_test['pred'] = df_test.apply(lambda x: np.array(x['cand'])[np.argsort(-x['ranking_score'])], axis=1)
```

では評価をします。nDCG は以下のような式で計算します。10件の中で、groud_truth を上に持ってこれるほどスコアが高くなります。
```python
def ndcg(pred, gt):
    try:
        idx = pred.index(gt) + 1
        return 1.0 / np.log2(idx + 1)
    except ValueError:
        return 0
```
以下結果です。

```python
df_test.apply(lambda row: ndcg(row['cand'], row['movie_title']), axis=1).mean()
>>> 0.005736147190140732
df_test.apply(lambda row: ndcg(row['pred'].tolist(), row['movie_title']), axis=1).mean()
>>> 0.007693552905167569
```
下のリランキング後の方が、nDCG が向上していることがわかります。Retrieval モデルは rating が高いか低いかという情報を学ばせていないので、当然と言えば当然なのですが、期待通りの結果が得られました。


# その他
TFRSに実装されている内容はそんなに複雑ではないので、別にわざわざライブラリとして切り出す必要もなさそうな気がします。Googleとしては、DNNを使ったレコメンドモデルのスタンダードとして普及させようという感じなのでしょうか。ドキュメントに目を通すことで、レコメンドにおけるニューラルネットワーク活用の勉強にはなると思います。
