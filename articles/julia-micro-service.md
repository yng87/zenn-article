---
title: "Juliaでマイクロサービスを作る"
emoji: "📖" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["Julia"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

# 概要
JuliaCon2020 の ["Building microservices and applications in Julia"](https://live.juliacon.org/talk/Y3H7FG) というワークショップのトークを見て、Juliaを使ってマイクロサービスを作る方法を勉強したのでメモ的に記事を書きます。いわゆる「チュートリアルやってみた」記事です。

ワークショップでは音楽アルバム管理アプリを例として作っていましたが、ここではそれとほぼ同じようにして簡易的な論文管理アプリを作ろうと思います。マイクロサービスなので、フロントを作ったりはせず、HTTPでJSONをやりとりするものになります。

Julia を実際のサービスで使っているというような例は全然聞きませんが、昨今のトレンドらしいマイクロサービス化の波に乗って使われる例が増えると良いなと思っています。

# やること

- Julia で論文リストを保存し管理するアプリケーションを作る
- Julia の `HTTP.jl` を使って、そのアプリケーションにアクセスするAPIエンドポイントを設定する
- エンドポイントを公開する

最初に紹介したトークをなぞっているだけなので、ちゃんと知りたい人はそちらを見てください。[HTTP.jlのドキュメント](https://juliaweb.github.io/HTTP.jl/stable/examples/)も参考になります。

実装するアーキテクチャは以下のような感じです。
![](https://storage.googleapis.com/zenn-user-upload/u1t8vdeuz9bwttzlqb5j7k069d9i)*"Building microservices and applications in Julia"より*

この図によると、アプリケーションは大別して3層からなります。
- Resource layer: 外部からのアクセスに対して、HTTPのルーティングを行います
- Service layer: Resource layer からの呼び出しに応じてアプリケーションのロジックを実行します
- Repositry/Mapper layer: 扱うオブジェクトをストレージに保存したり、ストレージから読んだり削除したりします

これらの層で共通に使われるのが Model objects です。アプリケーションが扱うオブジェクトを Julia の構造体として定義します。

# 実装
上のアーキテクチャに従って実装します。

## Papers.jl
`Papers.jl` はメインのモジュールです。ライブラリの使用者はこのモジュールを `using Papers` でインポートして使います。よくある Julia のライブラリの書き方です。
ここでは4つのサブモジュールを読み込んでいます。上のアーキテクチャ図と見比べてください。これからこの4つをそれぞれ実装していきます。

```julia:Papers.jl
module Papers

export Model, Mapper, Resource, Service

include("Model.jl")
using .Model

include("Mapper.jl")
using .Mapper

include("Service.jl")
using .Service

include("Resource.jl")
using .Resource

end
```

## Model.jl
論文を管理するアプリケーションを作るので、まず一つ一つの論文情報を扱う構造体を作ります。ここでは簡単のために、論文が持つ情報は著者・タイトル・アブストラクト・出版日だとしておきます。

```julia:Model.jl
module Model
import Base: ==
using StructTypes, Dates

export Paper

mutable struct Paper
    id::Int64 # service-managed
    title::String
    abst::String
    authors::Vector{String}
    publishdate::Date
    timespicked::Int64 # service-managed
end

==(x::Paper, y::Paper) = x.id == y.id
Paper() = Paper(0, "", "", String[], Date("1900-01-01", "y-m-d"), 0)
Paper(title, abst, authors, publishdate) = Paper(0, title, abst, authors, Date(publishdate, "y-m-d"), 0)

StructTypes.StructType(::Type{Paper}) = StructTypes.Mutable()

end
```
`id` と `timespicked` はサービスが管理する情報です。`timespicked` については後で説明します。

ここでは `StructTypes.jl` というライブラリを用いて、Juliaの純正 struct を拡張しています。後でJSONライブラリと合わせて構造体の serialization/deserialization を行うのに必要です。

## Mapper.jl
続いて論文を集めて管理するデータベースとそれに対する操作を作ります。
```julia:Mapper.jl
module Mapper

using ..Model

const STORE = Dict{Int64,Paper}()
const COUNTER = Ref{Int64}(0)

function store!(paper)
    if haskey(STORE, paper.id)
        # updating
        STORE[paper.id] = paper
    else
        # creating new
        paper.id = COUNTER[] += 1
        STORE[paper.id] = paper
    end
    return
end

function get(id)
    return STORE[id]
end

function delete(id)
    delete!(STORE, id)
    return
end

function getAllPapers()
    return collect(values(STORE))
end

end
```

* `STORE` を論文データベースとします。ここではオンメモリの辞書で全てのデータを持つことにします。もちろん本当にサービスに組み込むならこんなやり方ではダメですが、今回はあくまでデモなので簡単のためにこうします。最初に紹介したトークでは、きちんとデータベースを使うやり方も紹介されています。
* `store!`, `get`, `delete`, `getAllPapers` はデータベースへアクセスする関数です。論文の `id` はグローバルな `COUNTER` で付与します。`getAllPapers` はデータベースの全論文を取得する関数です。

## Service.jl
続いて、`Mapper.jl` のアトミックな関数を使って、アプリケーションのロジックを実装します。データベースに対する新しいデータの保存、既存のデータの更新・削除を実装します。また読むべき論文を一つ選び出す関数も実装します。

```julia:Service.jl
module Service

using ..Model, ..Mapper

function createPaper(obj)
    # 引数として json オブジェクトを受け取る。
    # 必要なフィールドが入力に格納されているかをチェックし、
    # 問題なければ新しい `Paper` オブジェクトを作成してデータベースへ保存。
    @assert haskey(obj, :title) && !isempty(obj.title)
    @assert haskey(obj, :abst) && !isempty(obj.abst)
    @assert haskey(obj, :authors) && !isempty(obj.authors)
    @assert haskey(obj, :publishdate) && !isempty(obj.publishdate)
    paper = Paper(obj.title, obj.abst, obj.authors, obj.publishdate)
    Mapper.store!(paper)
    return paper
end

# データベースから論文のオブジェクトを取得
getPaper(id) = Mapper.get(id)

# 登録済みの論文オブジェクトをアップデート
function updatePaper(id, updated)
    paper = Mapper.get(id)
    paper.title = updated.title
    paper.abst = updated.abst
    paper.authors = updated.authors
    paper.publishdate = updated.publishdate
    Mapper.store!(paper)
    return paper
end

# データベースから論文を削除
function deletePaper(id)
    Mapper.delete(id)
    return
end

function pickPaperToRead()
    papers = Mapper.getAllPapers()
    leastTimesPicked = minimum(x -> x.timespicked, papers)
    leastPickedPapers = filter(x -> x.timespicked == leastTimesPicked, papers)
    pickedPaper = rand(leastPickedPapers)
    pickedPaper.timespicked += 1
    Mapper.store!(pickedPaper)
    return pickedPaper
end

end
```
`pickPaperToRead`は読むべき論文を一つピックアップする関数です。`Paper` オブジェクトには `timespicked` という情報が入っていますが、これはその論文が今までピックアップされた回数です。ここではピックアップされた回数が最も少ない論文をデータベースから取得し、(複数あればランダムに一つ選んで) 返します。

## Resource.jl
いよいよAPIエンドポイントを作成します。Julia の `HTTP.jl` を利用して、`Service.jl` の関数を呼び出すエンドポイントを以下のように実装します。
```julia:Resource.jl
module Resource
using HTTP, JSON3, Sockets
using ..Model, ..Service

const ROUTER = HTTP.Router()

health(req) = Dict("status"=>"ok")
HTTP.@register(ROUTER, "GET", "/", health)

createPaper(req) = Service.createPaper(JSON3.read(req.body))::Paper
HTTP.@register(ROUTER, "POST", "/paper", createPaper)

getPaper(req) = Service.getPaper(
    parse(Int, HTTP.URIs.splitpath(req.target)[2]) #/paper/1 から1を取ってくる
)::Paper
HTTP.@register(ROUTER, "GET", "/paper/*", getPaper)

updatePaper(req) = Service.updatePaper(
    parse(Int, HTTP.URIs.splitpath(req.target)[2]),
    JSON3.read(req.body, Paper)
)::Paper
HTTP.@register(ROUTER, "PUT", "/paper/*", updatePaper)

deletePaper(req) = Service.deletePaper(
    parse(Int, HTTP.URIs.splitpath(req.target)[2])
)
HTTP.@register(ROUTER, "DELETE", "/paper/*", deletePaper)

pickPaperToRead(req) = Service.pickPaperToRead()::Paper
HTTP.@register(ROUTER, "GET", "/paper", pickPaperToRead)
```
まず重要なのは`ROUTER`です。これはHTTPリクエストのルーティングを行うオブジェクトです。これを使ってエンドポイントを登録していきます。そのためのマクロが `HTTP.@register` です。メソッド名とリソース名、そして呼び出す関数を登録します。これで例えばルーターが `/paper` への POST を受け取ると、createPaper 関数が呼び出されます。

ここでは同名の関数が出てきて confusing ですが、例えば Resource モジュール内の `createPaper` 関数は、HTTPリクエストを読んで、ロジックが実装されている Service モジュールの `createPaper` 関数に投げるという働きをします。

最後にHTTPリクエストのハンドラーとサービング用の関数を実装します。以下のような感じです。

```julia:Resource.jl
function requestHandler(req)
    obj = HTTP.handle(ROUTER, req)
    resp = HTTP.Response(200, JSON3.write(obj))
    return resp
end

function run()
    HTTP.serve(requestHandler, ip"0.0.0.0", 8080)
end

end # module
```

# エンドポイントを公開する
以上で作成したアプリケーションを実際にデプロイして実行テストをしてみます。

## デプロイ
アプリケーション実行用に、以下のようなスクリプトを書き、`run.jl` として保存しました。
```julia:run.jl
using Papers
Resource.run()
```
そして次のように Dockerfile を書きます。ディレクトリの構成は[リポジトリ](https://github.com/yng87/Papers)を見てください。
```dockerfile:Dockerfile
FROM julia:1.5.3

RUN mkdir /app
COPY ./ /app/

WORKDIR /app

RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

CMD ["julia", "--project=.", "run.jl"]
```
これを docker build してイメージを docker hub に push しました。

今回はただのデモで運用等は考えていないので、何も難しいことは考えずAWSの Elastic Beanstalk でデプロイしました。以下のような `docker-compose.yml` を書いてアップロードすれば勝手に実行してくれます。簡単でした。
```yaml:docker-compose.yml
version: "3"
services:
  paper:
    image: docker/hub:latest # docker hub のイメージ
    ports:
      - 80:8080
```

## 実行結果
ローカルからリクエストを投げてみます。

ヘルスチェック
```bash
$ curl -X GET http://papers-env.xxx
{"status":"ok"}
```

論文情報の登録
```bash
$ curl -X POST -d '{"title":"A Supersymmetry Primer","authors":["Stephen P. Martin"],"abst":"I provide a pedagogical introduction to supersymmetry...", "publishdate":"1997-09-16"}' http://papers-env.xxx/paper
{"id":1,"title":"A Supersymmetry Primer","abst":"I provide a pedagogical introduction to supersymmetry...","authors":["Stephen P. Martin"],"publishdate":"1997-09-16","timespicked":0}

$ curl -X POST -d '{"title":"Cooling Theory Faced with Old Warm Neutron Stars", "authors":["Keisuke Yanagi", "Natsumi Nagata", "Koichi Hamaguchi"], "abst":"Recent observations have found several candidates...", "publishdate":"2019-04-08"}' http://papers-env.xxx/paper
{"id":2,"title":"Cooling Theory Faced with Old Warm Neutron Stars","abst":"Recent observations have found several candidates...","authors":["Keisuke Yanagi","Natsumi Nagata","Koichi Hamaguchi"],"publishdate":"2019-04-08","timespicked":0}
```

読むべき論文を一つピックアップ
```bash
$ curl -X GET http://papers-env.xxx/paper
{"id":2,"title":"Cooling Theory Faced with Old Warm Neutron Stars","abst":"Recent observations have found several candidates...","authors":["Keisuke Yanagi","Natsumi Nagata","Koichi Hamaguchi"],"publishdate":"2019-04-08","timespicked":1}

$ curl -X GET http://papers-env.xxx/paper
{"id":1,"title":"A Supersymmetry Primer","abst":"I provide a pedagogical introduction to supersymmetry...","authors":["Stephen P. Martin"],"publishdate":"1997-09-16","timespicked":1}
```

上手く動いていることが確認できました。

# 感想
上で作ったのは本当におもちゃのようなデモですが、ワークショップではこのような実装をベースにしてデータベースと連携させたり、複数人で使えるように認証システムを作ったりということまでしています。

そもそも僕はWebアプリの開発自体が素人なので、そういう意味でも勉強になるワークショップでした。Julia は今まで科学技術計算にしか使ったことがありませんでしたが、このように簡単にAPIを作れるようになっていてだいぶエコシステムが充実してきている印象です。

Julia には [Genie.jl](https://github.com/GenieFramework/Genie.jl) というWebフレームワークもあります。使ったことはないのですが、単体で完結したWebサービスを作るにはこのようなフレームワークも使えそうです。

Python で作成した機械学習モデルを Flask を利用してAPIとしてサービスに組み込むというのはよくあるパターンだと思いますが、Juliaでも同じようなことができるのではないでしょうか？Python だと色々と制約がきつかったりする部分を Julia で楽に作ってそのまま実用化できたら良いなと思っています。