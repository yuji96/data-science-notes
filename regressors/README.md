# regressors

sklearn で提供されている分類モデル比較表の回帰モデルバージョンを作りたかったというのが発端。  
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

## 概要
人工データに対して様々なアルゴリズムで回帰した結果の図を見て、それぞれの特徴を考察していく。
厳密に理論と照らし合わせると教科書みたいなボリュームになってしまうので、ちらっとだけ触れる。


## notebook

### 線形回帰
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuji96/data-science-notes/blob/main/regressors/notebooks/1_linear.ipynb)  
グラフが直線という意味ではなく、基底関数が線形結合しているという意味での「線形」。

### ベイズ理論
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuji96/data-science-notes/blob/main/regressors/notebooks/2_bayes.ipynb)

### ニューラルネットワーク
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuji96/data-science-notes/blob/main/regressors/notebooks/3_neural-network.ipynb)

### 決定木ベース
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yuji96/data-science-notes/blob/main/regressors/notebooks/4_tree.ipynb)

### その他
実装から省いたモデル。

- K近傍法

    得意なこと: 欠損値置換。回帰アルゴリズムの中には、入力変数によっては奇想天外な値を返してしまうことがある。
    ただ、K近傍法は実際に存在する値の平均値を返すので置換処理によって分布が崩れにくいらしい。回帰線は決定木みたいなジグザグした線を描く。
- スプライン回帰

    スプライン補完という名前の方がよく聞くかもしれない。点と点どうしを繋げてなめらかな曲線を作るときによく使われる。
    デザインツールとかで使われてそう。回帰曲線が全ての点を通るというのが、他のアルゴリズムと違う特徴で、スプライン回帰の目的でもある。

## 流れ
以下の手順を繰り返す。
- `model.fit(x, y)` でモデルを学習する。
- `model.predict(x)` で回帰式を描く。
- （model の属性に代入されている学習結果を使って、自分で回帰式を描く。）
-  図を見て自分なりの解釈を述べる。

## 備考
- パラメータチューニングは目視です。最適化はサボりました。
- 数式中で、ベクトルもスカラーも同じ字体で書いています。`boldsybmol` を毎回書くの大変なのでサボりました。添字の有無とかで判断していただけたらと思います。
- 補足説明は偏りがちの個人の解釈です。
