# MNIST Pytorch 練習教材

MNISTを通してPytorchと機械学習の理解を深める目的です。

## 環境構築

```bash
conda create -n MNIST-practice python=3.9
```

## ライブラリのインストール

```bash
pip install -r requirements.txt
```

## 実行

```bash
python3 train.py
```
## Note

- CUDA(GPU)がないと動きません。
  - `device = 'cpu'`のコメントアウトを外してください。
- MNISTのデータセットは自動でダウンロードされるので必要ありません。
