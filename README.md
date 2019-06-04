# TensorFlowチュートリアル 映画レビューのテキスト分類

https://www.tensorflow.org/tutorials/keras/basic_text_classification?hl=ja

```
venv/lib/python3.6/site-packages/tensorflow/python/keras/datasets/imdb.py
```

allow_pickle=Trueを追加
```
 85   with np.load(path, allow_pickle=True) as f:
 86     x_train, labels_train = f['x_train'], f['y_train']
 87     x_test, labels_test = f['x_test'], f['y_test']
```
