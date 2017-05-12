#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import tensorflow as tf
import os
import random
import main

# OpenCVのデフォルトの顔の分類器のpath
cascade_path = './opencv/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

# 識別ラベルと各ラベル番号に対応する名前
HUMAN_NAMES = {
  0: u"ザッカーバーグ",
  1: u"イローンマスク",
  2: u"ビルゲイツ"
}

#指定した画像(img_path)を学習結果(ckpt_path)を用いて判定する
def evaluation(img_path, ckpt_path):
  # GraphのReset(らしいが、何をしているのかよくわかっていない…)
  tf.reset_default_graph()
  # ファイルを開く
  f = open(img_path, 'r')
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  # モノクロ画像に変換
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = faceCascade.detectMultiScale(gray, 1.1, 3)
  if len(face) > 0:
    for rect in face:
      # 加工した画像に何でもいいので適当な名前をつけたかった。日付秒数とかでいいかも
      random_str = str(random.random())
      # 顔部分を赤線で書こう
      cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=2)
      # 顔部分を赤線で囲った画像の保存先
      face_detect_img_path = './static/images/face_detect/' + random_str + '.jpg'
      # 顔部分を赤線で囲った画像の保存
      cv2.imwrite(face_detect_img_path, img)
      x = rect[0]
      y = rect[1]
      w = rect[2]
      h = rect[3]
      # 検出した顔を切り抜いた画像を保存
      cv2.imwrite('./static/images/cut_face/' + random_str + '.jpg', img[y:y+h, x:x+w])
      # TensorFlowへ渡す切り抜いた顔画像
      target_image_path = './static/images/cut_face/' + random_str + '.jpg'
  else:
    # 顔が見つからなければ処理終了
    print 'image:NoFace'
    return
  f.close()

  f = open(target_image_path, 'r')
  # データを入れる配列
  image = []
  img = cv2.imread(target_image_path)
  img = cv2.resize(img, (28, 28))
  # 画像情報を一列にした後、0-1のfloat値にする
  image.append(img.flatten().astype(np.float32)/255.0)
  # numpy形式に変換し、TensorFlowで処理できるようにする
  image = np.asarray(image)
  # 入力画像に対して、各ラベルの確率を出力して返す(main.pyより呼び出し)
  logits = main.inference(image, 1.0)
  # We can just use 'c.eval()' without passing 'sess'
  sess = tf.InteractiveSession()
  # restore(パラメーター読み込み)の準備
  saver = tf.train.Saver()
  # 変数の初期化
  sess.run(tf.initialize_all_variables())
  if ckpt_path:
    # 学習後のパラメーターの読み込み
    saver.restore(sess, ckpt_path)
  # sess.run(logits)と同じ
  softmax = logits.eval()
  # 判定結果
  result = softmax[0]
  # 判定結果を%にして四捨五入
  rates = [round(n * 100.0, 1) for n in result]
  humans = []
  # ラベル番号、名前、パーセンテージのHashを作成
  for index, rate in enumerate(rates):
    name = HUMAN_NAMES[index]
    humans.append({
      'label': index,
      'name': name,
      'rate': rate
    })
  # パーセンテージの高い順にソート
  rank = sorted(humans, key=lambda x: x['rate'], reverse=True)

  # 判定結果と加工した画像のpathを返す
  return [rank, face_detect_img_path, target_image_path]

# コマンドラインからのテスト用
if __name__ == '__main__':
  evaluation('testimage.jpg', './model2.ckpt')
