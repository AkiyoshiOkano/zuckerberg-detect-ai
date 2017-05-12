# -*- coding: utf-8 -*-

import tensorflow as tf
import multiprocessing as mp

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from werkzeug import secure_filename
import os
import eval

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)
app.config['DEBUG'] = True
# 投稿画像の保存先
UPLOAD_FOLDER = './static/images/default'

# ルーティング。/にアクセス時
@app.route('/')
def index():
  return render_template('index.html')

# 画像投稿時のアクション
@app.route('/post', methods=['GET','POST'])
def post():
  if request.method == 'POST':
    if not request.files['file'].filename == u'':
      # アップロードされたファイルを保存
      f = request.files['file']
      img_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
      f.save(img_path)
      # eval.pyへアップロードされた画像を渡す
      result = eval.evaluation(img_path, './model2.ckpt')
    else:
      result = []
    return render_template('index.html', result=result)
  else:
  	# エラーなどでリダイレクトしたい場合
  	return redirect(url_for('index'))

if __name__ == '__main__':
  app.debug = True
  app.run(host='0.0.0.0')