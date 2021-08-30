from collections import defaultdict

import numpy as np
from keras import callbacks
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib



class LearningVisualizationCallback(callbacks.Callback):
    """学習曲線を可視化するためのコールバッククラス"""

    def __init__(self, higher_better_metrics, fig=None, ax=None):
        self._metric_histories = defaultdict(list)
        self._metric_history_lines = {}
        self._higher_better_metrics = set(higher_better_metrics)
        self._metric_type_higher_better = defaultdict(bool)
        self._best_score_vlines = {}
        self._best_score_texts = {}

        # 描画領域を初期化する
        self._fig = fig
        self._ax = ax
        if self._fig is None and self._ax is None:
            self._fig, self._ax = plt.subplots()
        self._ax.set_title('learning curve')
        self._ax.set_xlabel('epoch')
        self._ax.set_ylabel('score')
        self._fig.canvas.draw()
        self._fig.show()

    def on_epoch_end(self, epoch, logs=None):
        """各エポック毎に呼ばれるコールバック"""

        # 各メトリックのスコアを保存する
        for metric, score in logs.items():
            self._metric_histories[metric].append(score)

            # 初回だけの設定
            if epoch == 0:
                # メトリックの種別を保存する
                for higher_better_metric in self._higher_better_metrics:
                    if higher_better_metric in metric:
                        self._metric_type_higher_better[metric] = True
                        break
                # スコアの履歴を描画するオブジェクトを生成する
                history_line, = self._ax.plot([], [])
                self._metric_history_lines[metric] = history_line
                history_line.set_label(metric)
                if 'val' not in metric:
                    # 学習データのメトリックは検証データに比べると重要度が落ちるので点線
                    history_line.set_linestyle('--')
                else:
                    # ベストスコアの線を描画するオブジェクトを生成する
                    best_vline = self._ax.axvline(0)
                    best_vline.set_color(history_line.get_color())
                    best_vline.set_linestyle(':')
                    self._best_score_vlines[metric] = best_vline
                    # ベストスコアの文字列を描画するオブジェクトを生成する
                    vpos = 'top' if self._metric_type_higher_better[metric] else 'bottom'
                    best_text = self._ax.text(0, 0, '',
                                              va=vpos, ha='right', weight='bold')
                    best_text.set_color(history_line.get_color())
                    self._best_score_texts[metric] = best_text

        # 描画内容を更新する
        for metric, scores in self._metric_histories.items():
            # グラフデータを更新する
            history_line = self._metric_history_lines[metric]
            history_line.set_data(np.arange(len(scores)), scores)
            if 'val' in metric:
                if self._metric_type_higher_better[metric]:
                    best_score_find_func = np.max
                    best_epoch_find_func = np.argmax
                else:
                    best_score_find_func = np.min
                    best_epoch_find_func = np.argmin
                best_score = best_score_find_func(scores)
                # 縦線
                best_epoch = best_epoch_find_func(scores)
                best_vline = self._best_score_vlines[metric]
                best_vline.set_xdata(best_epoch)
                # テキスト
                best_text = self._best_score_texts[metric]
                best_text.set_text('epoch:{}, score:{:.6f}'.format(best_epoch, best_score))
                best_text.set_x(best_epoch)
                best_text.set_y(best_score)

        # グラフの見栄えを調整する
        self._ax.legend()
        self._ax.relim()
        self._ax.autoscale_view()

        # 再描画する
        plt.pause(0.001)

    def show_until_close(self):
        """ウィンドウを閉じるまで表示し続けるためのメソッド"""
        plt.show()
