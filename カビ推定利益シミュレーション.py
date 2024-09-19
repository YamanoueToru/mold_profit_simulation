import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

st.title('カビ推定純利益シミュレーション')

st.write('パラメータを入力してください')
with st.form("input_form"):
    x = st.number_input('分け前 (小数)', min_value=0.0, max_value=1.0, value=0.1, step=0.01)  # 分け前
    N = st.number_input('IoT接続機器台数', min_value=0, max_value=10000000, value=700000, step=1000)  # IoT接続台数
    t = st.number_input('カビ有り比率', min_value=0.0, max_value=1.0, value=0.5, step=0.01)  # カビ有り比率
    y1 = st.number_input('カビあり通知を受けて、エアクリしてもらった人の割合', min_value=0.0, max_value=1.0, value=0.1, step=0.01)  # エアクリ通知を受けて、
    y0 = st.number_input('カビあり通知を受けてエアクリ依頼したけどカビはなかった人の割合', min_value=0.0, max_value=1.0, value=0.05, step=0.01)  # カビ有り通知してカビはなかったけど
    a = st.number_input('サーバ・アルゴ運用費', min_value=0, max_value=100000000, value=1000000, step=100000)  # サーバ・アルゴ運用費
    b = st.number_input('エアクリ費用(出張費等)', min_value=0, max_value=100000, value=15000, step=1000)  # エアクリ費用
    c = st.number_input('エアクリ単価', min_value=0, max_value=100000, value=25000, step=1000)  # エアクリ単価

    submit_button = st.form_submit_button(label='シミュレーション開始')

if submit_button:
    recall_values = np.linspace(0.1, 0.9, 9)

    # 利益関数の定義
    def profit(precision, recall):
        return N * t * recall * y1 * (c - b) * x + N * t * recall * ((1 - precision) / precision) * y0 * (-b) - a

    # 利益の計算
    results = np.zeros((999, len(recall_values)))
    precisions = np.linspace(0.001, 1.0, 999)

    for j, recall_val in enumerate(recall_values):
        for i, precision_val in enumerate(precisions):
            if precision_val != 0:  # ゼロ除算を防ぐため
                results[i, j] = profit(precision_val, recall_val)

    # 利益関数をプロット
    plt.figure(figsize=(10, 6))
    for i, recall_val in enumerate(recall_values):
        plt.plot(precisions, results[:, i], label=f'recall={recall_val:.1f}')
    plt.legend()
    plt.xlabel('Precision')
    plt.ylabel('利益')
    plt.ylim(bottom=0, top=3e7)
    plt.title('Precision vs Profit for different Recall values')

    st.pyplot(plt)

    # 利益が0になるprecisionを求める
    zero_precisions = []

    for recall_val in recall_values:
        precision_zero = fsolve(lambda p: profit(p, recall_val), 0.5)  # 初期解を0.5に設定
        zero_precisions.append(precision_zero[0])

    # 結果の表示
    result_df = pd.DataFrame({
        'Recall': recall_values,
        'プラス転換するprecisionの値': zero_precisions
    })
    st.write(result_df)
