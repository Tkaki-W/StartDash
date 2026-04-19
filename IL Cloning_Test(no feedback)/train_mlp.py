import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import glob
import os

def train():
    # スクリプトがあるディレクトリの直下に 'data' フォルダを指定
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # 1. 'data/' ディレクトリ内の最新のCSVファイルを探す
    list_of_files = glob.glob(os.path.join(data_dir, 'training_data_*.csv'))
    if not list_of_files:
        print(f"エラー: '{data_dir}/training_data_*.csv' が見つかりません。")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"学習に使用するファイル: {latest_file}")

    # 2. データの読み込み
    try:
        df = pd.read_csv(latest_file)
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return

    # 入力(X): センサー値 s0~s4, 出力(y): 角度データ a0~a4
    X = df[['s0', 's1', 's2', 's3', 's4']]
    y = df[['a0', 'a1', 'a2', 'a3', 'a4']]

    print(f"総データ数: {len(df)} サンプル")

    # 3. データの正規化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # 4. 分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. MLPモデルの設定
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 100), 
        activation='relu', 
        solver='adam', 
        max_iter=1000, 
        verbose=True, 
        random_state=42
    )

    # 6. 学習実行
    print("\n--- 学習開始 ---")
    mlp.fit(X_train, y_train)
    print("--- 学習終了 ---\n")

    # 7. 評価
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test, y_test)
    print(f"学習データでのスコア (R^2): {train_score:.4f}")
    print(f"テストデータでのスコア (R^2): {test_score:.4f}")

    # 8. モデルとスケーラーを 'data/' 内に保存
    model_path = os.path.join(data_dir, 'mlp_model.pkl')
    scaler_path = os.path.join(data_dir, 'scaler_X.pkl')
    joblib.dump(mlp, model_path)
    joblib.dump(scaler_X, scaler_path)
    print(f"\nモデルとスケーラーを保存しました: '{model_path}', '{scaler_path}'")

    # 9. 学習曲線の表示 (ロスが下がっていく様子を確認)
    plt.figure(figsize=(8, 5))
    plt.plot(mlp.loss_curve_)
    plt.title("Learning Curve (Loss Over Time)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
