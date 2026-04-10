import serial as pyserial
import time
import joblib
import os
import numpy as np

class AIController:
    def __init__(self, rx_port, rx_baud_rate, tx_port, tx_baud_rate):
        self.rx_port = rx_port
        self.rx_baud_rate = rx_baud_rate
        self.tx_port = tx_port
        self.tx_baud_rate = tx_baud_rate
        self.rx = None
        self.tx = None
        
        # スクリプトがあるディレクトリの直下に 'data' フォルダを指定
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        
        # モデルの読み込み先
        self.model_path = os.path.join(data_dir, 'mlp_model.pkl')
        self.scaler_path = os.path.join(data_dir, 'scaler_X.pkl')

    def connect(self):
        try:
            # 学習済みモデルとスケーラーの読み込み
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                print(f"エラー: AIモデルが見つかりません。")
                print(f"先に 'train_mlp.py' を実行して '{self.model_path}' を作成してください。")
                return False
                
            print("AIモデルを読み込み中...")
            self.mlp = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            # シリアル接続
            self.rx = pyserial.Serial(self.rx_port, self.rx_baud_rate, timeout=0.01)
            self.tx = pyserial.Serial(self.tx_port, self.tx_baud_rate, timeout=0.01)
            
            print(f"Connected: RX={self.rx_port}, TX={self.tx_port}")
            print("AI制御モード起動完了！")
            return True
        except Exception as e:
            print(f"起動失敗: {e}")
            return False

    def run(self):
        print("-" * 30)
        print(" AIによる自動制御中... (Ctrl+C で停止)")
        print("-" * 30)
        
        try:
            while True:
                # 1. マスターからセンサー生データを受信
                line = self.rx.readline().decode('utf-8', errors='ignore').strip()
                
                if line:
                    try:
                        # 2. センサー値をリストに変換
                        sensors = [float(x) for x in line.split(',')]
                        
                        if len(sensors) >= 5:
                            # 3. AI（MLP）用にデータを整形
                            # センサーの最初の5つを使用
                            input_data = np.array([sensors[:5]])
                            
                            # 4. 学習時と同じスケーラーで正規化
                            input_scaled = self.scaler.transform(input_data)
                            
                            # 5. AIによる角度予測
                            predicted_angles = self.mlp.predict(input_scaled)[0]
                            
                            # 6. 予測値を整数にし、サーボの範囲(0-180)に収める
                            angles = np.clip(predicted_angles, 0, 180).astype(int)
                            
                            # 7. スレーブArduinoへ送信
                            send_data = ",".join(map(str, angles)) + "\n"
                            self.tx.write(send_data.encode())
                            
                            # リアルタイム表示
                            print(f"\rセンサー入力: {sensors[:5]} -> AI予測角度: {angles.tolist()}", end="", flush=True)
                            
                    except ValueError:
                        pass # 変換エラーは無視
                    except Exception as e:
                        print(f"\n推論エラー: {e}")
                        
        except KeyboardInterrupt:
            print("\nAI制御を終了します。")
        finally:
            if self.rx: self.rx.close()
            if self.tx: self.tx.close()
            print("Serial port closed.")

if __name__ == "__main__":
    # COMポートは環境に合わせて設定してください
    controller = AIController("COM3", 9600, "COM4", 9600)
    
    if controller.connect():
        controller.run()
