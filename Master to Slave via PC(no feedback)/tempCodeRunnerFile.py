import serial as pyserial
import time

class SerialCommunicator:

    def __init__(self, rx_port, rx_baud_rate, tx_port, tx_baud_rate):
        self.rx_port = rx_port
        self.rx_baud_rate = rx_baud_rate
        self.tx_port = tx_port
        self.tx_baud_rate = tx_baud_rate
        self.rx = None
        self.tx = None
        self.finger_data = None

    def connect(self):
        try:
            self.rx = pyserial.Serial(self.rx_port, self.rx_baud_rate, timeout=1)
            self.tx = pyserial.Serial(self.tx_port, self.tx_baud_rate, timeout=1)
            print(f"Connected to {self.rx_port} as rx, Connected to {self.tx_port} as tx")
        except Exception as e:
            print(f"接続失敗: {e}")

    def disconnect(self):
        if self.rx and self.rx.is_open:
            self.rx.close()
            print("Serial port closed.")

    def arduino_map(self, x, master_min, master_max, slave_min, slave_max):
        #比例関係に落とし込む
        """
        master_min, maxはマスターデバイスの可変抵抗値の最小値、最大値
        slave_min, maxはスレーブデバイスのサーボモータの最小値、最大値
        """
        
        val = (x - master_min) * (slave_max - slave_min) / (master_max - master_min) + slave_min
        
        # 範囲内に収める（クランプ処理：サーボの保護）
        if slave_min < slave_max:
            return max(slave_min, min(slave_max, val))
        else:
            return max(slave_max, min(slave_min, val))

    def communicate(self):
        if self.rx is None:
            print("未接続です")
            return

        try:
            while True:
                # マスターからの生データ受信 (例: "500,600,700,800,900")
                line = self.rx.readline().decode('utf-8').strip()

                if line:
                    try:
                        self.finger_data = [float(x) for x in line.split(',')]

                        if len(self.finger_data) >= 5:
                            # Python側で角度計算（10〜170の範囲）を行う
                            angles=[
                                int(self.arduino_map(self.finger_data[0],400,700,10,170)),
                                int(180-self.arduino_map(self.finger_data[1],400,600,10,180)),
                                int(180-self.arduino_map(self.finger_data[2],450,600,10,180)),
                                int(180-self.arduino_map(self.finger_data[3],300,500,10,180)),
                                int(180-self.arduino_map(self.finger_data[4],400,500,10,180))
                            ]
                            

                            # 計算済みの角度をスレーブArduinoへ送信
                            send_data = ",".join(map(str, angles)) + "\n"
                            self.tx.write(send_data.encode())
                            print(f"角度送信中: {send_data.strip()}")

                        else:
                            print(f"受信データ不備: {self.finger_data}")

                    except ValueError:
                        print(f"スキップ: {line}")

        except KeyboardInterrupt:
            print("\n停止...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.disconnect()
            


if __name__ == "__main__":
    # 環境に合わせてCOMポートを変更してください
    master = SerialCommunicator("COM5", 9600, "COM4", 9600)
    master.connect()
    master.communicate()
