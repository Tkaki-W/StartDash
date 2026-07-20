import sys
import os
import time
import numpy as np
from collections import deque

fscal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FSCal'))
sys.path.append(fscal_path)

try:
    from FS_MMS101 import MMS101ForceSensor
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

def main():
    PORT = "COM7"
    OUTPUT_6AXIS = True 
    
    print(f"Initializing MMS101 on {PORT}...")
    try:
        fs = MMS101ForceSensor(PORT, baudrate=1000000, output6=OUTPUT_6AXIS)
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    if not fs.is_opened():
        print("センサーのオープンに失敗しました。")
        return

    print("Starting continuous read...")
    fs.start_continuous_read()
    
    time.sleep(2)
    print("Setting zero point...")
    fs.set_zero()
    
    print("-" * 70)
    print(" Reading Data with Contact Detection")
    print(" [Status]  Fz (Current)  Fz (1s ago)  Diff")
    print("-" * 70)

    # 履歴を保存するデック (timestamp, fz)
    history = deque()
    contact_status = "非接触"
    
    total_samples = 0
    try:
        while True:
            data_list = fs.get_data()
            if data_list:
                now = time.time()
                for new_data, ts in data_list:
                    total_samples += len(new_data)
                    avg_force = np.mean(new_data, axis=0)
                    current_fz = avg_force[2] # Fzはインデックス2

                    # 履歴に追加
                    history.append((now, current_fz))

                    # 1秒より古いデータを削除
                    while history and history[0][0] < now - 1.0:
                        past_time, past_fz = history.popleft()
                        
                        # 1秒前のデータと比較
                        # (popした直後のデータが一番「1秒前」に近い)
                        diff = current_fz - past_fz

                        # 判定ロジック
                        # 1秒前と比べて -0.3 小さくなったら接触
                        if diff <= -0.3:
                            contact_status = "接触"
                        # 大きくなったら非接触
                        elif diff > 0.6:
                            contact_status = "非接触"

                    # 表示の更新
                    # 1秒前の値がまだない場合は 0.0 とする
                    ref_fz = history[0][1] if history else current_fz
                    display_diff = current_fz - ref_fz
                    
                    sys.stdout.write(f"\r [{contact_status}]  Fz: {current_fz:6.2f}  Ref: {ref_fz:6.2f}  Diff: {display_diff:6.2f}")
                    sys.stdout.flush()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        fs.stop_continuous_read()
        print("Done.")

if __name__ == "__main__":
    main()
