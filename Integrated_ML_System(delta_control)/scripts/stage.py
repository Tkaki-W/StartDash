import sys
import os
import time
import msvcrt

# このファイル(stage.py)から2階層上 = Code/ を計算
this_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.abspath(os.path.join(this_dir, "..", ".."))
sys.path.append(code_dir)

from FSCal.stage_ctrl import StageControl

PORT = "COM6"
BAUD = 115200

# GRBLステージの初期化
stage_ctrl = StageControl(PORT, baudrate=BAUD, stage_type='GRBL')

# ポートが開けているか確認
if not stage_ctrl.is_port_opened():
    print("Error: ステージに接続できませんでした。")
    sys.exit(1)

# GRBLのロック解除
stage_ctrl.stage.transaction("$X")
stage_ctrl.stage.set_origin()
print("GRBL初期化完了 (原点設定済み)\n")

print("=" * 70)
print("  キーボード操作でステージを動かします")
print("=" * 70)
print("  [W/S] X軸移動  [A/D] Y軸移動  [Q/E] Z軸移動")
print("  [ESC] 終了")
print("-" * 70)

# 現在の目標座標
target_pos = [0.0, 0.0, 0.0]
step_size = 1.0  # 1mmずつ移動

try:
    while True:
        # キー入力チェック
        if msvcrt.kbhit():
            key = msvcrt.getch()

            if key == b'\x1b':  # ESC
                print("\n\n終了します。")
                break
            elif key == b'w': target_pos[0] += step_size
            elif key == b's': target_pos[0] -= step_size
            elif key == b'a': target_pos[1] -= step_size
            elif key == b'd': target_pos[1] += step_size
            elif key == b'q': target_pos[2] += step_size
            elif key == b'e': target_pos[2] -= step_size
            else:
                continue

            # ステージを移動
            stage_ctrl.stage.move_to(target_pos)

        # 現在位置を表示
        pos, t = stage_ctrl.stage.get_current_pos()
        x, y, z = pos
        print(f'\r目標: X={target_pos[0]:7.2f} Y={target_pos[1]:7.2f} Z={target_pos[2]:7.2f} | '
              f'現在: X={x:7.3f} Y={y:7.3f} Z={z:7.3f} mm', end='', flush=True)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n中断されました。")