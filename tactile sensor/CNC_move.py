import sys
import os
import time

# 親ディレクトリ (Code フォルダ) をパスに追加して FSCal モジュールを読み込めるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from FSCal.stage_ctrl import GRBLControl
except ImportError:
    print("Error: FSCal.stage_ctrl が見つかりません。")
    sys.exit(1)

def fast_move(cnc, pos_mm, feedrate=500):
    """
    指定した速度(feedrate)でCNCを移動させる関数
    pos_mm: [X, Y, Z] のリスト
    feedrate: 分速 (mm/min)。F500なら分速500mm
    """
    # FSCal.stage_ctrl.GRBLControl の仕様に合わせた符号反転
    x = -float(pos_mm[0])
    y =  float(pos_mm[1])
    z = -float(pos_mm[2])
    
    # ジョグ移動コマンド ($J=...) を作成
    # G21: 単位mm, G90: 絶対座標
    cmd = f"$J=G21G90X{x:0.3f}Y{y:0.3f}Z{z:0.3f}F{feedrate}"
    
    print(f"Moving to X:{pos_mm[0]} Y:{pos_mm[1]} Z:{pos_mm[2]} (Speed: F{feedrate})")
    return cnc.transaction(cmd)

# --- 設定 ---
PORT = "COM6" 
BAUD = 115200

# CNCの初期化
GRBL = GRBLControl(PORT, BAUD)

if GRBL.is_port_opened():
    print(f"Success: Connected to CNC on {PORT}")
    
    # 最初の一回、ロック解除が必要な場合があります
    GRBL.transaction("$X")
    
    # 現在地を 0,0,0 にリセット
    GRBL.set_origin()
    
    # テスト動作: X軸に20mm移動 (速度 F500)
    # もしこれでも遅ければ、F1000, F2000 と上げてみてください
    fast_move(GRBL, [0, 0.0, -10.0], feedrate=100)
    
else:
    print(f"Failed: {PORT} を接続してください。")
