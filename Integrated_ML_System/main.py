import sys
import os

def menu(dummy_mode):
    print("\n" + "="*40)
    print("  5本指ロボットハンド 学習システム" + (" [DUMMY MODE]" if dummy_mode else ""))
    print("="*40)
    print("1. データ収集 (模倣学習用)")
    print("2. 模倣学習 (BC) のトレーニング")
    print("3. 強化学習 (PPO) でのファインチューン")
    print("4. 強化学習 (SAC) でのファインチューン")
    print("5. モデルの評価 (BC / RL)")
    print("d. ダミーモード切替 (現在: " + ("ON" if dummy_mode else "OFF") + ")")
    print("q. 終了")
    print("-"*40)
    
    choice = input("選択してください: ").lower()
    return choice

def main():
    dummy_mode = False
    while True:
        choice = menu(dummy_mode)
        
        args = " --dummy" if dummy_mode else ""
        
        if choice == '1':
            os.system(f"{sys.executable} scripts/collect_data.py{args}")
        elif choice == '2':
            os.system(f"{sys.executable} scripts/train_bc.py") 
        elif choice == '3':
            os.system(f"{sys.executable} scripts/train_rl.py{args}")
        elif choice == '4':
            os.system(f"{sys.executable} scripts/train_rl_sac.py{args}")
        elif choice == '5':
            m = input("評価対象 (bc / rl_ppo / rl_sac): ").lower()
            os.system(f"{sys.executable} scripts/evaluate.py{args} --mode {m}")
        elif choice == 'd':
            dummy_mode = not dummy_mode
            print(f"ダミーモードを {'ON' if dummy_mode else 'OFF'} にしました。")
        elif choice == 'q':
            print("終了します。")
            break
        else:
            print("無効な選択です。")

if __name__ == "__main__":
    # カレントディレクトリをプロジェクトルートに設定
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
