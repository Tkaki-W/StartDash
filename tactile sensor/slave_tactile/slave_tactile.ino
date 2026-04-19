#include <Servo.h>

Servo servos[5]; // 5本の指
int servoPins[5] = {7, 6, 5, 4, 3}; // 指のピン番号 (必要に応じて調整してください)
const int tactilePin = A0; // 触覚センサーをA0に接続

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(10); // 短いタイムアウトでレスポンスを向上

  for (int i = 0; i < 5; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(90); // 初期位置
  }
}

void loop() {
  if (Serial.available() > 0) {
    // 1. PCから角度データを受信 (カンマ区切り)
    int v1 = Serial.parseInt();
    int v2 = Serial.parseInt();
    int v3 = Serial.parseInt();
    int v4 = Serial.parseInt();
    int v5 = Serial.parseInt();

    // 受信バッファの残りをクリア
    while (Serial.available() > 0) Serial.read();

    // 2. サーボを動かす
    servos[0].write(v1);
    servos[1].write(v2);
    servos[2].write(v3);
    servos[3].write(v4);
    servos[4].write(v5);

    // 3. 触覚センサー値を読み取り、PCへ返信 (フィードバック)
    // 次のマスターデータ処理に間に合うよう、即座に送信
    int tactileValue = analogRead(tactilePin);
    Serial.println(tactileValue); 
  }
}
