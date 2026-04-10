#include <Servo.h>

Servo servos[6]; // 5本の指
int servoPins[6] = {7, 6, 5, 4, 3, 2}; // 指のピン番号

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(10);

  for (int i = 0; i < 5; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(90); // 初期位置
  }
}

void loop() {
  if (Serial.available() > 0) {
    // Python側で計算済みの角度（10〜170程度）を受信
    int v1 = Serial.parseInt();
    int v2 = Serial.parseInt();
    int v3 = Serial.parseInt();
    int v4 = Serial.parseInt();
    int v5 = Serial.parseInt();

    // 受信バッファの残りをクリア
    while (Serial.available() > 0) Serial.read();

    // 受信した角度をそのままサーボに適用
    servos[0].write(v1);
    servos[1].write(v2);
    servos[2].write(v3);
    servos[3].write(v4);
    servos[4].write(v5);
  }
}
