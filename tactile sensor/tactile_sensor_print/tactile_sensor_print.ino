void setup() {
  Serial.begin(9600);
}
void loop() {
  float sensorValue = analogRead(A0);
  Serial.print("weight = ");
  Serial.print(sensorValue / 1024 * 200);
  Serial.println(" N");
  delay(200);
}