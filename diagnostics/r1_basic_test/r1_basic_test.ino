/*
 * r1_basic_test.ino - throwaway diagnostic sketch.
 * Bypasses all custom TDOA logic (Timer1/PCINT) and just does a plain
 * pulseIn()-based distance read on ECHO1 (A0), same TRIG (A2) as usual.
 * Purpose: find out whether board 1 (T+R1) itself is stuck reporting a
 * fixed ~4.6cm reflection regardless of the real target, or whether that
 * was an artifact of the custom interrupt-driven firmware.
 */
const int TRIG_PIN = A2;
const int ECHO_PIN = A0; // R1 only

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(ECHO_PIN, INPUT);
  Serial.println("# r1_basic_test ready (plain pulseIn, bypasses custom TDOA code)");
}

void loop() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(4);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long durationUs = pulseIn(ECHO_PIN, HIGH, 30000UL);
  float distanceCm = durationUs * 0.0343f / 2.0f;

  Serial.print("pulse_us=");
  Serial.print(durationUs);
  Serial.print(" distance_cm=");
  Serial.println(distanceCm, 2);

  delay(150);
}
