/*
 * esp32_wifitput.ino - วัดว่า WiFi ของ ESP32 ตัวนี้ส่งได้จริงกี่ MB/s
 *
 * ทำไมต้องวัด
 * แผนใช้ AD7606 8 ช่องทำให้เฟรมโตจาก 6,992 เป็น 41,488 ไบต์ ถ้ายังส่งผ่าน
 * UART 2 Mbaud (200 KB/s) จะเหลือ 4.5 fps จาก 20.8 fps ที่ทำได้ตอนนี้
 * ทางออกที่ไม่ต้องซื้ออะไรคือ WiFi ซึ่งบนกระดาษเร็วกว่ามาก - แต่ตัวเลขที่
 * เล่ากันมี ตั้งแต่ 0.5 ถึง 5 MB/s ต่างกันสิบเท่า **ต้องวัดของจริงก่อนตัดสินใจ**
 *
 * สองโหมด เลือกตอนรัน ไม่ต้องคอมไพล์ใหม่
 *   ap                  ESP32 เป็นตัวปล่อยสัญญาณเอง - ไม่ต้องใช้รหัสเราเตอร์เลย
 *                       ได้เพดานของชิปตรง ๆ ไม่มีเราเตอร์มาปน แต่เครื่อง PC
 *                       ต้องย้ายมาต่อ AP นี้ ซึ่งแปลว่าเน็ตหลุดชั่วคราว
 *   sta <ssid> <pass>   ต่อเข้าเราเตอร์ที่มีอยู่ - PC อยู่ที่เดิม เน็ตไม่หลุด
 *                       แต่เลขที่ได้จะรวมคุณภาพเราเตอร์เข้าไปด้วย
 *
 * **พิมพ์คำสั่ง sta ในเทอร์มินัลของคุณเอง** รหัส WiFi ไม่ควรผ่านมือใคร
 * เฟิร์มแวร์นี้ไม่พิมพ์รหัสกลับออกมา และไม่เก็บลงแฟลช
 *
 * วัดสองแบบ เพราะสองแบบตอบคนละคำถาม
 *   stream  ส่งรัว ๆ ไม่หยุด          -> เพดานบนสุดของช่องทาง
 *   frame   ส่งเป็นก้อน 41,488 ไบต์  -> ของจริง มีค่าโสหุ้ยต่อเฟรมรวมอยู่
 * ตัวเลขที่เอาไปคิด fps คือ frame ไม่ใช่ stream
 *
 * SERIAL 2000000 baud · TCP port 5001
 */

#include <Arduino.h>
#include <WiFi.h>

#define TPUT_PORT 5001
#define AP_SSID "esp32-tput"
#define AP_PASS "tput12345"     /* ใช้ครั้งเดียวเพื่อวัด ไม่ใช่รหัสของใคร */

/* 41,488 = ขนาดเฟรมจริงของแผน AD7606 8 ช่องที่ระยะ 200 ซม.
 * (2,593 ตัวอย่าง/ช่อง x 8 ช่อง x 2 ไบต์) วัดด้วยขนาดนี้เลยจะได้ไม่ต้องเดา */
#define FRAME_BYTES 41488
#define CHUNK 1460              /* = MSS ของ TCP ส่งพอดีซองไม่ต้องแตก */

static WiFiServer server(TPUT_PORT);
static uint8_t buf[CHUNK];
static bool net_up = false;

static void help() {
  Serial.println("# ap                  - ปล่อยสัญญาณเอง (ไม่ต้องใช้รหัสเราเตอร์)");
  Serial.println("# sta <ssid> <pass>   - ต่อเข้าเราเตอร์ (พิมพ์เองในเทอร์มินัลคุณ)");
  Serial.println("# i                   - สถานะตอนนี้");
}

static void start_server(const IPAddress &ip) {
  server.begin();
  server.setNoDelay(true);
  net_up = true;
  Serial.printf("#READY ip=%s port=%d frame=%d\n",
                ip.toString().c_str(), TPUT_PORT, FRAME_BYTES);
}

static void do_ap() {
  WiFi.mode(WIFI_AP);
  if (!WiFi.softAP(AP_SSID, AP_PASS)) {
    Serial.println("!! เปิด AP ไม่ได้");
    return;
  }
  Serial.printf("# AP \"%s\" รหัส \"%s\"\n", AP_SSID, AP_PASS);
  start_server(WiFi.softAPIP());
}

static void do_sta(char *line) {
  /* แยก ssid กับ pass ด้วยช่องว่างตัวแรก - ssid ที่มีช่องว่างจะใช้ไม่ได้
   * ยอมรับข้อจำกัดนี้เพราะเป็นเครื่องมือวัดชั่วคราว ไม่ใช่ของใช้จริง */
  char *ssid = strtok(line, " ");
  char *pass = strtok(NULL, "");
  if (!ssid) { Serial.println("!! ใช้: sta <ssid> <pass>"); return; }
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, pass ? pass : "");
  Serial.printf("# กำลังต่อ \"%s\" ...\n", ssid);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) delay(250);
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("!! ต่อไม่ได้ - เช็คชื่อกับรหัส แล้วลองใหม่");
    return;
  }
  Serial.printf("# ต่อแล้ว rssi=%d dBm\n", WiFi.RSSI());
  start_server(WiFi.localIP());
}

/* ส่งจนครบ n ไบต์ · คืนจำนวนที่ส่งได้จริง (น้อยกว่า n = สายขาด) */
static uint32_t blast(WiFiClient &c, uint32_t n) {
  uint32_t sent = 0;
  while (sent < n && c.connected()) {
    uint32_t want = min((uint32_t)CHUNK, n - sent);
    int w = c.write(buf, want);
    if (w <= 0) { delay(1); if (!c.connected()) break; continue; }
    sent += w;
  }
  return sent;
}

static void serve(WiFiClient &c) {
  c.setNoDelay(true);
  Serial.printf("# ลูกข่ายต่อเข้ามา %s\n", c.remoteIP().toString().c_str());

  /* 1) stream - ส่งรัว 8 วินาที ไม่มีขอบเฟรม = เพดานบนของช่องทาง */
  uint32_t t0 = millis(), sent = 0;
  while (millis() - t0 < 8000 && c.connected()) sent += blast(c, CHUNK * 16);
  uint32_t ms = millis() - t0;
  Serial.printf("#STREAM bytes=%lu ms=%lu MBps=%.3f\n",
                (unsigned long)sent, (unsigned long)ms, sent / 1000.0 / ms);

  /* 2) frame - ส่งทีละเฟรมจริง 60 เฟรม วัดเวลาต่อเฟรมทีละอัน
   * เก็บช้าสุดด้วย เพราะ fps ที่รู้สึกได้ถูกกำหนดโดยเฟรมที่ช้า ไม่ใช่ค่าเฉลี่ย */
  uint32_t worst = 0, total = 0, nf = 0;
  for (int i = 0; i < 60 && c.connected(); i++) {
    uint32_t f0 = millis();
    if (blast(c, FRAME_BYTES) < FRAME_BYTES) break;
    uint32_t dt = millis() - f0;
    total += dt; nf++;
    if (dt > worst) worst = dt;
  }
  if (nf) {
    Serial.printf("#FRAME n=%lu avg_ms=%.1f worst_ms=%lu MBps=%.3f fps=%.1f\n",
                  (unsigned long)nf, (float)total / nf, (unsigned long)worst,
                  (float)FRAME_BYTES * nf / 1000.0 / total, 1000.0 * nf / total);
  }
  c.stop();
  Serial.println("# จบรอบ - ต่อเข้ามาใหม่ได้เลย");
}

void setup() {
  Serial.begin(2000000);
  delay(300);
  for (int i = 0; i < CHUNK; i++) buf[i] = (uint8_t)i;
  Serial.println("\n# esp32_wifitput - วัด WiFi throughput");
  help();
  /* เปิด AP เองตั้งแต่บูต **ห้ามพึ่งคำสั่งทางซีเรียล**
   * เพราะการเปิด/ปิดพอร์ตซีเรียลจะกระตุก DTR/RTS ซึ่งรีเซ็ตบอร์ด
   * ถ้ารอคำสั่ง AP จะดับทุกครั้งที่สคริปต์ฝั่ง PC ปล่อยพอร์ต
   * ต่อสาย USB = AP พร้อมใช้ทันที ไม่ต้องคุยกับมันเลย */
  do_ap();
}

void loop() {
  static char line[160];
  static size_t len = 0;
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      line[len] = 0;
      if (!strncmp(line, "sta ", 4)) do_sta(line + 4);
      else if (!strcmp(line, "ap")) do_ap();
      else if (!strcmp(line, "i"))
        Serial.printf("# net_up=%d mode=%d ip=%s\n", net_up, (int)WiFi.getMode(),
                      (WiFi.getMode() == WIFI_AP ? WiFi.softAPIP()
                                                 : WiFi.localIP()).toString().c_str());
      else if (len) help();
      len = 0;
    } else if (len < sizeof(line) - 1) line[len++] = ch;
  }
  if (net_up) {
    WiFiClient c = server.accept();
    if (c) serve(c);
  }
}
