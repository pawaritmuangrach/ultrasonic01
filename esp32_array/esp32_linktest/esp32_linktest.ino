/*
 * esp32_linktest.ino - วัดว่าสายซีเรียลของ ESP32-S3 ส่งข้อมูลได้จริงกี่ MB/s และไม่เพี้ยน
 *
 * ทำไมต้องวัด
 * บอร์ด S3 มีสองช่อง USB-C: ช่อง "UART" ผ่านชิปแปลง CH343 กับช่อง "USB" ที่เป็น
 * USB ของตัวชิปเอง ข้อมูลของ AD7606 8 ช่อง = 41,488 ไบต์ต่อเฟรม ถ้าผ่านชิปแปลง
 * เพดานคือ baud ที่ชิปแปลงรับได้ ซึ่งเอกสารกับของจริงมักไม่ตรงกัน จึงต้องวัด
 *
 * แต่ละแพ็กเก็ต 1024 ไบต์ = หัว 4 (A5 5A C3 3C) + เลขลำดับ 4 + เนื้อหาที่คำนวณได้
 * ฝั่ง PC เช็คได้ทุกไบต์ว่าตรง และนับได้ว่าแพ็กเก็ตหายกี่อัน
 * **วัดความเร็วอย่างเดียวไม่พอ** baud สูงเกินชิปแปลงมักยังรับ-ส่งได้ แต่ไบต์เพี้ยน
 *
 * คำสั่ง (ส่งทางพอร์ตไหนก็ได้ ตอบกลับพอร์ตเดียวกัน)
 *   b <baud>   เปลี่ยน baud ของ UART (ไม่มีผลกับ USB ของตัวชิป)
 *   g <ms>     ส่งรัวเป็นเวลา <ms> มิลลิวินาที จบด้วยท้ายแพ็กเก็ต DE AD BE EF + จำนวนที่ส่ง
 *   i          บอกชนิดชิปและ USB mode
 */

#include <Arduino.h>
#include "HWCDC.h"

#define PKT 1024
#define HDR 8

static const uint8_t MAGIC[4] = {0xA5, 0x5A, 0xC3, 0x3C};
static const uint8_t TRAIL[4] = {0xDE, 0xAD, 0xBE, 0xEF};
static uint8_t pkt[PKT];

/* แกนเก็บ USB ของตัวชิปไม่มีออบเจ็กต์สำเร็จรูปตอนปิด "CDC On Boot" (ค่าปกติ)
 * จึงสร้างเอง ถ้าสร้างตอนเปิด CDC On Boot จะชนกับของแกน */
static HWCDC nativeCdc;

/* เขียนจนครบ · พอร์ต USB ของตัวชิปคืน 0 ได้ถ้า PC ไม่อ่าน จึงต้องมีทางออก */
static bool put(Stream &p, const uint8_t *b, size_t n) {
  size_t off = 0;
  uint32_t stall = 0;
  while (off < n) {
    size_t w = p.write(b + off, n - off);
    if (w == 0) {
      if (++stall > 500) return false;
      delay(1);
    } else {
      stall = 0;
      off += w;
    }
  }
  return true;
}

static void blast(Stream &p, uint32_t ms) {
  uint32_t seq = 0, t0 = millis();
  while (millis() - t0 < ms) {
    memcpy(pkt, MAGIC, 4);
    memcpy(pkt + 4, &seq, 4);
    for (int i = HDR; i < PKT; i++) pkt[i] = (uint8_t)(seq * 7 + i);
    if (!put(p, pkt, PKT)) return;
    seq++;
  }
  uint8_t tr[8];
  memcpy(tr, TRAIL, 4);
  memcpy(tr + 4, &seq, 4);
  put(p, tr, 8);
  p.flush();
}

static void command(Stream &p, char *line) {
  if (line[0] == 'b') {
    uint32_t n = strtoul(line + 1, NULL, 10);
    if (&p != (Stream *)&Serial || n < 9600) {
      p.println("!! b ใช้ได้เฉพาะ UART");
      return;
    }
    p.printf("#BAUD %lu\n", (unsigned long)n);
    p.flush();
    delay(60);
    Serial.updateBaudRate(n);
  } else if (line[0] == 'g') {
    uint32_t ms = strtoul(line + 1, NULL, 10);
    blast(p, ms ? ms : 3000);
  } else if (line[0] == 'i') {
    p.printf("#INFO chip=%s rev=%d cpu=%luMHz usbmode=%d\n", ESP.getChipModel(),
             (int)ESP.getChipRevision(), (unsigned long)ESP.getCpuFreqMHz(),
             (int)ARDUINO_USB_MODE);
  }
}

static void poll(Stream &p, char *buf, size_t &len) {
  while (p.available()) {
    char ch = p.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      buf[len] = 0;
      if (len) command(p, buf);
      len = 0;
    } else if (len < 31) {
      buf[len++] = ch;
    }
  }
}

void setup() {
  Serial.setTxBufferSize(8192);   /* ต้องตั้งก่อน begin */
  Serial.begin(115200);
  nativeCdc.begin();
}

void loop() {
  static char ub[32], cb[32];
  static size_t ul = 0, cl = 0;
  poll(Serial, ub, ul);
  poll(nativeCdc, cb, cl);
}
