/*
 * esp32s3_adctest.ino - วัดว่า ADC แบบต่อเนื่อง (DMA) ของ ESP32-S3 เร็วสุดเท่าไร
 *
 * ทำไมต้องวัด
 * เฟิร์มแวร์ scope เดิมเขียนล็อกไว้ว่า S3 "ADC ช้ากว่ามาก" แต่เป็นหมายเหตุในโค้ด
 * ไม่ใช่ตัวเลขที่วัด ตัวเลขนี้ตัดสินว่าพอร์ตเฟิร์มแวร์ scope ลง S3 ได้ไหม
 * (ESP32 เดิมได้รวม ~266 kHz ที่ 4 ช่อง = 66 kHz/ช่อง)
 *
 * วัดแบบเดียวกับเฟิร์มแวร์เดิม: ขอ rate หนึ่ง แล้วนับ "ผลแปลงใหม่" จากการที่แท็กช่อง
 * เปลี่ยน ไม่นับคำที่ DMA เขียนซ้ำ เลขที่ได้คือ rate จริง ไม่ใช่ rate ที่ขอ
 * รายงานทั้ง "ไดรเวอร์ปฏิเสธ rate นี้" และ "รับแต่ซ้ำ" เพราะสองอาการนี้ต่างกัน
 *
 * ช่อง: ADC1 CH0-3 = GPIO1-4 (ขาลอย ค่าไม่มีความหมาย วัดแต่ความเร็ว)
 * สั่ง r ทางพอร์ต USB ของตัวชิป เพื่อรันทั้งตาราง
 */

#include <Arduino.h>
#include "HWCDC.h"
#include "esp_adc/adc_continuous.h"
#include "esp_timer.h"

static HWCDC usb;

#define NCH 4
static const uint32_t RATES[] = {20000, 40000, 60000, 80000, 83333, 100000,
                                 200000, 400000, 1000000, 2000000};

static void one(uint32_t hz) {
  adc_continuous_handle_t h = NULL;
  adc_continuous_handle_cfg_t hc = {};
  hc.max_store_buf_size = 16384;
  hc.conv_frame_size = 512;
  if (adc_continuous_new_handle(&hc, &h) != ESP_OK) {
    usb.printf("%8lu  สร้าง handle ไม่ได้\n", (unsigned long)hz);
    return;
  }
  adc_digi_pattern_config_t pat[NCH] = {};
  for (int i = 0; i < NCH; i++) {
    pat[i].atten = ADC_ATTEN_DB_12;
    pat[i].channel = i;
    pat[i].unit = ADC_UNIT_1;
    pat[i].bit_width = ADC_BITWIDTH_12;
  }
  adc_continuous_config_t cc = {};
  cc.pattern_num = NCH;
  cc.adc_pattern = pat;
  cc.sample_freq_hz = hz;
  cc.conv_mode = ADC_CONV_SINGLE_UNIT_1;
  cc.format = ADC_DIGI_OUTPUT_FORMAT_TYPE2;   /* S3 ใช้ TYPE2 ไม่ใช่ TYPE1 แบบ ESP32 */
  esp_err_t err = adc_continuous_config(h, &cc);
  if (err != ESP_OK) {
    usb.printf("%8lu  ไดรเวอร์ปฏิเสธ: %s\n", (unsigned long)hz, esp_err_to_name(err));
    adc_continuous_deinit(h);
    return;
  }
  if (adc_continuous_start(h) != ESP_OK) {
    usb.printf("%8lu  start ไม่ได้\n", (unsigned long)hz);
    adc_continuous_deinit(h);
    return;
  }
  uint8_t buf[512];
  uint32_t got = 0;
  const int64_t warm = esp_timer_get_time();
  while (esp_timer_get_time() - warm < 100000)
    adc_continuous_read(h, buf, sizeof(buf), &got, 5);   /* ทิ้งช่วงเริ่ม */

  uint32_t words = 0, conv = 0, seen = 0;
  int prev = -1;
  const int64_t t0 = esp_timer_get_time();
  while (esp_timer_get_time() - t0 < 1000000) {
    if (adc_continuous_read(h, buf, sizeof(buf), &got, 20) != ESP_OK || !got) continue;
    for (uint32_t i = 0; i + SOC_ADC_DIGI_RESULT_BYTES <= got; i += SOC_ADC_DIGI_RESULT_BYTES) {
      adc_digi_output_data_t *p = (adc_digi_output_data_t *)&buf[i];
      int ch = p->type2.channel;
      words++;
      if (ch != prev) { conv++; prev = ch; }
      if (ch < 8) seen |= 1u << ch;
    }
  }
  const double sec = (esp_timer_get_time() - t0) / 1e6;
  adc_continuous_stop(h);
  adc_continuous_deinit(h);
  usb.printf("%8lu  จริง %8.0f conv/s  (%.0f kHz/ช่อง)  คำที่ได้ %8.0f/s  ซ้ำ x%.2f  ช่องที่เห็น 0x%02lX\n",
             (unsigned long)hz, conv / sec, conv / sec / NCH / 1000.0, words / sec,
             conv ? (double)words / conv : 0.0, (unsigned long)seen);
}

void setup() {
  usb.begin();
}

void loop() {
  if (usb.available() && usb.read() == 'r') {
    usb.printf("#START ADC1 %d ช่อง · S3 · SOC_ADC_DIGI_RESULT_BYTES=%d\n", NCH,
               (int)SOC_ADC_DIGI_RESULT_BYTES);
    for (uint32_t hz : RATES) one(hz);
    usb.println("#DONE");
  }
}
