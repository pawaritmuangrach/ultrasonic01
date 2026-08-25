/*
 * esp32_selftest.ino - toolchain and ADC self-test for the ultrasonic array.
 *
 * WHY THIS EXISTS
 * The whole project rests on one unverified assumption: that the ESP32 can
 * stream ADC1 continuously at ~1-2 MS/s. If it cannot, the array design has
 * to change. This sketch answers that question before a single component is
 * soldered, using one jumper wire and nothing else.
 *
 * WIRING - one male-male jumper:
 *     GPIO25  ---->  GPIO34
 *
 * GPIO25 is DAC1 (and an LEDC output); GPIO34 is ADC1_CH6 and is input-only,
 * so nothing can be damaged by getting this backwards.
 *
 * WHAT IT DOES
 *   1. Generates a 40 kHz signal on GPIO25 - a real cosine via the DAC when
 *      the driver is available, otherwise a square wave via LEDC.
 *   2. Captures it on GPIO34 with the ESP-IDF continuous (DMA) ADC driver.
 *   3. Measures the sample rate it *actually* achieved, rather than trusting
 *      the rate it asked for, and reports both.
 *   4. Streams the samples to tools/scope_view.py.
 *
 * Note this uses esp_adc/adc_continuous.h directly, not Arduino's
 * analogContinuous(). The Arduino wrapper averages conversions_per_pin
 * samples before handing them over, which is fine for measuring a voltage
 * and useless for capturing a waveform.
 *
 * REQUIRES: Arduino-ESP32 core 3.x (built on ESP-IDF 5.x).
 * Board: "ESP32 Dev Module". Original ESP32 only - S2/S3/C3 have a far
 * slower continuous ADC and different channel maps.
 *
 * SERIAL COMMANDS (921600 baud):
 *   ?          help
 *   i          info: chip, core, config, heap
 *   r          run one capture and stream the frame
 *   f <hz>     set ADC sample rate      (default 1000000)
 *   n <count>  set samples per capture  (default 20000, max 20000)
 *   s <hz>     set source frequency     (default 40000)
 *   m <mode>   source mode: dac | pwm | off
 *   d <hz>     independent test tone on GPIO26 (0 = off), for RX gain checks
 */

#include <Arduino.h>
#include "esp_adc/adc_continuous.h"
#include "esp_timer.h"
#include "driver/ledc.h"

#if __has_include("driver/dac_cosine.h")
#include "driver/dac_cosine.h"
#define HAVE_DAC_COSINE 1
#endif

// ---------------------------------------------------------------- config --

/* One channel at a time, selectable at runtime.
 *
 * Multi-channel continuous ADC on the original ESP32 gave readings that could
 * not be true (a passive divider midpoint appearing to swing rail to rail),
 * so probing stays single-channel - which measured cleanly and repeatably -
 * and the `c` command switches which probe wire is being read instead. All
 * probe wires can stay plugged in. */
static adc_channel_t CH_LIST[] = { ADC_CHANNEL_6 };  // GPIO34 by default
static const int NUM_CH = 1;

#define MAX_SAMPLES 20000  // 20 ms at 1 MS/s; 40 KB of DRAM
#define SOURCE_GPIO 25
#define TEST_GPIO 26  // DAC2 - separate tone for measuring RX front-end gain

static uint32_t g_sample_hz = 1000000;
static uint32_t g_num_samples = MAX_SAMPLES;
static uint32_t g_source_hz = 40000;
static char g_source_mode[8] = "dac";

static uint16_t g_samples[MAX_SAMPLES * NUM_CH];
static uint32_t g_chan_seen = 0;

static adc_continuous_handle_t g_adc = NULL;
static bool g_adc_running = false;

#ifdef HAVE_DAC_COSINE
static dac_cosine_handle_t g_dac = NULL;
static dac_cosine_handle_t g_test = NULL;
#endif
static uint32_t g_test_hz = 0;

// ---------------------------------------------------------------- source --

static void source_stop() {
#ifdef HAVE_DAC_COSINE
  if (g_dac) {
    dac_cosine_stop(g_dac);
    dac_cosine_del_channel(g_dac);
    g_dac = NULL;
  }
#endif
  ledcDetach(SOURCE_GPIO);
  pinMode(SOURCE_GPIO, INPUT);
}

static void source_start() {
  source_stop();

  if (strcmp(g_source_mode, "off") == 0) {
    Serial.println("# source: off");
    return;
  }

#ifdef HAVE_DAC_COSINE
  if (strcmp(g_source_mode, "dac") == 0) {
    dac_cosine_config_t cfg = {};
    cfg.chan_id = DAC_CHAN_0;  // GPIO25
    cfg.freq_hz = g_source_hz;
    cfg.clk_src = DAC_COSINE_CLK_SRC_DEFAULT;
    cfg.offset = 0;
    cfg.phase = DAC_COSINE_PHASE_0;
    cfg.atten = DAC_COSINE_ATTEN_DB_6;  // ~0.8 Vpp, well inside the ADC range
    cfg.flags.force_set_freq = true;
    esp_err_t err = dac_cosine_new_channel(&cfg, &g_dac);
    if (err == ESP_OK && dac_cosine_start(g_dac) == ESP_OK) {
      Serial.printf("# source: DAC cosine, requested %u Hz on GPIO%d\n",
                    (unsigned)g_source_hz, SOURCE_GPIO);
      Serial.println("# note: the DAC quantises frequency - trust the FFT, not this number");
      return;
    }
    Serial.println("# DAC cosine unavailable, falling back to PWM");
    g_dac = NULL;
  }
#endif

  // LEDC square wave. Harmonic-rich, but for validating the capture chain
  // that is a feature: the FFT should show 40k, 120k, 200k... and any that
  // land above fs/2 will fold back, which itself confirms the sample rate.
  ledcAttach(SOURCE_GPIO, g_source_hz, 8);
  ledcWrite(SOURCE_GPIO, 128);  // 50% duty
  strcpy(g_source_mode, "pwm");
  Serial.printf("# source: LEDC square %u Hz on GPIO%d\n", (unsigned)g_source_hz,
                SOURCE_GPIO);
}

/* Independent test tone on GPIO26.
 *
 * Once GPIO25 is driving the 74HCT04 it can no longer double as a signal
 * generator, so gain measurements on the receive chain need their own
 * output. Feed this through a 100k/100R divider to get ~0.8 mVpp - roughly
 * what a real echo delivers - and compare the amplifier output against the
 * expected gain. */
static void test_tone(uint32_t hz) {
#ifdef HAVE_DAC_COSINE
  if (g_test) {
    dac_cosine_stop(g_test);
    dac_cosine_del_channel(g_test);
    g_test = NULL;
  }
  g_test_hz = 0;
  if (hz == 0) {
    Serial.println("# test tone: off");
    return;
  }
  dac_cosine_config_t cfg = {};
  cfg.chan_id = DAC_CHAN_1;  // GPIO26
  cfg.freq_hz = hz;
  cfg.clk_src = DAC_COSINE_CLK_SRC_DEFAULT;
  cfg.offset = 0;
  cfg.phase = DAC_COSINE_PHASE_0;
  cfg.atten = DAC_COSINE_ATTEN_DB_6;  // ~0.8 Vpp -> 0.8 mVpp after the divider
  cfg.flags.force_set_freq = true;
  if (dac_cosine_new_channel(&cfg, &g_test) == ESP_OK &&
      dac_cosine_start(g_test) == ESP_OK) {
    g_test_hz = hz;
    Serial.printf("# test tone: %u Hz on GPIO%d (~0.8 Vpp)\n", (unsigned)hz, TEST_GPIO);
  } else {
    Serial.println("!! test tone failed to start");
    g_test = NULL;
  }
#else
  Serial.println("!! dac_cosine not available in this core");
#endif
}

// ------------------------------------------------------------------- adc --

static void adc_teardown() {
  if (g_adc) {
    // Only stop if it is actually running - calling stop on a stopped driver
    // logs an ESP_ERR line that looks like a real fault and is not one.
    if (g_adc_running) {
      adc_continuous_stop(g_adc);
      g_adc_running = false;
    }
    adc_continuous_deinit(g_adc);
    g_adc = NULL;
  }
}

static bool adc_setup() {
  adc_teardown();

  adc_continuous_handle_cfg_t hcfg = {};
  hcfg.max_store_buf_size = 8192;  // bytes; must be a multiple of conv_frame_size
  hcfg.conv_frame_size = 1024;
  esp_err_t err = adc_continuous_new_handle(&hcfg, &g_adc);
  if (err != ESP_OK) {
    Serial.printf("!! adc_continuous_new_handle failed: %s\n", esp_err_to_name(err));
    return false;
  }

  adc_digi_pattern_config_t pattern[NUM_CH] = {};
  for (int i = 0; i < NUM_CH; i++) {
    pattern[i].atten = ADC_ATTEN_DB_12;  // ~0-3.1 V input range
    pattern[i].channel = CH_LIST[i] & 0x7;
    pattern[i].unit = ADC_UNIT_1;
    pattern[i].bit_width = ADC_BITWIDTH_12;
  }

  adc_continuous_config_t ccfg = {};
  ccfg.pattern_num = NUM_CH;
  ccfg.adc_pattern = pattern;
  ccfg.sample_freq_hz = g_sample_hz;
  ccfg.conv_mode = ADC_CONV_SINGLE_UNIT_1;
  ccfg.format = ADC_DIGI_OUTPUT_FORMAT_TYPE1;  // correct for the original ESP32

  err = adc_continuous_config(g_adc, &ccfg);
  if (err != ESP_OK) {
    Serial.printf("!! adc_continuous_config failed at %u Hz: %s\n",
                  (unsigned)g_sample_hz, esp_err_to_name(err));
    Serial.println("!! the rate is out of range for this chip - try a lower 'f'");
    adc_teardown();
    return false;
  }
  return true;
}

/* Capture g_num_samples per channel. Returns samples collected. */
static uint32_t capture(uint32_t &elapsed_us) {
  const uint32_t want = g_num_samples * NUM_CH;
  uint8_t rbuf[1024];
  uint32_t n = 0;
  int64_t t_first = 0;
  bool started = false;

  g_chan_seen = 0;
  elapsed_us = 0;

  if (adc_continuous_start(g_adc) != ESP_OK) {
    Serial.println("!! adc_continuous_start failed");
    return 0;
  }
  g_adc_running = true;

  const int64_t give_up = esp_timer_get_time() + 2000000;  // 2 s ceiling

  while (n < want && esp_timer_get_time() < give_up) {
    uint32_t got = 0;
    esp_err_t err = adc_continuous_read(g_adc, rbuf, sizeof(rbuf), &got, 200);
    if (err != ESP_OK || got == 0) continue;

    // Discard the first batch: it contains the DMA ramp-up, and timing from
    // here gives an honest steady-state rate rather than a flattering one.
    if (!started) {
      started = true;
      t_first = esp_timer_get_time();
      continue;
    }

    for (uint32_t i = 0; i + SOC_ADC_DIGI_RESULT_BYTES <= got && n < want;
         i += SOC_ADC_DIGI_RESULT_BYTES) {
      adc_digi_output_data_t *p = (adc_digi_output_data_t *)&rbuf[i];
      g_chan_seen |= (1u << p->type1.channel);
      g_samples[n++] = p->type1.data;
    }
  }

  const int64_t t_last = esp_timer_get_time();
  adc_continuous_stop(g_adc);
  g_adc_running = false;

  if (started && n > 1) elapsed_us = (uint32_t)(t_last - t_first);
  return n;
}

static void send_frame(uint32_t n_total, uint32_t elapsed_us) {
  const uint32_t per_ch = n_total / NUM_CH;

  Serial.printf("#FRAME rate=%u ch=%d n=%u us=%u order=", (unsigned)g_sample_hz,
                NUM_CH, (unsigned)per_ch, (unsigned)elapsed_us);
  for (int i = 0; i < NUM_CH; i++) {
    Serial.printf("%d%s", (int)(CH_LIST[i] & 0x7), (i + 1 < NUM_CH) ? "," : "");
  }
  Serial.print("\n");
  Serial.flush();

  Serial.write((const uint8_t *)"USC1", 4);

  uint16_t checksum = 0;
  const uint8_t *bytes = (const uint8_t *)g_samples;
  const uint32_t nbytes = per_ch * NUM_CH * 2;
  for (uint32_t i = 0; i < nbytes; i += 512) {
    const uint32_t chunk = min((uint32_t)512, nbytes - i);
    Serial.write(bytes + i, chunk);
    for (uint32_t k = 0; k < chunk; k++) checksum += bytes[i + k];
  }
  Serial.write((const uint8_t *)&checksum, 2);
  Serial.flush();
}

static void run_capture() {
  uint32_t elapsed_us = 0;
  const uint32_t n = capture(elapsed_us);

  if (n == 0) {
    Serial.println("!! capture returned nothing - ADC not streaming");
    return;
  }

  // Sanity-check the assumption that the data really is the channel we asked
  // for. If this trips, the DMA result layout differs from what we expect.
  uint32_t expect_mask = 0;
  for (int i = 0; i < NUM_CH; i++) expect_mask |= (1u << (CH_LIST[i] & 0x7));
  if (g_chan_seen != expect_mask) {
    Serial.printf("!! channel mask mismatch: saw %#x, expected %#x\n",
                  (unsigned)g_chan_seen, (unsigned)expect_mask);
  }

  if (elapsed_us > 0) {
    const double achieved = (double)(n / NUM_CH - 1) / (elapsed_us * 1e-6);
    Serial.printf("# requested %.1f kS/s, achieved %.1f kS/s (%.1f%%)\n",
                  g_sample_hz / 1000.0, achieved / 1000.0,
                  100.0 * achieved / g_sample_hz);
  }

  send_frame(n, elapsed_us);
}

// ------------------------------------------------------------------ info --

static void print_info() {
  Serial.println("# ---------------- esp32_selftest ----------------");
  Serial.printf("# chip           : %s rev %d, %d core(s)\n", ESP.getChipModel(),
                ESP.getChipRevision(), ESP.getChipCores());
  Serial.printf("# arduino core   : %s\n", ESP_ARDUINO_VERSION_STR);
  Serial.printf("# idf            : %s\n", esp_get_idf_version());
  Serial.printf("# cpu            : %u MHz\n", (unsigned)getCpuFrequencyMhz());
  Serial.printf("# free heap      : %u bytes\n", (unsigned)ESP.getFreeHeap());
  Serial.printf("# adc rate req   : %u Hz\n", (unsigned)g_sample_hz);
  Serial.printf("# samples/capture: %u\n", (unsigned)g_num_samples);
  {
    const int ch = (int)(CH_LIST[0] & 0x7);
    const char *pin = (ch == 6) ? "GPIO34" : (ch == 7) ? "GPIO35"
                    : (ch == 4) ? "GPIO32" : (ch == 5) ? "GPIO33"
                    : (ch == 0) ? "GPIO36" : (ch == 3) ? "GPIO39" : "?";
    Serial.printf("# adc probe      : %s (ADC1_CH%d)\n", pin, ch);
  }
  Serial.printf("# source         : %s @ %u Hz on GPIO%d\n", g_source_mode,
                (unsigned)g_source_hz, SOURCE_GPIO);
  Serial.printf("# test tone      : %u Hz on GPIO%d\n", (unsigned)g_test_hz, TEST_GPIO);
#ifdef HAVE_DAC_COSINE
  Serial.println("# dac_cosine     : available");
#else
  Serial.println("# dac_cosine     : NOT available (PWM fallback only)");
#endif
  if (String(ESP.getChipModel()).indexOf("ESP32-D") < 0 &&
      String(ESP.getChipModel()) != "ESP32") {
    Serial.println("!! this is not an original ESP32 - continuous ADC will be much slower");
  }
  Serial.println("# jumper GPIO25 -> GPIO34, then send 'r'");
  Serial.println("# -----------------------------------------------");
}

static void print_help() {
  Serial.println("# ?          this help");
  Serial.println("# i          info");
  Serial.println("# r          run one capture");
  Serial.println("# f <hz>     ADC sample rate");
  Serial.println("# n <count>  samples per capture");
  Serial.println("# s <hz>     source frequency");
  Serial.println("# m <mode>   source: dac | pwm | off");
  Serial.println("# d <hz>     test tone on GPIO26 for RX gain checks (0 = off)");
  Serial.println("# c <pin>    probe pin: 34 | 35 | 32 | 33 | 36 | 39");
}

// ------------------------------------------------------------- main loop --

static void handle_line(String line) {
  line.trim();
  if (line.length() == 0) return;
  const char cmd = line[0];
  const String arg = line.substring(1);

  switch (cmd) {
    case '?': print_help(); break;
    case 'i': print_info(); break;
    case 'r': run_capture(); break;

    case 'f': {
      const uint32_t hz = arg.toInt();
      // Bounds come from the chip itself, not a guess. On the original ESP32
      // these are 20 kHz and 2 MS/s - note 2 MS/s is the hard ceiling, so the
      // 6-channel array runs with no headroom at all.
      if (hz < SOC_ADC_SAMPLE_FREQ_THRES_LOW || hz > SOC_ADC_SAMPLE_FREQ_THRES_HIGH) {
        Serial.printf("!! rate must be %u..%u Hz\n",
                      (unsigned)SOC_ADC_SAMPLE_FREQ_THRES_LOW,
                      (unsigned)SOC_ADC_SAMPLE_FREQ_THRES_HIGH);
        break;
      }
      g_sample_hz = hz;
      if (adc_setup()) Serial.printf("# adc rate set to %u Hz\n", (unsigned)g_sample_hz);
      break;
    }

    case 'n': {
      const uint32_t cnt = arg.toInt();
      if (cnt < 100 || cnt > MAX_SAMPLES) {
        Serial.printf("!! count must be 100..%d\n", MAX_SAMPLES);
        break;
      }
      g_num_samples = cnt;
      Serial.printf("# samples per capture = %u (%.2f ms window)\n",
                    (unsigned)g_num_samples,
                    1000.0 * g_num_samples / g_sample_hz);
      break;
    }

    case 's': {
      const uint32_t hz = arg.toInt();
      if (hz == 0) {                       // "s 0" parks the pin
        source_stop();
        Serial.println("# source stopped");
        break;
      }
      if (hz < 100 || hz > 200000) {
        Serial.println("!! source must be 0, or 100..200000 Hz");
        break;
      }
      g_source_hz = hz;
      source_start();
      break;
    }

    case 'd': {
      const long hz = arg.toInt();
      if (hz != 0 && (hz < 100 || hz > 200000)) {
        Serial.println("!! test tone must be 0 (off) or 100..200000 Hz");
        break;
      }
      test_tone((uint32_t)hz);
      break;
    }

    case 'c': {
      const long pin = arg.toInt();
      int ch = -1;
      switch (pin) {
        case 34: ch = 6; break; case 35: ch = 7; break; case 32: ch = 4; break;
        case 33: ch = 5; break; case 36: ch = 0; break; case 39: ch = 3; break;
      }
      if (ch < 0) {
        Serial.println("!! probe pin must be 34, 35, 32, 33, 36 or 39");
        break;
      }
      CH_LIST[0] = (adc_channel_t)ch;
      if (adc_setup()) Serial.printf("# probing GPIO%d (ADC1_CH%d)\n", (int)pin, ch);
      break;
    }

    case 'm': {
      String mode = arg;
      mode.trim();
      if (mode != "dac" && mode != "pwm" && mode != "off") {
        Serial.println("!! mode must be dac | pwm | off");
        break;
      }
      strncpy(g_source_mode, mode.c_str(), sizeof(g_source_mode) - 1);
      source_start();
      break;
    }

    default:
      Serial.printf("!! unknown command '%c' - send '?' for help\n", cmd);
  }
}

void setup() {
  Serial.begin(921600);
  delay(300);
  Serial.println();
  print_info();

  if (!adc_setup()) {
    Serial.println("!! ADC init failed - nothing else will work until this is fixed");
  }
  source_start();
  Serial.println("# ready");
}

void loop() {
  static String line;
  while (Serial.available()) {
    const char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (line.length()) {
        handle_line(line);
        line = "";
      }
    } else if (line.length() < 32) {
      line += c;
    }
  }
}
