/*
 * ultrasonic_tdoa_nano.ino
 *
 * Arduino Nano (ATmega328P) - 1 Transmitter + 2 Receivers (US-015 x2) pseudo-TDOA sketch.
 * See US015_TDOA_Codex_Spec.md in this repo for the full derivation of every
 * formula/threshold used below.
 *
 * Wiring:
 *   A2 (PC2) -> TRIG (shared by both US-015 boards)
 *   A0 (PC0) -> ECHO1 (board 1, has transmitter T + receiver R1)
 *   A1 (PC1) -> ECHO2 (board 2, receiver R2 only)
 *
 * Hard requirements from the spec (section 24):
 *   - No pulseIn(), no analogRead() on ECHO pins
 *   - Timer1 prescaler 8 (0.5 us/tick), Pin Change Interrupt on Port C
 *   - ISR: short, no Serial, no floating point
 *   - Timer1 overflow handled, volatile snapshot read atomically
 *   - timeout, calibration offset, physical plausibility gate, median filter,
 *     cycle-slip detector (25 us period), confidence score
 *   - human-readable AND CSV output, angle only printed when status == VALID
 */

#include <avr/interrupt.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Pin definitions (named per spec; these are the digital pin numbers behind
// the "A0/A1/A2" silkscreen labels -- they are NOT read with analogRead()).
// ---------------------------------------------------------------------------
const uint8_t TRIG_PIN  = A2; // PC2
const uint8_t ECHO1_PIN = A0; // PC0 = PCINT8
const uint8_t ECHO2_PIN = A1; // PC1 = PCINT9

// Set to 1 if using the discrete-transistor analog front-end (gain + envelope
// detector + transistor threshold stage), whose output idles HIGH and pulls
// LOW while a strong-enough signal is present. Set to 0 for a normal
// HC-SR04/US-015-style module whose ECHO idles LOW and goes HIGH during the
// measurement window.
#define ECHO_ACTIVE_LOW 1

// ---------------------------------------------------------------------------
// Tunable parameters (section 16.3). Some are runtime-adjustable via serial
// commands and therefore not `const`.
// ---------------------------------------------------------------------------
float SOUND_SPEED_CM_PER_US = 0.0343f;     // updated by 't<celsius>' command
float RECEIVER_SPACING_CM   = 5.0f;        // updated by 'd<cm>' command
float CALIBRATION_OFFSET_US = 0.0f;        // updated by 'c' calibration routine

const uint32_t MEASUREMENT_TIMEOUT_US = 30000UL; // ~515 cm one-way ceiling
const float    MIN_VALID_US   = 100.0f;          // reject ringing / direct coupling
const float    MAX_DISTANCE_CM = 100.0f;         // sanity ceiling for pulse width
const size_t   FILTER_WINDOW  = 5;               // odd window, median filter
const float    CYCLE_PERIOD_US = 25.0f;          // 1 / 40 kHz
const float    CYCLE_SLIP_TOLERANCE_US = 3.0f;
const uint16_t MEASUREMENT_INTERVAL_MS = 100;    // ~10 Hz ping rate
const uint16_t CALIBRATION_SAMPLES = 200;

// ---------------------------------------------------------------------------
// Timer1: prescaler 8 => 16 MHz / 8 = 2 MHz => 0.5 us per tick.
// We extend the 16-bit hardware counter to 32 bits using the overflow ISR.
// ---------------------------------------------------------------------------
volatile uint32_t timer1OverflowCount = 0;

ISR(TIMER1_OVF_vect) {
  timer1OverflowCount++;
}

// Overflow-safe 32-bit tick read. Must be called with interrupts disabled
// by the caller if a perfectly consistent snapshot with other state is
// required; it disables/restores interrupts internally otherwise.
uint32_t timestampTicks() {
  uint8_t oldSREG = SREG;
  cli();
  uint16_t tcnt = TCNT1;
  uint32_t ovf = timer1OverflowCount;
  // If an overflow happened but the ISR hasn't run yet (flag pending) and
  // TCNT1 already wrapped to a small value, account for it manually.
  if ((TIFR1 & _BV(TOV1)) && tcnt < 0x8000) {
    ovf++;
  }
  SREG = oldSREG;
  return (ovf << 16) | (uint32_t)tcnt;
}

inline float ticksToUs(uint32_t ticks) {
  return (float)ticks * 0.5f;
}

// ---------------------------------------------------------------------------
// Per-channel edge state, updated only inside the PCINT1 ISR.
// ---------------------------------------------------------------------------
enum ChannelState : uint8_t { NOT_STARTED, RISE_SEEN, FALL_SEEN };

volatile ChannelState ch1State = NOT_STARTED;
volatile ChannelState ch2State = NOT_STARTED;
volatile uint32_t echo1RiseTicks = 0;
volatile uint32_t echo1FallTicks = 0;
volatile uint32_t echo2RiseTicks = 0;
volatile uint32_t echo2FallTicks = 0;
volatile uint8_t previousPortC = 0;

// PCINT1 fires on ANY change of PC0..PC7 that is unmasked (we mask PC0, PC1).
// Keep this ISR minimal: no Serial, no float, no long math beyond what is here.
ISR(PCINT1_vect) {
  uint8_t currentPortC = PINC;
  uint8_t changed = currentPortC ^ previousPortC;
  uint32_t nowTicks = timestampTicks();

  if (changed & _BV(PC0)) { // ECHO1 edge
#if ECHO_ACTIVE_LOW
    bool active1 = !(currentPortC & _BV(PC0));
#else
    bool active1 = (currentPortC & _BV(PC0));
#endif
    if (active1) {
      if (ch1State == NOT_STARTED) {
        echo1RiseTicks = nowTicks;
        ch1State = RISE_SEEN;
      }
    } else {
      if (ch1State == RISE_SEEN) {
        echo1FallTicks = nowTicks;
        ch1State = FALL_SEEN;
      }
    }
  }

  if (changed & _BV(PC1)) { // ECHO2 edge
#if ECHO_ACTIVE_LOW
    bool active2 = !(currentPortC & _BV(PC1));
#else
    bool active2 = (currentPortC & _BV(PC1));
#endif
    if (active2) {
      if (ch2State == NOT_STARTED) {
        echo2RiseTicks = nowTicks;
        ch2State = RISE_SEEN;
      }
    } else {
      if (ch2State == RISE_SEEN) {
        echo2FallTicks = nowTicks;
        ch2State = FALL_SEEN;
      }
    }
  }

  previousPortC = currentPortC;
}

// ---------------------------------------------------------------------------
// Measurement state machine (section 16.2)
// ---------------------------------------------------------------------------
enum MeasureState : uint8_t { IDLE, WAITING, COMPLETE, TIMEOUT_STATE };

// History buffers (main-loop side only, no ISR access).
float deltaHistory[FILTER_WINDOW];
size_t deltaHistoryCount = 0;
size_t deltaHistoryIndex = 0;
float previousFilteredDeltaUs = 0.0f;
bool havePreviousFiltered = false;

bool outputCsv = true; // default: CSV, easy to parse for the 2D simulator

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
float medianOf(float *values, size_t n) {
  // Insertion sort on a small stack copy; n is at most FILTER_WINDOW (<=7).
  float sorted[FILTER_WINDOW];
  for (size_t i = 0; i < n; i++) sorted[i] = values[i];
  for (size_t i = 1; i < n; i++) {
    float key = sorted[i];
    long j = i - 1;
    while (j >= 0 && sorted[j] > key) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = key;
  }
  return sorted[n / 2];
}

void pushDeltaHistory(float value) {
  deltaHistory[deltaHistoryIndex] = value;
  deltaHistoryIndex = (deltaHistoryIndex + 1) % FILTER_WINDOW;
  if (deltaHistoryCount < FILTER_WINDOW) deltaHistoryCount++;
}

void resetChannelStates() {
  uint8_t oldSREG = SREG;
  cli();
  ch1State = NOT_STARTED;
  ch2State = NOT_STARTED;
  previousPortC = PINC & (_BV(PC0) | _BV(PC1));
  SREG = oldSREG;
}

void sendTriggerPulse() {
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
}

// ---------------------------------------------------------------------------
// Serial command parsing (section 23). Non-blocking line accumulation.
// ---------------------------------------------------------------------------
String cmdBuffer;

void printParameters() {
  Serial.println(F("--- parameters ---"));
  Serial.print(F("sound_speed_cm_per_us=")); Serial.println(SOUND_SPEED_CM_PER_US, 5);
  Serial.print(F("receiver_spacing_cm="));   Serial.println(RECEIVER_SPACING_CM, 3);
  Serial.print(F("calibration_offset_us=")); Serial.println(CALIBRATION_OFFSET_US, 3);
  Serial.print(F("output_mode="));           Serial.println(outputCsv ? F("CSV") : F("HUMAN"));
  Serial.println(F("------------------"));
}

void runCalibration() {
  Serial.println(F("# calibration: place symmetric reflector 30-50cm away, keep still"));
  float samples[CALIBRATION_SAMPLES];
  uint16_t collected = 0;
  uint16_t attempts = 0;
  const uint16_t MAX_ATTEMPTS = CALIBRATION_SAMPLES * 4;

  while (collected < CALIBRATION_SAMPLES && attempts < MAX_ATTEMPTS) {
    attempts++;
    resetChannelStates();
    delay(2); // guard interval so previous echo fully settles
    uint32_t triggerTicks = timestampTicks();
    sendTriggerPulse();

    MeasureState state = WAITING;
    while (state == WAITING) {
      uint32_t nowTicks;
      ChannelState s1, s2;
      uint8_t oldSREG = SREG;
      cli();
      nowTicks = timestampTicks();
      s1 = ch1State;
      s2 = ch2State;
      SREG = oldSREG;

      if (s1 == FALL_SEEN && s2 == FALL_SEEN) {
        state = COMPLETE;
      } else if (ticksToUs(nowTicks - triggerTicks) > MEASUREMENT_TIMEOUT_US) {
        state = TIMEOUT_STATE;
      }
    }

    if (state == COMPLETE) {
      uint32_t r1, f1, r2, f2;
      uint8_t oldSREG = SREG;
      cli();
      r1 = echo1RiseTicks; f1 = echo1FallTicks;
      r2 = echo2RiseTicks; f2 = echo2FallTicks;
      SREG = oldSREG;
      float pulse1 = ticksToUs(f1 - r1);
      float pulse2 = ticksToUs(f2 - r2);
      if (pulse1 >= MIN_VALID_US && pulse2 >= MIN_VALID_US) {
        samples[collected] = pulse2 - pulse1;
        collected++;
      }
    }
    delay(MEASUREMENT_INTERVAL_MS);
  }

  if (collected == 0) {
    Serial.println(F("# calibration failed: no valid samples"));
    return;
  }

  // Median of however many samples we actually collected.
  for (size_t i = 1; i < collected; i++) {
    float key = samples[i];
    long j = i - 1;
    while (j >= 0 && samples[j] > key) { samples[j + 1] = samples[j]; j--; }
    samples[j + 1] = key;
  }
  CALIBRATION_OFFSET_US = samples[collected / 2];

  Serial.print(F("# calibration done, samples="));
  Serial.print(collected);
  Serial.print(F(", offset_us="));
  Serial.println(CALIBRATION_OFFSET_US, 3);

  // Calibration invalidates the running delta history / cycle-slip reference.
  deltaHistoryCount = 0;
  deltaHistoryIndex = 0;
  havePreviousFiltered = false;
}

void handleCommand(const String &line) {
  if (line.length() == 0) return;
  char c = line.charAt(0);
  switch (c) {
    case 'c':
      runCalibration();
      break;
    case 'p':
      printParameters();
      break;
    case 'r':
      deltaHistoryCount = 0;
      deltaHistoryIndex = 0;
      havePreviousFiltered = false;
      Serial.println(F("# filter history reset"));
      break;
    case 's':
      outputCsv = !outputCsv;
      Serial.print(F("# output_mode="));
      Serial.println(outputCsv ? F("CSV") : F("HUMAN"));
      break;
    case 't': {
      float tempC = line.substring(1).toFloat();
      SOUND_SPEED_CM_PER_US = (331.3f + 0.606f * tempC) / 10000.0f; // m/s -> cm/us
      Serial.print(F("# sound_speed_cm_per_us="));
      Serial.println(SOUND_SPEED_CM_PER_US, 5);
      break;
    }
    case 'd': {
      float spacing = line.substring(1).toFloat();
      if (spacing > 0.0f) {
        RECEIVER_SPACING_CM = spacing;
        Serial.print(F("# receiver_spacing_cm="));
        Serial.println(RECEIVER_SPACING_CM, 3);
      }
      break;
    }
    default:
      Serial.println(F("# unknown command (use c/p/r/s/t<C>/d<cm>)"));
      break;
  }
}

void pollSerialCommands() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (cmdBuffer.length() > 0) {
        handleCommand(cmdBuffer);
        cmdBuffer = "";
      }
    } else {
      cmdBuffer += ch;
    }
  }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------
void printResult(uint32_t timestampMs, float pulse1Us, float pulse2Us,
                  float distance1Cm, float distance2Cm, float rawDeltaUs,
                  float calibratedDeltaUs, float pathDiffCm, bool angleValid,
                  float angleDeg, int confidence, const char *status,
                  int cycleShift) {
  if (outputCsv) {
    Serial.print(timestampMs); Serial.print(',');
    Serial.print(pulse1Us, 1); Serial.print(',');
    Serial.print(pulse2Us, 1); Serial.print(',');
    Serial.print(distance1Cm, 3); Serial.print(',');
    Serial.print(distance2Cm, 3); Serial.print(',');
    Serial.print(rawDeltaUs, 1); Serial.print(',');
    Serial.print(calibratedDeltaUs, 1); Serial.print(',');
    Serial.print(pathDiffCm, 3); Serial.print(',');
    if (angleValid) Serial.print(angleDeg, 2); else Serial.print(F("NA"));
    Serial.print(',');
    Serial.print(confidence); Serial.print(',');
    Serial.print(status); Serial.print(',');
    Serial.println(cycleShift);
  } else {
    Serial.print(F("R1=")); Serial.print(pulse1Us, 1); Serial.print(F("us "));
    Serial.print(distance1Cm, 2); Serial.print(F("cm | R2="));
    Serial.print(pulse2Us, 1); Serial.print(F("us "));
    Serial.print(distance2Cm, 2); Serial.print(F("cm | raw_dt="));
    Serial.print(rawDeltaUs, 1); Serial.print(F("us | calibrated_dt="));
    Serial.print(calibratedDeltaUs, 1); Serial.print(F("us | path="));
    Serial.print(pathDiffCm, 3); Serial.print(F("cm | angle="));
    if (angleValid) { Serial.print(angleDeg, 2); Serial.print(F("deg")); }
    else Serial.print(F("NA"));
    Serial.print(F(" | confidence=")); Serial.print(confidence);
    Serial.print(F(" | status=")); Serial.print(status);
    Serial.print(F(" | cycle_shift=")); Serial.println(cycleShift);
  }
}

void printCsvHeaderIfNeeded() {
  if (outputCsv) {
    Serial.println(F("timestamp_ms,r1_pulse_us,r2_pulse_us,r1_cm,r2_cm,raw_dt_us,"
                      "calibrated_dt_us,path_diff_cm,angle_deg,confidence,status,cycle_shift"));
  }
}

// ---------------------------------------------------------------------------
// setup / loop
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(ECHO1_PIN, INPUT);
  pinMode(ECHO2_PIN, INPUT);

  // Timer1: normal mode, prescaler 8 => 0.5 us/tick, enable overflow interrupt.
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;
  TCCR1B |= _BV(CS11);      // prescaler 8
  TIMSK1 |= _BV(TOIE1);     // enable Timer1 overflow interrupt

  // Pin Change Interrupt on Port C, pins PC0 (A0/ECHO1) and PC1 (A1/ECHO2).
  PCICR  |= _BV(PCIE1);
  PCMSK1 |= _BV(PCINT8) | _BV(PCINT9); // PCINT8=PC0, PCINT9=PC1
  previousPortC = PINC & (_BV(PC0) | _BV(PC1));
  interrupts();

  Serial.println(F("# ultrasonic_tdoa_nano ready"));
  Serial.println(F("# commands: c=calibrate p=params r=reset-filter s=toggle-csv t<C>=temp d<cm>=spacing"));
  printParameters();
  printCsvHeaderIfNeeded();
}

void loop() {
  pollSerialCommands();

  // --- arm and fire one measurement ---
  resetChannelStates();
  delay(2); // guard interval for residual ringing from the previous cycle
  uint32_t triggerTicks = timestampTicks();
  sendTriggerPulse();

  MeasureState state = WAITING;
  while (state == WAITING) {
    uint32_t nowTicks;
    ChannelState s1, s2;
    uint8_t oldSREG = SREG;
    cli();
    nowTicks = timestampTicks();
    s1 = ch1State;
    s2 = ch2State;
    SREG = oldSREG;

    if (s1 == FALL_SEEN && s2 == FALL_SEEN) {
      state = COMPLETE;
    } else if (ticksToUs(nowTicks - triggerTicks) > (float)MEASUREMENT_TIMEOUT_US) {
      state = TIMEOUT_STATE;
    }
  }

  uint32_t r1, f1, r2, f2;
  {
    uint8_t oldSREG = SREG;
    cli();
    r1 = echo1RiseTicks; f1 = echo1FallTicks;
    r2 = echo2RiseTicks; f2 = echo2FallTicks;
    SREG = oldSREG;
  }

  uint32_t timestampMs = millis();
  int confidence = 100;
  const char *status = "VALID";
  int cycleShift = 0;
  bool angleValid = false;
  float angleDeg = 0.0f;
  float pulse1Us = 0.0f, pulse2Us = 0.0f;
  float distance1Cm = 0.0f, distance2Cm = 0.0f;
  float rawDeltaUs = 0.0f, correctedDeltaUs = 0.0f, pathDiffCm = 0.0f;

  if (state == TIMEOUT_STATE) {
    confidence = 60; // 100 - 40 (section 21)
    status = "TIMEOUT";
  } else {
    pulse1Us = ticksToUs(f1 - r1);
    pulse2Us = ticksToUs(f2 - r2);
    distance1Cm = pulse1Us * SOUND_SPEED_CM_PER_US / 2.0f;
    distance2Cm = pulse2Us * SOUND_SPEED_CM_PER_US / 2.0f;
    rawDeltaUs = pulse2Us - pulse1Us;
    float calibratedDeltaUs = rawDeltaUs - CALIBRATION_OFFSET_US;

    float maxPulseUs = 2.0f * MAX_DISTANCE_CM / SOUND_SPEED_CM_PER_US;
    bool pulsesSane = (pulse1Us >= MIN_VALID_US) && (pulse2Us >= MIN_VALID_US) &&
                       (pulse1Us <= maxPulseUs) && (pulse2Us <= maxPulseUs);

    // --- cycle-slip detection against the median of recent corrected deltas ---
    correctedDeltaUs = calibratedDeltaUs;
    if (havePreviousFiltered) {
      float diff = calibratedDeltaUs - previousFilteredDeltaUs;
      float cycleCountF = roundf(diff / CYCLE_PERIOD_US);
      float residual = diff - cycleCountF * CYCLE_PERIOD_US;
      if (fabsf(residual) < CYCLE_SLIP_TOLERANCE_US && fabsf(cycleCountF) >= 1.0f) {
        cycleShift = -(int)cycleCountF;
        correctedDeltaUs = calibratedDeltaUs + (float)cycleShift * CYCLE_PERIOD_US;
      }
    }

    pathDiffCm = correctedDeltaUs * SOUND_SPEED_CM_PER_US;
    float maxDtUs = RECEIVER_SPACING_CM / SOUND_SPEED_CM_PER_US;
    bool withinPhysicalBound = fabsf(correctedDeltaUs) <= maxDtUs;

    if (!pulsesSane) {
      confidence -= 30;
      status = "INVALID_PULSE";
    } else if (!withinPhysicalBound) {
      confidence -= 30;
      status = "OUT_OF_BOUND";
    }
    if (cycleShift != 0) confidence -= 20;

    pushDeltaHistory(correctedDeltaUs);
    float filteredNow = medianOf(deltaHistory, deltaHistoryCount);
    float deviation = fabsf(correctedDeltaUs - filteredNow);
    if (deviation > 5.0f) confidence -= 15;

    float deadZoneMarginUs = 20.0f;
    if ((pulse1Us - MIN_VALID_US) < deadZoneMarginUs ||
        (pulse2Us - MIN_VALID_US) < deadZoneMarginUs) {
      confidence -= 10;
    }

    if (confidence < 0) confidence = 0;
    if (confidence > 100) confidence = 100;

    if (pulsesSane && withinPhysicalBound) {
      if (confidence < 50) status = "UNRELIABLE";
      else status = "VALID";

      if (strcmp(status, "VALID") == 0) {
        float ratio = pathDiffCm / RECEIVER_SPACING_CM;
        if (ratio > 1.0f) ratio = 1.0f;
        if (ratio < -1.0f) ratio = -1.0f;
        angleDeg = asinf(ratio) * 180.0f / (float)M_PI;
        angleValid = true;
      }
    }

    previousFilteredDeltaUs = filteredNow;
    havePreviousFiltered = true;
  }

  printResult(timestampMs, pulse1Us, pulse2Us, distance1Cm, distance2Cm,
              rawDeltaUs, correctedDeltaUs, pathDiffCm, angleValid, angleDeg,
              confidence, status, cycleShift);

  delay(MEASUREMENT_INTERVAL_MS);
}
