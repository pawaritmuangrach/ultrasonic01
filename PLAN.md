# แผนสร้าง Ultrasonic Array 40kHz: จาก 0 สู่ 3TX + 9RX บน ESP32

> เก็บสำเนาแผนที่อนุมัติแล้วไว้ในโปรเจกต์ (ต้นฉบับอยู่ที่ `C:\Users\User\.claude\plans\esp32-ads1115-valiant-octopus.md`)
> อัปเดตสถานะแต่ละ stage ในไฟล์นี้ได้เรื่อยๆ ระหว่างทำงานจริง

## Context

โปรเจกต์นี้เริ่มจากต้นแบบ 1 TX + 2 RX บน Arduino Nano ที่ใช้โมดูล US-015 เป็น black-box
receiver ผลที่ได้เป็นแค่ **pseudo-TDOA** (ความต่างของ *ความกว้างพัลส์* ECHO ระหว่างสองบอร์ด)
ไม่ใช่ TDOA จริง เพราะเข้าไม่ถึง raw waveform 40kHz — ตามที่ [`README.md`](README.md) §5 และ
[`US015_TDOA_Codex_Spec.md`](US015_TDOA_Codex_Spec.md) §8 ระบุไว้เอง

เป้าหมายใหม่คือ **ใช้ ultrasonic array แทน LiDAR สำหรับ obstacle avoidance ระยะใกล้ (<5-10 m)**
โดยเทรน NN ด้วยกล้องเป็น ground truth ผังสุดท้ายที่ออกแบบไว้คือ **3 TX + 9 RX แบบ Y-array
(รัศมี 40/80/140mm, แขน 120°) + MIMO coded TX = 27 เส้นทางเสมือน** ซึ่งต้องใช้ ESP32
เพราะ Nano จับ raw chirp 40kHz ไม่ได้

ผู้ใช้ **ยังไม่มีอุปกรณ์และเครื่องมือวัดใดๆ เลย** จะไปซื้อของวันนี้ แผนนี้จึงเริ่มจากรายการซื้อ
แล้วไล่เป็น stage ที่แต่ละ stage ต้องพิสูจน์ผ่านก่อนถึงจะไป stage ถัดไป

---

## ⚠️ 3 ข้อผิดพลาดในเอกสารเดิมที่แผนนี้แก้

| เอกสารเดิมบอก | ความจริง | ผลต่อการซื้อ/ออกแบบ |
|---|---|---|
| `README.md` §7.5: "ESP32 เดี่ยว → ADC1 8 channel" | โมดูล **WROOM-32 ไม่ต่อขา GPIO37/38 ออกมา** เหลือใช้จริง **6 ช่อง** (GPIO 32,33,34,35,36,39) | 6 ช่อง raw = RX near+mid 6 ตัวพอดี, far 3 ตัวต้องไปทางอื่น |
| `README.md` §7.5: far RX ใช้ ADS1115 envelope | ADS1115 860 SPS ÷ 4 ช่อง = **215 SPS/ช่อง = 4.6 ms/sample → error ~80 cm** ใช้ทำ TDOA ไม่ได้ | ใช้ **envelope → LM393 comparator → ขา digital** แทน (±2-5 µs ดีกว่า ~1000×) ซื้อ ADS1115 มาเทียบด้วยตามที่ต้องการ |
| `scaled_array_layout.html` BOM: "วงจร VREF **2.5V**" | 2.5V หมายถึงเลี้ยง op-amp ด้วย 5V → สัญญาณสวิงเกิน 3.3V → **ทำ ADC ของ ESP32 พัง** | เลี้ยง RX front-end ด้วย **3.3V, VREF = 1.65V** (ต้องใช้ op-amp rail-to-rail) |

**หลักคิดที่ทำให้ split raw/envelope นี้ถูกต้อง**: ความคลาดเคลื่อนเชิงมุม ≈ `c·σ_Δt / D`
ช่อง far มี baseline ยาวสุด (D=280mm) จึงทนความคลาดเคลื่อนเวลาได้มากที่สุด —
comparator ±5µs บน 280mm ให้ error เพียง **0.35°** ส่วนช่อง near baseline สั้น (80mm)
ต้องการ raw waveform + cross-correlation (±0.5µs → 0.12°) โครงนี้คือ
**coarse-to-fine interferometry**: near ให้คำตอบหยาบที่ไม่กำกวม → far ให้ความละเอียด

---

## Stage 0 — รายการซื้อ (BOM)

รายละเอียดเต็มอยู่ที่ [`hardware/BOM.md`](hardware/BOM.md) — สรุปกลยุทธ์:
ซื้อของที่ร้านไทยมีแน่ๆ วันนี้ (Stage 1-3 + ทรานสดิวเซอร์/passive ครบ 9 ช่อง)
แล้ว**สั่ง op-amp ตัวหลัก (MCP6024) ออนไลน์คืนนี้เลย** เพราะร้านไทยมักไม่มี
ระหว่างรอใช้ op-amp สำรอง (MCP6004) ที่ซื้อได้วันนี้ทำ Stage 1 ไปก่อน

**งบรวมโดยประมาณ 2,100 – 2,400฿**

**สถานะ:** ☐ ยังไม่ได้ซื้อ

---

## Stage 1 — ESP32 Scope + ยิงคลื่นได้ + รับ waveform ดิบ (1 TX + 1 RX)

**นี่คือ stage ที่เสี่ยงที่สุดของทั้งโปรเจกต์** ถ้า front-end ไม่ให้ waveform 40kHz ที่สะอาด
stage ที่เหลือไม่มีความหมายเลย ห้ามข้าม

### วงจร (บน breadboard)
```
[TCT40-16T] ←push-pull← 74HCT04 (3 gate ขนาน/ข้าง) ← GPIO25, GPIO26 (complementary จาก MCPWM)
                          ไฟ 5V + คาป 100µF//100nF ติดตัวชิป

[TCT40-16R] → C 10nF → [Stage1 gain ~40] → [Stage2 gain ~40] → R 100Ω → GPIO34 (ADC1_CH6)
                            ↑ bias VREF 1.65V              ↓ C 1nF ลงกราวด์ (anti-alias + กัน ADC)
             VREF = แบ่งแรงดัน 10k/10k จาก 3V3 + 10µF + บัฟเฟอร์ด้วย op-amp 1 section
```
**กฎเหล็ก**: ไฟ TX (5V) กับไฟ RX (3.3V) ต้องแยกเส้น กราวด์รวมกันจุดเดียว (star ground)
กระแสพุ่งตอนยิง TX เข้าไปกวนกราวด์ RX คือสาเหตุอันดับหนึ่งที่วงจรแบบนี้ไม่ทำงาน

รายละเอียดวงจรเต็ม (ค่า R/C ทุกตัว + การต่อบน breadboard ทีละแถว): [`hardware/rx_frontend.md`](hardware/rx_frontend.md), [`hardware/tx_driver.md`](hardware/tx_driver.md)

### ไฟล์ที่จะสร้าง
- `esp32_array/esp32_scope/esp32_scope.ino` — ยิง burst 40kHz N รอบ, เก็บ 1 ช่อง @ ~1 MS/s หน้าต่าง 10 ms, ส่ง binary ออก serial 921600
- `tools/scope_view.py` — pyserial + numpy + matplotlib: plot waveform / envelope / FFT
- `tools/requirements.txt`

### เกณฑ์ผ่าน
1. เห็น waveform ไซน์ 40kHz ชัด (FFT มี peak ที่ 40k ±1k)
2. แยก **ringing ของตัวส่ง** (ตอนต้น) ออกจาก **echo** (ตอนหลัง) ได้ด้วยตา
3. ขยับแผ่นสะท้อน 20→50 cm แล้วตำแหน่ง echo เลื่อนตามสูตร `t = 2d/c`
4. SNR ของ echo ที่ 50 cm ≥ 10 dB

**สถานะ:** ☐ ยังไม่เริ่ม

---

## Stage 2 — TDOA จริงด้วย Cross-Correlation (1 TX + 2 RX)

จุดที่โปรเจกต์เปลี่ยนจาก pseudo-TDOA เป็น **true TDOA** — ทำสิ่งที่
`US015_TDOA_Codex_Spec.md` §27 เขียนไว้ว่าเป็นทางที่ถูกต้อง

- ขยาย ADC เป็น 2 ช่อง (GPIO34, 35) — ESP32 **สลับอ่านทีละช่อง ไม่ใช่พร้อมกัน**
  แต่ **skew คงที่และรู้ค่าแน่นอน** (= 1/sample_rate ต่อสลอต) จึงหักออกได้แม่นยำ
  ต่างจาก jitter ที่หักไม่ได้ — นี่คือเหตุผลที่ ADC มัลติเพล็กซ์ใช้ทำ TDOA ได้
- Python: bandpass → **cross-correlation** → parabolic interpolation รอบ peak → Δt ระดับ sub-sample
- ใช้ **envelope ให้ค่าหยาบที่ไม่กำกวม** แล้ว **phase refine ให้ละเอียด** — แก้ปัญหา cycle-slip
  25 µs ที่ทรมานต้นแบบ Nano มาตลอด (`README.md` §7.7)

### เกณฑ์ผ่าน
- σ(Δt) < 1 µs เมื่อทุกอย่างอยู่นิ่ง (เทียบต้นแบบเดิมที่กระโดดเป็นก้อน 25 µs)
- เลื่อนแผ่นสะท้อนซ้าย/กลาง/ขวา แล้วเครื่องหมายและขนาดของมุมเปลี่ยนตามจริง
- มุมที่คำนวณได้ตรงกับมุมที่วัดด้วยไม้บรรทัด/ไม้โปรแทรกเตอร์ ±3°

**สถานะ:** ☐ ยังไม่เริ่ม

---

## Stage 3 — ขยายเป็น 6 ช่อง raw + คาลิเบรต skew ระหว่างช่อง

- สร้าง RX front-end **6 ช่องที่เหมือนกันทุกช่อง** (MCP6024 3 ชิป) → GPIO 32,33,34,35,36,39
- rate รวม 2 MS/s ÷ 6 ช่อง = **333 kS/s/ช่อง = 8.3 sample ต่อคาบ 40kHz** เพียงพอ
- Buffer: 6 ช่อง × 333 kS/s × 10 ms × 2 byte = **40 KB** (ESP32 มี DRAM 320 KB) พอ
- **ขั้นตอนคาลิเบรตสำคัญ**: วางทรานสดิวเซอร์รับทั้ง 6 ตัวเป็นแนวเดียวกัน ระยะเท่ากันจาก TX
  แล้วเก็บ 500 ping → หา offset ต่อช่อง (รวม skew ของ ADC + ความต่างของ front-end)
  เก็บลง NVS ของ ESP32

### เกณฑ์ผ่าน
- หลังคาลิเบรต ทุกช่องรายงาน Δt = 0 ±1 µs ในผังสมมาตร
- ไม่มีอาการ crosstalk (ปิดบังช่องหนึ่ง แล้วช่องอื่นต้องไม่เปลี่ยน)

**สถานะ:** ☐ ยังไม่เริ่ม

---

## Stage 4 — แผง Y-array 1 TX + 6 RX + DOA 2 แกน

พิกัดจาก `simulation/front_array_plate_layout.html` (origin = TX กลางแผง, mm):

| id | x | y | | id | x | y |
|---|---|---|---|---|---|---|
| T | 0.0 | 0.0 | | R4 | −69.3 | −40.0 |
| R1 | 0.0 | 40.0 | | R5 | 34.6 | −20.0 |
| R2 | 0.0 | 80.0 | | R6 | 69.3 | −40.0 |
| R3 | −34.6 | −20.0 | | | | |

แผง **190 × 170 mm, เจาะ 7 รู ⌀16mm**

- ตัดแผง: อะคริลิก 3mm เลเซอร์ (ส่งไฟล์ DXF ให้ร้าน ~250฿) หรือพลาสวูด/ไม้อัด + โฮลซอว์ 16mm
- generate ไฟล์ **SVG/DXF drill template** จากพิกัดข้างบน
- Firmware/Python: จาก Δt หลายคู่ → **least-squares DOA** ได้ทั้ง azimuth และ elevation
  (แขน 120° ให้ข้อมูลสองแกนพร้อมกัน)
- **ต้องเขียน viewer ใหม่**: `simulation/index.html` ปัจจุบันฮาร์ดโค้ดไว้กับ 2 ช่องทั้งหมด —
  `parseCsvLine()` ล็อกที่ 12 คอลัมน์, `buildShot()` ใช้ `intersectCircles()` สองวงกลม,
  และสมมติว่า TX อยู่ตำแหน่งเดียวกับ R1 ซึ่งไม่จริงสำหรับผัง Y-array
  → สร้าง `simulation/array_viewer.html` ใหม่ (parser อ่านจาก header, ตาราง RX เป็น (x,y),
  multilateration แทน two-circle) เก็บของเดิมไว้ไม่แตะ

**สถานะ:** ☐ ยังไม่เริ่ม

---

## Stage 5 — เพิ่มช่อง far 3 ตัว → ครบ 9 RX

- ช่อง far (R3/R6/R9 ที่ 140mm): `RX → gain 2 สเตจ → 1N4148 + RC (τ≈100µs) envelope
  → LM393 เทียบกับ trimpot → GPIO (pullup 3.3V)`
- ESP32 จับเวลาขอบด้วย **GPIO interrupt + `esp_timer_get_time()`** หรือ MCPWM capture (แม่นกว่า)
- **การทดลองเปรียบเทียบ**: ต่อ ADS1115 ขนานกับช่อง far หนึ่งช่อง แล้ววัดพร้อมกัน —
  จะเห็นด้วยตัวเลขจริงว่า ADS1115 ให้ ~4.6 ms/sample เทียบกับ LM393 ที่ ~3 µs
  (เก็บผลไว้เขียนลง README เป็นข้อสรุปที่พิสูจน์แล้ว)
- ขยายแผงเป็น **⌀316mm, 12 รู** ตามพิกัดใน `simulation/scaled_array_layout.html`
  (θ_res เป้าหมาย 1.75° ที่ D=280mm)

**สถานะ:** ☐ ยังไม่เริ่ม

---

## Stage 6 — MIMO: 3 TX coded ยิงพร้อมกัน

- 3 TX ที่รัศมี 14mm รอบศูนย์กลาง (มุม 90/210/330°) = ห่างกันเอง 24.25mm (ตัวถัง 16mm ไม่ชน)
- **โค้ดที่จะใช้**: BPSK ด้วย **Golay complementary pair** หรือ **Hadamard row** ความยาว 8-32 ชิป
  บนพาหะ 40kHz (เอกสารเดิมบอกแค่ "matched filter" ไม่ได้ระบุโค้ด)
- อัปเกรด driver เป็น **TC4428A ที่ 12V** (24Vpp differential) เพื่อระยะที่ไกลขึ้น
- ได้ 3 TX × 9 RX = **27 เส้นทาง** virtual position = TXi + RXj
- ⚠️ ข้อควรรู้ล่วงหน้า: TCT40-16 เป็นทรานสดิวเซอร์ **แถบแคบ (Q สูง ~2 kHz bandwidth)**
  ดังนั้น chirp/pulse compression จะ**ไม่**ช่วยเรื่อง range resolution มากอย่างที่หวัง
  (c/2B ≈ 8.6 cm) ประโยชน์จริงที่ได้คือ **correlation gain (SNR) และการแยก TX 3 ตัวออกจากกัน**
  ซึ่งก็คุ้มอยู่ดี แต่ต้องคาดหวังให้ถูก

**สถานะ:** ☐ ยังไม่เริ่ม

---

## Stage 7 — Camera ground truth + Neural Network

- ตามที่ `sparse_array_layout.html` วางไว้: กล้องมองตั้งฉากลงมาจากไตรพอด สูง 50-70 cm
  ต้องมี **marker อย่างน้อย 4 จุดที่รู้พิกัดจริง (มุมกระดาษ 4 มุม) เพื่อทำ homography pixel→mm**
- เก็บ dataset: raw echo 9 ช่อง × 27 เส้นทาง + label ตำแหน่งจากกล้อง
- เทรน NN แบบ learned beamforming / deep DOA

**สถานะ:** ☐ ยังไม่เริ่ม

---

## ไฟล์ที่จะสร้าง/แก้ (สรุป)

| ไฟล์ | Stage | สถานะ |
|---|---|---|
| `hardware/BOM.md` | 0 | ✅ สร้างแล้ว |
| `hardware/rx_frontend.md` | 1 | ☐ กำลังทำ |
| `hardware/tx_driver.md` | 1 | ☐ รอคิว |
| `esp32_array/esp32_scope/esp32_scope.ino` | 1 | ☐ รอคิว |
| `tools/scope_view.py`, `tools/requirements.txt` | 1 | ☐ รอคิว |
| `esp32_array/esp32_tdoa/esp32_tdoa.ino` | 2→6 | ☐ ยังไม่เริ่ม |
| `tools/tdoa_process.py` (xcorr + DOA) | 2→5 | ☐ ยังไม่เริ่ม |
| `hardware/plate_1tx6rx.svg` / `.dxf` | 4 | ☐ ยังไม่เริ่ม |
| `hardware/plate_3tx9rx.svg` / `.dxf` | 5 | ☐ ยังไม่เริ่ม |
| `simulation/array_viewer.html` | 4 | ☐ ยังไม่เริ่ม |
| `README.md` | ทุก stage | ☐ ยังไม่อัปเดต — ต้องเพิ่มหัวข้อ 8 "ESP32 build log" + แก้ 3 จุดผิดในตาราง §7.5 |
| `ultrasonic_tdoa_nano/`, `diagnostics/`, `simulation/*.html` เดิม | — | **ไม่แตะ** เก็บเป็นบันทึกงานเฟส Nano |

---

## Verification

**Stage 1** (ทำได้ทันทีที่ของมาถึง):
```bash
python tools/scope_view.py --port COM3 --baud 921600
```
ต้องเห็นกราฟ 3 แผง: waveform ดิบ / envelope / FFT — FFT ต้องมี peak เดียวชัดที่ 40 kHz
แล้วขยับแผ่นสะท้อนจาก 20 → 50 cm ตำแหน่ง echo ต้องเลื่อนจาก ~1.17 ms → ~2.92 ms

**Stage 2-3**: สคริปต์ `tdoa_process.py --calibrate` เก็บ 500 ping แล้วรายงาน
median / std / จำนวน cluster ของ Δt ต่อช่อง — ต้องมี **cluster เดียว** (ต้นแบบ Nano เดิมมี 6 cluster)

**Stage 4-5**: วางวัตถุที่มุมรู้ค่า (0°, ±15°, ±30°, ±45°) ด้วยไม้โปรแทรกเตอร์
เทียบกับมุมที่ระบบรายงาน → ต้องได้ RMS error < 3°

**ทุก stage**: ใช้ `esp32_scope` เป็นเครื่องมือวัด — ก่อนโทษซอฟต์แวร์ ให้ดู waveform ดิบก่อนเสมอ

---

## ลำดับงานถัดไป

1. ✅ `hardware/BOM.md` — รายการซื้อของ
2. ☐ `hardware/rx_frontend.md` + `tx_driver.md` — วงจรพร้อมค่า R/C ทุกตัว
3. ☐ `esp32_scope.ino` + `scope_view.py` — เครื่องมือวัด
4. ☐ รอผลการต่อวงจรจริง → ดีบักร่วมกัน → ไป Stage 2
