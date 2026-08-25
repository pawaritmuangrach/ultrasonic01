# ข้อกำหนดสำหรับพัฒนาโค้ด Arduino Nano: ระบบ Ultrasonic 40 kHz แบบ 1 Transmitter + 2 Receivers ด้วย US-015

## 1. วัตถุประสงค์ของโปรเจกต์

ต้องการทดลองใช้โมดูลอัลตราโซนิก US-015 จำนวน 2 บอร์ด เพื่อสร้างระบบที่มี:

- ตัวส่งคลื่นอัลตราโซนิก 40 kHz จำนวน 1 ตัว
- ตัวรับคลื่นอัลตราโซนิกจำนวน 2 ตัว
- Arduino Nano รุ่นคลาสสิกที่ใช้ ATmega328P
- ใช้ความต่างของเวลาที่ตัวรับทั้งสองตรวจพบสัญญาณ เพื่อประมาณความต่างของเส้นทางเสียง และอาจต่อยอดไปสู่การประมาณทิศทางของแหล่งสะท้อนหรือ TDOA

โครงการนี้เป็นการทดลองดัดแปลง US-015 ซึ่งเดิมถูกออกแบบให้แต่ละบอร์ดทำงานเป็นเซนเซอร์วัดระยะอิสระ ไม่ได้ออกแบบมาเป็นระบบ TDOA หลายตัวรับโดยตรง ดังนั้นโค้ดต้องแยกให้ชัดเจนระหว่าง:

1. ค่าที่วัดได้จริงจากโมดูล
2. ค่าที่อนุมานจากโมดูล
3. ข้อจำกัดที่ทำให้ค่าดังกล่าวยังไม่ใช่ TDOA บริสุทธิ์

---

## 2. ฮาร์ดแวร์และการต่อปัจจุบัน

สมมติการต่อดังนี้:

```text
US-015 บอร์ด 1
- VCC  -> 5V
- GND  -> GND
- TRIG -> A2
- ECHO -> A0
- มีหัวส่ง T
- มีหัวรับ R1

US-015 บอร์ด 2
- VCC  -> 5V
- GND  -> GND
- TRIG -> A2
- ECHO -> A1
- ถอดหรือไม่ใช้งานหัวส่ง T
- ใช้หัวรับ R2
```

ขา Analog ของ Arduino Nano ถูกใช้เป็นขา Digital:

```cpp
A0 = ECHO1
A1 = ECHO2
A2 = TRIG
```

แม้สายจะต่อเข้าขาที่มีชื่อ A0, A1 และ A2 แต่ในงานนี้ไม่ควรใช้ `analogRead()` กับ ECHO เนื่องจาก ECHO ของ US-015 เป็นสัญญาณดิจิทัล LOW/HIGH ไม่ใช่แรงดัน Analog ที่แสดงรูปคลื่น 40 kHz

---

## 3. ทฤษฎีของคลื่นอัลตราโซนิก 40 kHz

### 3.1 ความถี่และคาบ

ความถี่ของทรานสดิวเซอร์:

\[
f = 40{,}000\ \text{Hz}
\]

คาบของคลื่น:

\[
T = \frac{1}{f}
\]

ดังนั้น:

\[
T = \frac{1}{40{,}000} = 25\ \mu s
\]

ความหมายสำคัญคือ หากวงจรตรวจจับพลาดคลื่นไปหนึ่งรอบ เวลาที่วัดอาจกระโดดประมาณ 25 ไมโครวินาที หากพลาดสองหรือสามรอบ อาจกระโดดประมาณ 50 หรือ 75 ไมโครวินาที

ค่าที่พบจาก Serial Monitor มีการกระโดดเป็นกลุ่มใกล้เคียงกับหลายเท่าของ 25 ไมโครวินาที เช่น:

```text
+19 us
-56 us
-77 us
0 us
+55 us
+75 us
```

จึงมีความเป็นไปได้ว่าระบบตรวจพบคนละ cycle หรือคนละ echo path ในแต่ละรอบ

### 3.2 ความเร็วเสียง

ที่อุณหภูมิห้องโดยประมาณ สามารถใช้:

\[
c \approx 343\ \text{m/s}
\]

หรือ:

\[
c \approx 0.0343\ \text{cm}/\mu s
\]

ค่าความเร็วเสียงเปลี่ยนตามอุณหภูมิ โดยประมาณ:

\[
c \approx 331.3 + 0.606T
\]

เมื่อ \(T\) คืออุณหภูมิอากาศหน่วยองศาเซลเซียส

ตัวอย่าง:

```text
ที่ 20 °C -> c ≈ 343.4 m/s
ที่ 30 °C -> c ≈ 349.5 m/s
```

สำหรับการทดลองระยะสั้น อาจใช้ 343 m/s ได้ก่อน แต่ถ้าต้องการความแม่นยำ ควรรับค่าอุณหภูมิเป็นตัวแปรและปรับความเร็วเสียงตามอุณหภูมิ

### 3.3 ความยาวคลื่น

\[
\lambda = \frac{c}{f}
\]

เมื่อ:

```text
c = 343 m/s
f = 40,000 Hz
```

จะได้:

\[
\lambda \approx 8.575\ \text{mm}
\]

ดังนั้นการเปลี่ยนตำแหน่งทรานสดิวเซอร์เพียงไม่กี่มิลลิเมตรก็อาจเปลี่ยนเฟสและรูปแบบการแทรกสอดได้อย่างมีนัยสำคัญ

---

## 4. ความแตกต่างระหว่าง Distance Measurement และ TDOA

### 4.1 การวัดระยะด้วย pulse width

โมดูลอัลตราโซนิกทั่วไปส่งพัลส์ ECHO ซึ่งความกว้างแทนเวลาที่คลื่นเดินทาง:

```text
ตัวส่ง -> วัตถุ -> ตัวรับ
```

ระยะทางไปยังวัตถุจึงคำนวณโดย:

\[
d = \frac{ct}{2}
\]

ต้องหาร 2 เพราะเสียงเดินทางไปและกลับ

ในโค้ดปัจจุบัน:

```cpp
distanceCm = pulseWidthUs * 0.0343 / 2.0;
```

สูตรนี้เหมาะกับการวัดระยะจากตัวส่งไปวัตถุแล้วสะท้อนกลับมายังตัวรับ

### 4.2 การวัดความต่างเส้นทางด้วย TDOA

กรณีตัวรับสองตัวรับคลื่นเดียวกัน:

\[
\Delta t = t_2 - t_1
\]

ความต่างของเส้นทางเสียงคือ:

\[
\Delta L = c\Delta t
\]

กรณีนี้ไม่หาร 2 เพราะกำลังเปรียบเทียบเส้นทางที่คลื่นไปถึงตัวรับสองตัว ไม่ได้คำนวณการเดินทางไป-กลับของวัตถุหนึ่งจุดโดยตรง

ตัวอย่าง:

```text
Δt = 18 us
c = 0.0343 cm/us
```

จะได้:

\[
\Delta L = 18 \times 0.0343 = 0.6174\ \text{cm}
\]

ประมาณ 0.62 cm

---

## 5. ทำไม `analogRead()` ไม่เหมาะกับงานนี้

Arduino Nano รุ่น ATmega328P มี ADC เพียงชุดเดียว แล้วใช้ Multiplexer สลับอ่าน A0, A1 และ A2 ทีละช่อง

โดยค่าเริ่มต้นของ Arduino:

```text
CPU clock       = 16 MHz
ADC prescaler   = 128
ADC clock       = 125 kHz
1 conversion    ≈ 13 ADC clocks
เวลาอ่านหนึ่งครั้ง ≈ 104 us
```

แต่คลื่น 40 kHz มีคาบเพียง 25 us

ดังนั้นระหว่างการอ่าน A0 และ A1 คลื่นอาจผ่านไปแล้วประมาณ 4 รอบ และระหว่าง A0 ถึง A2 อาจผ่านไปประมาณ 8 รอบ

ปัญหาหลักไม่ใช่แค่ว่า ADC ช้า แต่คือ `analogRead()` คืนค่าแรงดัน ณ เวลาหนึ่ง ไม่ได้คืน timestamp ที่สัญญาณเริ่มมาถึง

ยิ่งไปกว่านั้น ขา ECHO ของ US-015 เป็นสัญญาณดิจิทัล:

```text
LOW  ≈ 0 V
HIGH ≈ 5 V
```

ดังนั้น `analogRead()` จะได้เพียงค่าประมาณ:

```text
LOW  -> 0
HIGH -> 1023
```

จึงไม่มีประโยชน์ในการอ่านรูปคลื่น 40 kHz จริง

วิธีที่เหมาะกว่าคือใช้:

- Digital input
- Pin Change Interrupt
- Timer1
- จับขอบ LOW -> HIGH และ HIGH -> LOW
- บันทึก timestamp ด้วยความละเอียดระดับไมโครวินาทีหรือต่ำกว่า

---

## 6. Timer1 และความละเอียดเวลา

Arduino Nano ใช้ clock 16 MHz

หากตั้ง Timer1 prescaler = 8:

\[
f_{timer} = \frac{16\text{ MHz}}{8} = 2\text{ MHz}
\]

ดังนั้นหนึ่ง tick เท่ากับ:

\[
\frac{1}{2\text{ MHz}} = 0.5\ \mu s
\]

ค่าความละเอียด 0.5 us เทียบเป็นความต่างเส้นทางได้:

\[
0.5 \times 0.0343 = 0.01715\ \text{cm}
\]

หรือประมาณ 0.1715 mm ในเชิง resolution ของ timer

อย่างไรก็ตาม resolution ของ timer ไม่เท่ากับ accuracy ของระบบจริง เพราะยังมีความผิดพลาดจาก:

- ความหน่วงภายในโมดูล
- Comparator threshold
- jitter
- interrupt latency
- crosstalk
- ringing
- multipath
- cycle ambiguity
- ความคลาดเคลื่อนของทรานสดิวเซอร์
- อุณหภูมิ
- การวางตำแหน่งทางกล

ดังนั้นแม้ timer จะละเอียด 0.5 us แต่ความแม่นยำจริงอาจแย่กว่ามาก

---

## 7. Pin Change Interrupt

A0 และ A1 อยู่บน Port C ของ ATmega328P:

```text
A0 = PC0 = PCINT8
A1 = PC1 = PCINT9
```

ทั้งสองขาสามารถใช้ Pin Change Interrupt กลุ่มเดียวกันได้

เมื่อ A0 หรือ A1 เปลี่ยนสถานะ ISR เดียวจะทำงาน แล้วอ่านค่า `PINC` เพื่อดูว่าขาใดเปลี่ยน

แนวคิด:

```cpp
currentPort = PINC;
changed = currentPort ^ previousPort;
```

หาก bit ของ PC0 เปลี่ยน:

```text
LOW -> HIGH = บันทึก rise timestamp ของ R1
HIGH -> LOW = บันทึก fall timestamp ของ R1
```

หาก bit ของ PC1 เปลี่ยน:

```text
LOW -> HIGH = บันทึก rise timestamp ของ R2
HIGH -> LOW = บันทึก fall timestamp ของ R2
```

ข้อดี:

- ตรวจทั้งสองช่องใน interrupt group เดียว
- ไม่ต้อง polling แบบช้า
- ลดความคลาดเคลื่อนจากการเรียกฟังก์ชันทีละช่อง
- เก็บ timestamp ด้วย Timer1 เดียวกัน

---

## 8. สิ่งที่โค้ดปัจจุบันวัดได้จริง

โค้ดปัจจุบันคำนวณ:

```cpp
pulse1Us = fall1 - rise1;
pulse2Us = fall2 - rise2;
deltaTimeUs = pulse2Us - pulse1Us;
```

ความหมายจริงคือ:

```text
deltaTimeUs = ความต่างของความกว้างพัลส์ ECHO ระหว่าง US-015 สองบอร์ด
```

มันยังไม่สามารถรับประกันได้ว่าเป็น:

```text
เวลาที่คลื่นมาถึง Receiver 2 - เวลาที่คลื่นมาถึง Receiver 1
```

เพราะ US-015 แต่ละบอร์ดมีวงจรภายในอิสระ เช่น:

- วงจรสร้าง burst
- blanking time หลังส่ง
- automatic gain หรือ gain profile
- amplifier
- envelope detector
- comparator
- threshold
- logic timing
- pulse generation

หากวงจรภายในของสองบอร์ดไม่เท่ากัน ความกว้าง ECHO จะมี offset และ jitter ที่ไม่เกี่ยวกับ propagation time

ดังนั้นต้องเรียกผลปัจจุบันว่า:

```text
pseudo-TDOA
หรือ
difference of module-reported echo pulse widths
```

ไม่ควรเรียกว่า true TDOA จนกว่าจะพิสูจน์ด้วยการ calibrate และทดสอบซ้ำ

---

## 9. ปัญหาที่เห็นจากข้อมูลจริง

ตัวอย่างข้อมูล:

```text
R1: 212.5 us | R2: 230.5 us | dt: +18.0 us
R1: 287.5 us | R2: 231.0 us | dt: -56.5 us
R1: 308.5 us | R2: 231.0 us | dt: -77.5 us
R1: 250.0 us | R2: 250.0 us | dt: 0.0 us
R1: 193.5 us | R2: 268.5 us | dt: +75.0 us
```

หากตำแหน่งอุปกรณ์และวัตถุไม่ขยับ ค่า dt ควรเกาะกลุ่มใกล้ค่าเดียวกัน แต่ข้อมูลนี้แบ่งเป็นหลายกลุ่ม:

```text
ประมาณ +19 us
ประมาณ -56 us
ประมาณ -77 us
ประมาณ 0 us
ประมาณ +55 us
ประมาณ +75 us
```

ความต่างระหว่างหลายกลุ่มใกล้เคียงกับ:

```text
25 us
50 us
75 us
```

ซึ่งสัมพันธ์กับคาบของ 40 kHz

จึงมีสมมติฐานสำคัญว่าเกิด cycle slip หรือ cycle ambiguity:

```text
บอร์ดหนึ่งจับ cycle แรก
อีกบอร์ดจับ cycle ที่สองหรือสาม
```

นอกจากนี้ระยะที่รายงานเพียงประมาณ 3-5 cm ใกล้กับบริเวณที่อาจได้รับผลจาก:

- direct acoustic coupling
- transmitter ringing
- mechanical vibration
- board crosstalk
- dead zone
- near-field pattern
- reflection จากโต๊ะหรือโครงยึด

---

## 10. ขอบเขตทางกายภาพของ TDOA

หาก Receiver สองตัวห่างกัน \(D\):

\[
|\Delta L| \le D
\]

เพราะความต่างของระยะทางไปยังตัวรับสองตัวไม่สามารถมากกว่าระยะห่างระหว่างตัวรับได้

ดังนั้น:

\[
|\Delta t| \le \frac{D}{c}
\]

ตัวอย่าง Receiver ห่างกัน 2 cm:

\[
|\Delta t|_{\max} = \frac{2}{0.0343}
\]

\[
|\Delta t|_{\max} \approx 58.3\ \mu s
\]

ดังนั้นค่า dt = 75 us จะเป็นไปไม่ได้ทางเรขาคณิต หาก Receiver ห่างกันเพียง 2 cm

โค้ดควรมี plausibility gate:

```cpp
maxDtUs = receiverSpacingCm / soundSpeedCmPerUs;

if (abs(deltaTimeUs) > maxDtUs) {
    rejectMeasurement();
}
```

---

## 11. ทฤษฎีการประมาณมุมด้วย Receiver สองตัว

สมมติ Receiver สองตัววางเป็นเส้นตรง ระยะห่าง \(D\) และแหล่งกำเนิดอยู่ไกลพอจนคลื่นประมาณเป็น plane wave

\[
\Delta L = D\sin\theta
\]

ดังนั้น:

\[
\theta = \sin^{-1}\left(\frac{c\Delta t}{D}\right)
\]

โดย:

- \(\theta = 0^\circ\) หมายถึงคลื่นมาจากด้านหน้าตรงกลาง
- ค่าเป็นบวกหรือลบขึ้นกับนิยามว่า R1 หรือ R2 อยู่ด้านใด
- ต้องกำหนด sign convention ให้ชัดเจน

ตัวอย่าง:

```text
Receiver spacing D = 5 cm
Δt = 50 us
c = 0.0343 cm/us
```

\[
\Delta L = 50 \times 0.0343 = 1.715\ \text{cm}
\]

\[
\theta = \sin^{-1}(1.715/5)
\]

\[
\theta \approx 20.1^\circ
\]

ข้อจำกัด:

- สูตรนี้เหมาะกับ far-field หรือ plane-wave approximation
- หากวัตถุอยู่ใกล้มาก geometry จะเป็น hyperbolic ไม่ใช่เส้นตรงง่าย ๆ
- สำหรับระบบสะท้อนแบบ monostatic/bistatic ความสัมพันธ์จะซับซ้อนกว่ากรณีแหล่งเสียงตรง
- หากใช้ transmitter อยู่ระหว่าง receivers ต้องพิจารณาเส้นทาง TX -> target -> RX1 และ TX -> target -> RX2

---

## 12. Near-field และ Far-field

ทรานสดิวเซอร์อัลตราโซนิกมีขนาด aperture จำกัด และเกิด near-field interference pattern

ความยาว near-field โดยประมาณของ circular transducer:

\[
N \approx \frac{a^2}{\lambda}
\]

หรือบางนิยามใช้:

\[
N \approx \frac{D_t^2}{4\lambda}
\]

เมื่อ:

- \(a\) คือรัศมีของทรานสดิวเซอร์
- \(D_t\) คือเส้นผ่านศูนย์กลางของทรานสดิวเซอร์
- \(\lambda\) คือความยาวคลื่น

ใน near field ความดันเสียงอาจมีจุดสูงและต่ำหลายตำแหน่ง ทำให้ Receiver สองตัวที่อยู่ใกล้กันรับ amplitude และ phase ต่างกันมาก แม้ว่าวัตถุจะอยู่ตรงกลาง

ดังนั้นการทดสอบควรเริ่มที่ระยะประมาณ 20-50 cm กับแผ่นสะท้อนขนาดใหญ่และผิวเรียบ ก่อนทดสอบระยะใกล้

---

## 13. Ringing และ Blanking Time

หลังส่ง burst 40 kHz ตัวส่งจะยังสั่นต่อช่วงหนึ่ง เรียกว่า ringing

ผลของ ringing:

- ตัวรับที่อยู่ใกล้อาจรับคลื่นตรงทันที
- โครงสร้างบอร์ดหรือฐานยึดอาจส่งแรงสั่นทางกล
- วงจรรับอาจ saturate
- Comparator อาจ trigger ก่อน echo จริง
- โมดูลต้องใช้ blanking interval เพื่อไม่รับสัญญาณช่วงแรก

US-015 แต่ละบอร์ดอาจมี blanking time ไม่เท่ากันเล็กน้อย ทำให้สองบอร์ดเริ่มรับจริงคนละเวลา

โค้ดควรมี minimum valid time:

```cpp
if (pulseWidthUs < MIN_VALID_US) {
    reject as ringing or direct coupling;
}
```

แต่ต้องระวังว่า pulse width ไม่เท่ากับ absolute time of arrival หากใช้ output ECHO ของโมดูล

---

## 14. Threshold Walk

Comparator ตรวจพบสัญญาณเมื่อ amplitude ข้าม threshold

หากสัญญาณแรง:

```text
ข้าม threshold เร็ว
```

หากสัญญาณอ่อน:

```text
ข้าม threshold ช้า
```

ปรากฏการณ์นี้เรียกว่า threshold walk หรือ time walk

ดังนั้น Receiver ที่รับคลื่นแรงกว่าอาจดูเหมือนรับก่อน แม้เวลาที่ wavefront มาถึงจริงเท่ากัน

วิธีลดผลกระทบ:

- ใช้ constant fraction discriminator
- ใช้ matched filtering
- บันทึกรูปคลื่น analog แล้วทำ cross-correlation
- ใช้ threshold หลายระดับ
- calibrate amplitude-time relation
- ใช้ comparator และ gain ที่เหมือนกันทุกช่อง
- ใช้ front-end ร่วม clock และ supply เดียวกัน

US-015 แบบ black box ไม่เปิดโอกาสให้ควบคุมองค์ประกอบเหล่านี้ได้เต็มที่

---

## 15. Multipath

คลื่นอัลตราโซนิกอาจสะท้อนจาก:

- โต๊ะ
- ผนัง
- มือ
- สายไฟ
- ฐานยึด
- ขอบวัตถุ
- ตัวบอร์ดเอง

Receiver อาจตรวจพบเส้นทางที่ไม่ใช่เส้นทางตรง เช่น:

```text
TX -> โต๊ะ -> R1
TX -> วัตถุ -> R2
```

หรือ:

```text
TX -> วัตถุหลัก -> R1
TX -> ผนัง -> R2
```

ทำให้ dt เปลี่ยนเครื่องหมายหรือกระโดดเป็นหลายกลุ่ม

ควรทดสอบในพื้นที่โล่งและใช้วัสดุดูดซับเสียงรอบด้านที่ไม่ต้องการสะท้อน หากทำได้

---

## 16. แนวทางซอฟต์แวร์ที่ Codex ต้องพัฒนา

### 16.1 เป้าหมายหลัก

สร้าง Arduino sketch ที่:

1. ใช้ Arduino Nano ATmega328P
2. ใช้ A0 และ A1 เป็น digital input
3. ใช้ A2 เป็น trigger output
4. ใช้ Timer1 prescaler 8
5. ใช้ Pin Change Interrupt ของ Port C
6. จับ rise และ fall timestamps ของ ECHO ทั้งสองช่อง
7. ไม่ใช้ `pulseIn()`
8. ไม่ใช้ `analogRead()`
9. มี timeout
10. มี calibration offset
11. มี physical plausibility filter
12. มี median filter
13. มี cycle-slip detection
14. มี confidence score
15. ส่งออกข้อมูลผ่าน Serial ที่ 115200 baud
16. โค้ดต้อง compile บน Arduino Nano ATmega328P

### 16.2 State machine ที่แนะนำ

สถานะ:

```text
IDLE
ARMED
WAITING_FOR_RISE
WAITING_FOR_FALL
COMPLETE
TIMEOUT
```

แต่ละ channel มีสถานะย่อย:

```text
NOT_STARTED
RISE_SEEN
FALL_SEEN
INVALID
```

ลำดับการวัด:

```text
1. reset timestamps
2. clear interrupt flags
3. reset Timer1
4. wait guard interval
5. send 10 us trigger
6. wait for both ECHO channels
7. timeout after defined limit
8. copy volatile variables atomically
9. calculate pulse widths
10. apply calibration
11. validate
12. filter
13. print result
```

### 16.3 ตัวแปรสำคัญ

```cpp
const float SOUND_SPEED_CM_PER_US = 0.0343f;
const float RECEIVER_SPACING_CM = ...;
const float CALIBRATION_OFFSET_US = ...;
const uint32_t MEASUREMENT_TIMEOUT_US = ...;
const float MIN_DISTANCE_CM = ...;
const float MAX_DISTANCE_CM = ...;
const size_t FILTER_WINDOW = 5 or 7;
```

### 16.4 Timestamp

ใช้หน่วย timer ticks ภายใน ISR

```cpp
volatile uint32_t echo1RiseTicks;
volatile uint32_t echo1FallTicks;
volatile uint32_t echo2RiseTicks;
volatile uint32_t echo2FallTicks;
```

แปลงเป็นไมโครวินาทีภายนอก ISR:

```cpp
float ticksToUs(uint32_t ticks) {
    return ticks * 0.5f;
}
```

ห้ามใช้ floating-point ภายใน ISR

---

## 17. การคำนวณที่ควรแยกออกจากกัน

ต้องแสดงค่าต่อไปนี้แยกกัน:

### 17.1 Pulse width ของแต่ละโมดูล

```cpp
pulse1Us = fall1Us - rise1Us;
pulse2Us = fall2Us - rise2Us;
```

### 17.2 ระยะตาม output ของแต่ละโมดูล

```cpp
distance1Cm = pulse1Us * c / 2;
distance2Cm = pulse2Us * c / 2;
```

### 17.3 Raw pseudo-TDOA

```cpp
rawDeltaUs = pulse2Us - pulse1Us;
```

### 17.4 Calibrated pseudo-TDOA

```cpp
calibratedDeltaUs = rawDeltaUs - calibrationOffsetUs;
```

### 17.5 Path difference

```cpp
pathDifferenceCm = calibratedDeltaUs * c;
```

### 17.6 Angle estimate

เฉพาะเมื่อค่าผ่าน physical gate:

```cpp
ratio = pathDifferenceCm / receiverSpacingCm;
angleDeg = asin(ratio) * 180.0 / PI;
```

ต้อง clamp ratio ให้อยู่ในช่วง [-1, 1] หลังจากตรวจ validity แล้ว

---

## 18. Calibration

### 18.1 Static offset calibration

จัดอุปกรณ์แบบสมมาตร:

```text
R1        T        R2
|---------|---------|

แผ่นสะท้อนเรียบอยู่ตรงกลางด้านหน้า
```

เงื่อนไข:

- R1 และ R2 ห่างจาก T เท่ากัน
- ทรานสดิวเซอร์อยู่ระดับเดียวกัน
- หันหน้าไปทิศเดียวกัน
- แผ่นสะท้อนอยู่ห่าง 30-50 cm
- ไม่ขยับอุปกรณ์
- เก็บอย่างน้อย 100-500 measurements

คำนวณ median ของ raw delta:

```cpp
calibrationOffsetUs = median(rawDeltaUs samples);
```

ไม่ควรใช้ค่าเฉลี่ยธรรมดาอย่างเดียว เพราะ outlier จำนวนมากอาจดึงค่าเฉลี่ยผิด

### 18.2 Jitter characterization

คำนวณ:

- median
- mean
- minimum
- maximum
- standard deviation
- median absolute deviation
- percentage of rejected samples
- histogram หรือกลุ่มค่าที่พบบ่อย

หากพบหลาย cluster ห่างกันประมาณ 25 us ให้รายงาน cycle ambiguity

---

## 19. Median Filter

ใช้ window จำนวนคี่ เช่น 5 หรือ 7 ค่า

ตัวอย่าง:

```text
raw dt:
19, -56, 18, 20, 19
```

median คือ 19 us ซึ่งทนต่อ outlier -56 us ได้ดีกว่าค่าเฉลี่ย

แต่ถ้าค่ากระโดดเป็น cluster และ cluster ผิดมีจำนวนมาก median อาจยังผิดได้ จึงต้องใช้ร่วมกับ:

- plausibility gate
- cycle-slip correction
- confidence score
- temporal continuity

---

## 20. Cycle-slip Detection

เนื่องจากคาบ 40 kHz เท่ากับ 25 us สามารถตรวจว่าค่าปัจจุบันอยู่ห่างจากค่าก่อนหน้าใกล้เคียงจำนวนเต็มเท่าของ 25 us หรือไม่

แนวคิด:

```cpp
differenceFromPrevious = currentDeltaUs - previousFilteredDeltaUs;
cycleCount = round(differenceFromPrevious / 25.0f);
residual = differenceFromPrevious - cycleCount * 25.0f;
```

หาก:

```text
abs(residual) < tolerance
และ
abs(cycleCount) >= 1
```

อาจเป็น cycle slip

ตัวอย่าง tolerance:

```text
2-5 us
```

แนวทาง correction เชิงทดลอง:

```cpp
correctedDeltaUs = currentDeltaUs - cycleCount * 25.0f;
```

แต่ต้องทำเฉพาะเมื่อ:

- corrected value อยู่ใน physical bound
- corrected value ใกล้ median ของ history
- confidence เพียงพอ
- ไม่ใช้ correction เพื่อบังคับให้ทุกค่าดูดี

ต้องพิมพ์ flag ว่าเกิด correction หรือไม่:

```text
cycle_slip=1
cycle_shift=-2
```

หมายถึงลบสอง cycle หรือ 50 us

---

## 21. Confidence Score

สร้างคะแนน 0-100 จากปัจจัย:

- ผ่าน physical bound
- pulse width ของทั้งสองช่องอยู่ในช่วงสมเหตุสมผล
- ค่าใกล้ median history
- ไม่เกิด timeout
- ไม่เกิด cycle slip หรือ correction น้อย
- spread ของ filter window ต่ำ
- ทั้งสองช่องตรวจพบครบ
- ค่าไม่อยู่ใกล้ dead zone
- ค่าไม่กระโดดจากครั้งก่อนมากเกินไป

ตัวอย่างแนวคิด:

```text
เริ่ม 100
-40 หาก timeout
-30 หากเกิน physical bound
-20 หาก cycle slip
-15 หาก deviation จาก median สูง
-10 หากอยู่ใกล้ dead zone
```

ผลลัพธ์ควรมี:

```text
confidence=85
status=VALID
```

หรือ:

```text
confidence=25
status=UNRELIABLE
```

---

## 22. Output Format

ต้องรองรับสองโหมด:

### 22.1 Human-readable

```text
R1=212.5us 3.64cm | R2=230.5us 3.95cm | raw_dt=18.0us | calibrated_dt=0.5us | path=0.02cm | angle=0.2deg | confidence=91 | status=VALID
```

### 22.2 CSV

Header:

```text
timestamp_ms,r1_pulse_us,r2_pulse_us,r1_cm,r2_cm,raw_dt_us,calibrated_dt_us,path_diff_cm,angle_deg,confidence,status,cycle_shift
```

ตัวอย่าง:

```text
1250,212.5,230.5,3.64,3.95,18.0,0.5,0.017,0.20,91,VALID,0
```

ควรมี compile-time flag:

```cpp
#define OUTPUT_CSV 1
```

---

## 23. Commands ผ่าน Serial

เพิ่ม command parser แบบง่าย:

```text
c
```

เริ่ม calibration

```text
p
```

พิมพ์ parameters ปัจจุบัน

```text
r
```

reset filter history

```text
s
```

สลับ human-readable กับ CSV

```text
t25
```

ตั้งอุณหภูมิเป็น 25 °C แล้วคำนวณความเร็วเสียงใหม่

```text
d5.0
```

ตั้ง receiver spacing เป็น 5.0 cm

ค่าที่เปลี่ยนได้อาจเก็บไว้ใน RAM ก่อน ไม่จำเป็นต้องเขียน EEPROM ในเวอร์ชันแรก

---

## 24. Acceptance Criteria สำหรับซอฟต์แวร์

โค้ดถือว่าผ่านเมื่อ:

1. Compile บน Arduino Nano ATmega328P ได้
2. Serial Monitor ใช้ 115200 baud
3. ไม่มีการใช้ `pulseIn()`
4. ไม่มีการใช้ `analogRead()` กับ ECHO
5. ใช้ Timer1 และ Pin Change Interrupt
6. จับ ECHO1 และ ECHO2 ได้ใน measurement เดียวกัน
7. มี timeout ที่ไม่ทำให้โปรแกรมค้าง
8. มี calibration offset
9. มี physical bound จาก receiver spacing
10. มี median filter อย่างน้อย 5 ค่า
11. มี cycle-slip detector ที่ใช้ 25 us เป็นคาบอ้างอิง
12. มี confidence score
13. มี human-readable output
14. มี CSV output
15. มี comments อธิบาย register สำคัญ
16. ISR ต้องสั้นและไม่มี Serial หรือ floating-point
17. อ่านตัวแปร volatile แบบ atomic
18. รองรับ Timer1 overflow
19. แสดง status ของ measurement ชัดเจน
20. ไม่รายงาน angle เมื่อข้อมูลไม่ valid

---

## 25. แผนการทดลอง

### Experiment 1: ตรวจการทำงานพื้นฐาน

- วางแผ่นสะท้อนที่ 30 cm
- ยิงทุก 100 ms
- เก็บ 100 ค่า
- ตรวจ timeout
- ตรวจว่า R1 และ R2 มีค่าใกล้เคียงกัน

### Experiment 2: Symmetric calibration

- วาง R1 และ R2 สมมาตรรอบ T
- เก็บ 500 ค่า
- คำนวณ median offset
- ตรวจ distribution ของ raw dt
- ตรวจว่ามีกี่ cluster

### Experiment 3: Receiver spacing

ทดลอง spacing:

```text
2 cm
5 cm
10 cm
```

ตรวจว่า physical max dt เปลี่ยนตาม:

\[
\Delta t_{\max} = D/c
\]

### Experiment 4: Move reflector left-right

เลื่อนแผ่นสะท้อนทีละตำแหน่ง:

```text
ซ้ายมาก
ซ้าย
กลาง
ขวา
ขวามาก
```

ตรวจว่า sign ของ calibrated dt เปลี่ยนตามทิศอย่างสม่ำเสมอหรือไม่

### Experiment 5: Repeatability

ที่ตำแหน่งเดิม:

- เก็บ 1,000 ค่า
- คำนวณ valid rate
- คำนวณ standard deviation
- คำนวณ cycle-slip rate
- ตรวจ drift ตามเวลา

### Experiment 6: Distance sweep

ทดลองระยะ:

```text
10, 20, 30, 40, 50, 75, 100 cm
```

ตรวจว่าระยะใดมีค่าคงที่ที่สุด และระยะใดเกิด dead zone หรือ multipath มาก

---

## 26. ข้อจำกัดสำคัญที่ต้องเขียนไว้ใน README และ Serial Output

1. ระบบนี้ใช้ US-015 สองบอร์ดเป็น black-box receiver
2. output ECHO เป็นผลหลังผ่านวงจรประมวลผลภายใน ไม่ใช่ raw 40 kHz waveform
3. ความต่างของ pulse width ไม่รับประกันว่าเป็น true arrival-time difference
4. calibration ช่วยลบ static offset แต่ลบ dynamic jitter ไม่ได้ทั้งหมด
5. cycle-slip correction เป็น heuristic
6. angle estimate ใช้ได้เฉพาะเมื่อ physical assumptions ถูกต้อง
7. ผลลัพธ์ควรถือเป็น prototype และ qualitative direction estimate ก่อน
8. ไม่ควรใช้ในงาน safety-critical
9. หากต้องการ true TDOA ควรเข้าถึงสัญญาณก่อน comparator หรือออกแบบ receiver front-end แยกเอง
10. การวัดด้วยสอง receivers ให้มุมได้หนึ่งแกนและอาจมี ambiguity
11. การวัด position 2D/3D ต้องเพิ่มจำนวน receivers และใช้ geometry เพิ่มเติม

---

## 27. แนวทางฮาร์ดแวร์ที่แม่นกว่าระยะยาว

หากต้องการ true TDOA ควรเปลี่ยนจาก US-015 black-box เป็น architecture:

```text
RX1 -> Band-pass amplifier 40 kHz -> Comparator -> Digital timestamp
RX2 -> Band-pass amplifier 40 kHz -> Comparator -> Digital timestamp
```

หรือ:

```text
RX1 -> ADC channel 1
RX2 -> ADC channel 2
```

โดย ADC ต้อง:

- sampling เร็วกว่าความถี่ 40 kHz อย่างมี margin
- ควรอย่างน้อย 200-500 kS/s ต่อ channel
- หากต้องการ phase/TDOA แม่น ควร simultaneous sampling
- ใช้ clock เดียวกัน
- ทำ cross-correlation หรือ matched filtering

แนวทาง DSP:

1. บันทึก waveform ของ RX1 และ RX2
2. band-pass รอบ 40 kHz
3. normalize amplitude
4. cross-correlate
5. หา lag ที่ correlation สูงสุด
6. interpolation รอบ peak เพื่อเพิ่ม sub-sample precision
7. แปลง lag เป็น dt
8. แปลง dt เป็น path difference และ angle

สูตร cross-correlation:

\[
R_{xy}[k] = \sum_n x[n]y[n+k]
\]

ค่า \(k\) ที่ทำให้ \(R_{xy}[k]\) สูงสุดคือ sample delay โดยประมาณ

\[
\Delta t = \frac{k}{f_s}
\]

---

## 28. สิ่งที่ต้องการให้ Codex ส่งมอบ

ให้ Codex สร้างไฟล์:

```text
ultrasonic_tdoa_nano.ino
README.md
```

### เนื้อหาใน `ultrasonic_tdoa_nano.ino`

- Pin definitions
- Timer1 setup
- PCINT setup
- overflow handling
- atomic snapshot
- measurement state machine
- trigger generation
- timeout
- pulse width calculation
- calibration
- physical validation
- median filter
- cycle-slip detector
- confidence score
- human-readable output
- CSV output
- serial commands
- comments ภาษาไทยหรืออังกฤษที่เข้าใจง่าย

### เนื้อหาใน `README.md`

- wiring
- assumptions
- equations
- calibration instructions
- limitations
- experiment procedure
- interpretation of sign
- troubleshooting
- expected Serial output

---

## 29. Prompt สำหรับส่งให้ Codex โดยตรง

```text
พัฒนา Arduino sketch สำหรับ Arduino Nano ATmega328P ตาม specification ในเอกสารนี้

ข้อกำหนดบังคับ:
- A0 = ECHO1
- A1 = ECHO2
- A2 = TRIG ร่วมของ US-015 สองบอร์ด
- Serial 115200
- ใช้ Timer1 prescaler 8
- ใช้ Pin Change Interrupt ของ Port C
- ห้ามใช้ pulseIn()
- ห้ามใช้ analogRead() กับ ECHO
- ISR ต้องสั้น ไม่มี Serial และไม่มี floating-point
- ต้องรองรับ Timer1 overflow
- ต้อง snapshot volatile variables แบบ atomic
- ต้องมี timeout
- ต้องมี calibration offset
- ต้องมี physical plausibility gate จาก receiver spacing
- ต้องมี median filter window 5 หรือ 7
- ต้องมี cycle-slip detector โดยอ้างอิงคาบ 25 us ของ 40 kHz
- ต้องมี confidence score
- ต้องมี output แบบ human-readable และ CSV
- ต้องไม่คำนวณหรือแสดง angle หาก measurement invalid
- ต้อง compile ได้บน Arduino Nano classic/ATmega328P

กรุณาสร้าง:
1. ultrasonic_tdoa_nano.ino
2. README.md

ให้อธิบาย architecture, register setup, timing, equations, limitations และวิธี calibrate อย่างละเอียด
```

---

## 30. สรุปหลักการสำคัญ

- A0 และ A1 ใช้เป็น digital input ได้
- ECHO ของ US-015 ไม่ใช่ raw analog waveform
- `analogRead()` ไม่เหมาะกับ TDOA
- Timer1 ที่ prescaler 8 ให้ resolution 0.5 us
- 40 kHz มีคาบ 25 us
- ค่ากระโดดประมาณ 25, 50, 75 us อาจเกิด cycle slip
- ความต่าง path สำหรับ TDOA ใช้ \(c\Delta t\) ไม่หาร 2
- ระยะ pulse-echo ใช้ \(ct/2\)
- \(|\Delta t|\) ต้องไม่เกิน \(D/c\)
- US-015 สองบอร์ดมี internal delay ต่างกัน จึงต้อง calibrate
- แม้ calibrate แล้ว ผลยังเป็น pseudo-TDOA ไม่ใช่ true TDOA ที่รับประกัน
- หากต้องการความแม่นยำจริง ควรเข้าถึง raw receiver signal แล้วใช้ comparator ร่วมกันหรือ simultaneous ADC + cross-correlation
