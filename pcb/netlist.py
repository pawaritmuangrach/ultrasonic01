"""วงจรของทั้งสามบอร์ด เขียนเป็น netlist — **แหล่งความจริงเดียว**

ทุกอย่างที่ตามมา (ผังวงจร, ลายทองแดง, BOM, ไฟล์ Gerber) สร้างจากไฟล์นี้
และตรวจย้อนกลับมาที่ไฟล์นี้ ถ้าลายทองแดงต่อไม่ตรง netlist ตัวตรวจจะฟ้อง

ค่าทุกตัวยกมาจากวงจรที่ **ต่อบนแผ่นไข่ปลาแล้วใช้งานได้จริง**
ไม่ได้คิดใหม่ ที่มา: hardware/rx_frontend.md · hardware/tx_driver.md
                    hardware/perf_rx_layout.py · hardware/perf_tx_layout.py

รูปแบบ: ชิ้นส่วนคือ (อ้างอิง, ค่า, footprint) · เน็ตคือ ชื่อ -> รายการ "ref.pin"
"""

# ---------------------------------------------------------------- footprint
# ชื่อ footprint บอกรูปร่างรูเจาะจริง ตัวเลขคือระยะขาเป็นมิลลิเมตร
FP = {
    "DIP14": dict(kind="dip", n=14, pitch=2.54, span=7.62, drill=0.9, pad=1.7),
    # 7.62 ไม่ใช่ 10.16 — บอร์ดรวมมีตัวต้านทาน 51 ตัว ระยะสั้นลง 2.54 มม.
    # ต่อตัวประหยัดพื้นที่ไปมาก และ 1/4W ยังงอขาลงได้สบาย
    "R_AXIAL": dict(kind="axial", pitch=7.62, drill=0.9, pad=1.8),
    "C_DISC": dict(kind="axial", pitch=5.08, drill=0.9, pad=1.8),
    # แป้น 1.7 ไม่ใช่ 2.0 — ที่ระยะขา 2.54 แป้น 2.0 เหลือช่องว่างแค่ 0.54 มม.
    # ซึ่งบางเกินกติกา 0.4 มม. เมื่อปัดลงตาราง · 1.7 ได้ 0.84 มม. สบาย ๆ
    # และยังเหลือเนื้อทองแดงรอบรู 0.4 มม. ตามเกณฑ์
    "C_ELEC": dict(kind="axial", pitch=2.54, drill=0.9, pad=1.7),
    "HDR2": dict(kind="header", n=2, pitch=2.54, drill=1.0, pad=1.8),
    # เฮดเดอร์ตัวเมีย 20 ขา สำหรับเสียบ Blue Pill · สองแถวห่างกัน 17.78 มม.
    "HDR20": dict(kind="header", n=20, pitch=2.54, drill=1.0, pad=1.8),
    # เทอร์มินอลบล็อกขันสกรู KF301 ระยะขา 5.08 มม. รูใหญ่กว่าเพราะขาหนา
    "TERM2": dict(kind="header", n=2, pitch=5.08, drill=1.2, pad=2.4),
    "C_BULK": dict(kind="axial", pitch=5.08, drill=1.0, pad=1.9),
    "HDR3": dict(kind="header", n=3, pitch=2.54, drill=1.0, pad=1.8),
    "HDR4": dict(kind="header", n=4, pitch=2.54, drill=1.0, pad=1.8),
}


# ============================================================ บอร์ด TX 3 ช่อง
def tx3():
    """ภาคส่ง 3 หัว จาก 74HCT04 ตัวเดียว (6 เกต = 3 คู่)

    เกตสองชั้นต่อหนึ่งหัวส่ง เพื่อให้ขาออก **ตามขาเข้า** ตอนพัก GPIO เป็น LOW
    ขาออกก็ LOW = แรงดันคร่อมหัวส่ง 0V ถ้าใช้เกตชั้นเดียว ตอนพักจะมี 5V
    คร่อมหัวส่งตลอดเวลา ทำให้ร้อนและอายุสั้น

    R 100 โอห์ม อนุกรมทุกหัว — จงใจทำให้ขอบสัญญาณช้าลง ลดการแผ่คลื่นรบกวน
    และจำกัดกระแสพีค (5V / 145 โอห์ม = 34 mA ต่อเกต)

    ห้ามต่อขาออกของชิป (5V) กลับเข้า GPIO ของ ESP32 — พังถาวร ทิศทางมีทางเดียว
    """
    parts = [
        ("U1", "74HCT04", "DIP14"),
        ("R1", "100R", "R_AXIAL"), ("R2", "100R", "R_AXIAL"), ("R3", "100R", "R_AXIAL"),
        ("C1", "100nF", "C_DISC"), ("C2", "10uF", "C_ELEC"),
        ("J1", "ESP32", "HDR4"),          # GND, TX1, TX2, TX3 (สัญญาณ 3.3V เข้า)
        ("JP", "PWR", "HDR2"),            # +5V, GND
        ("T1", "TCT40-16T", "HDR2"),
        ("T2", "TCT40-16T", "HDR2"),
        ("T3", "TCT40-16T", "HDR2"),
    ]
    nets = {
        "+5V": ["U1.14", "C1.1", "C2.1", "JP.1"],
        "GND": ["U1.7", "C1.2", "C2.2", "JP.2", "J1.1",
                "T1.2", "T2.2", "T3.2"],
        # TX1 = เกต 1,2 (ขา 1->2, 3->4) ขาออก = ขา 4
        "TX1_IN": ["J1.2", "U1.1"],
        "TX1_MID": ["U1.2", "U1.3"],
        "TX1_OUT": ["U1.4", "R1.1"],
        "TX1_DRV": ["R1.2", "T1.1"],
        # TX2 = เกต 6,5 (ขา 13->12, 11->10) ขาออก = ขา 10
        "TX2_IN": ["J1.3", "U1.13"],
        "TX2_MID": ["U1.12", "U1.11"],
        "TX2_OUT": ["U1.10", "R2.1"],
        "TX2_DRV": ["R2.2", "T2.1"],
        # TX3 = เกต 3,4 (ขา 5->6, 9->8) ขาออก = ขา 8
        "TX3_IN": ["J1.4", "U1.5"],
        "TX3_MID": ["U1.6", "U1.9"],
        "TX3_OUT": ["U1.8", "R3.1"],
        "TX3_DRV": ["R3.2", "T3.1"],
    }
    return dict(name="tx3", title="TX 3 channels - 74HCT04",
                parts=parts, nets=nets)


# ============================================================ บอร์ด RX 2 ช่อง
def _rx_channel(ch, opa, opb, hdr, jout):
    """หนึ่งช่องรับ = ออปแอมป์สองสเตจ · คืน (ชิ้นส่วน, เน็ต)

    opa/opb = (ขา -IN, ขา +IN, ขา OUT) ของ section ที่ใช้ทำสเตจ 1 และ 2

    เกน AC = 1 + Rf/Rg = 1 + 33k/1k = 34 ต่อสเตจ · สองสเตจได้ 1156 เท่า
    เกน DC = 1 เพราะ Cg กั้นกระแสตรงที่ขา Rg ไม่ให้ผ่าน จุดทำงานจึงนิ่งที่ VREF
    ถ้าไม่มี Cg เกน DC จะเท่ากับเกน AC แล้วความต่างเล็กน้อยของออปแอมป์
    จะถูกขยาย 34 เท่าจนขาออกไปติดราง
    """
    n, p, o = opa
    n2, p2, o2 = opb
    parts = [
        (f"C{ch}1", "10nF", "C_DISC"),        # กั้นไฟตรงจากหัวรับ
        (f"R{ch}B", "1M", "R_AXIAL"),         # ดึงขาเข้าไปเกาะ VREF
        (f"R{ch}F1", "33k", "R_AXIAL"), (f"R{ch}G1", "1k", "R_AXIAL"),
        (f"C{ch}G1", "100nF", "C_DISC"),
        (f"R{ch}F2", "33k", "R_AXIAL"), (f"R{ch}G2", "1k", "R_AXIAL"),
        (f"C{ch}G2", "100nF", "C_DISC"),
        (f"R{ch}O", "1k", "R_AXIAL"),         # จำกัดกระแสเข้า ADC
        (f"C{ch}O", "1nF", "C_DISC"),         # กรองความถี่สูงก่อนเข้า ADC
        (hdr, "TCT40-16R", "HDR2"),
    ]
    nets = {
        f"CH{ch}_IN": [f"{hdr}.1", f"C{ch}1.1"],
        f"CH{ch}_G": [f"C{ch}1.2", f"R{ch}B.1", f"U1.{p}"],
        f"CH{ch}_S1O": [f"U1.{o}", f"R{ch}F1.1", f"U1.{p2}"],
        f"CH{ch}_S1N": [f"U1.{n}", f"R{ch}F1.2", f"R{ch}G1.1"],
        f"CH{ch}_G1": [f"R{ch}G1.2", f"C{ch}G1.1"],
        f"CH{ch}_S2O": [f"U1.{o2}", f"R{ch}F2.1", f"R{ch}O.1"],
        f"CH{ch}_S2N": [f"U1.{n2}", f"R{ch}F2.2", f"R{ch}G2.1"],
        f"CH{ch}_G2": [f"R{ch}G2.2", f"C{ch}G2.1"],
        f"CH{ch}_OUT": [f"R{ch}O.2", f"C{ch}O.1", f"{jout}"],
    }
    gnd = [f"{hdr}.2", f"C{ch}G1.2", f"C{ch}G2.2", f"C{ch}O.2"]
    vref = [f"R{ch}B.2"]
    return parts, nets, gnd, vref


def rx2():
    """ภาครับ 2 ช่อง ใช้ MCP6004/MCP6024 ตัวเดียวครบ 4 section

    section A+B = ช่อง 1 · C+D = ช่อง 2
    **บอร์ดนี้ไม่สร้าง VREF เอง** รับมาจากบอร์ด vref ผ่านสาย 3 เส้น
    เพราะ 4 section ถูกใช้เป็นสเตจขยายหมดแล้ว ไม่เหลือให้ทำบัฟเฟอร์

    ต้องบัฟเฟอร์ VREF ไม่ใช่ต่อตัวแบ่งเปล่า ๆ — ตัวแบ่ง 10k/10k มีอิมพีแดนซ์ 5k
    ถ้าเลี้ยงหลายช่องตรง ๆ กระแสของช่องหนึ่งจะกระเพื่อม VREF ของช่องอื่น = ครอสทอล์ก
    """
    # MCP6004 DIP-14: A(-1? ) ใช้ตามดาต้าชีต
    #   A: -IN=2  +IN=3  OUT=1
    #   B: -IN=6  +IN=5  OUT=7
    #   C: -IN=9  +IN=10 OUT=8
    #   D: -IN=13 +IN=12 OUT=14
    A, B, C, D = (2, 3, 1), (6, 5, 7), (9, 10, 8), (13, 12, 14)
    parts = [("U1", "MCP6004", "DIP14"),
             ("C1", "100nF", "C_DISC"), ("C2", "10uF", "C_ELEC"),
             ("J1", "PWR+VREF", "HDR3"),      # 3V3, GND, VREF
             ("J2", "ADC OUT", "HDR3")]       # OUT1, OUT2, GND
    nets = {"3V3": ["U1.4", "C1.1", "C2.1", "J1.1"]}
    gnd = ["U1.11", "C1.2", "C2.2", "J1.2", "J2.3"]
    vref = ["J1.3"]
    for ch, (oa, ob, hdr, jo) in enumerate(
            [(A, B, "K1", "J2.1"), (C, D, "K2", "J2.2")], start=1):
        p, n, g, v = _rx_channel(ch, oa, ob, hdr, jo)
        parts += p
        nets.update(n)
        gnd += g
        vref += v
    nets["GND"] = gnd
    nets["VREF"] = vref
    return dict(name="rx2", title="RX 2 channels - MCP6004",
                parts=parts, nets=nets)


# ============================================================== บอร์ด VREF
def vref():
    """สร้าง 1.65V ครั้งเดียว แล้วแจกให้ทุกบอร์ดรับ

    ทำแยกเพราะ **VREF ต้องเป็นจุดเดียวกันทั้งระบบ** ถ้าแต่ละบอร์ดสร้างเอง
    ค่าจะต่างกันเล็กน้อยตามความคลาดของตัวต้านทาน แล้วช่องต่าง ๆ จะมีจุดศูนย์
    ไม่ตรงกัน เวลาเทียบความแรงระหว่างช่อง (ซึ่งคือที่มาของมุม) จะเพี้ยน

    ใช้ MCP6004 หนึ่ง section เป็นตัวตาม อีกสาม section ผูกเป็นตัวตามที่ VREF
    เหมือนกัน ไม่ปล่อยขาลอย — ขาเข้าที่ลอยจะแกว่งและกินกระแส
    """
    parts = [("U1", "MCP6004", "DIP14"),
             ("R1", "10k", "R_AXIAL"), ("R2", "10k", "R_AXIAL"),
             ("C1", "100nF", "C_DISC"), ("C2", "10uF", "C_ELEC"),
             ("C3", "100nF", "C_DISC"), ("C4", "10uF", "C_ELEC"),
             ("J1", "3V3 IN", "HDR2"),
             ("J2", "OUT A", "HDR3"), ("J3", "OUT B", "HDR3"),
             ("J4", "OUT C", "HDR3"), ("J5", "OUT D", "HDR3")]
    nets = {
        "3V3": ["U1.4", "R1.1", "C1.1", "C2.1", "J1.1",
                "J2.1", "J3.1", "J4.1", "J5.1"],
        "GND": ["U1.11", "R2.2", "C1.2", "C2.2", "C3.2", "C4.2", "J1.2",
                "J2.2", "J3.2", "J4.2", "J5.2"],
        "MID": ["R1.2", "R2.1", "U1.3"],          # จุดกึ่งกลางตัวแบ่ง เข้า +INA
        # ตัวตาม: OUTA ป้อนกลับเข้า -INA · section ที่เหลือผูกเป็นตัวตามที่ VREF
        "VREF": ["U1.1", "U1.2", "C3.1", "C4.1",
                 "U1.5", "U1.6", "U1.7",
                 "U1.10", "U1.9", "U1.8",
                 "U1.12", "U1.13", "U1.14",
                 "J2.3", "J3.3", "J4.3", "J5.3"],
    }
    return dict(name="vref", title="VREF 1.65V buffer + power hub",
                parts=parts, nets=nets)


BOARDS = {"tx3": tx3, "rx2": rx2, "vref": vref}


def check(board):
    """ตรวจ netlist เองก่อนเอาไปใช้ — ขาซ้ำสองเน็ต หรือขาที่ไม่มีอยู่จริง"""
    parts = {r: (v, f) for r, v, f in board["parts"]}
    seen, errs = {}, []
    for net, pins in board["nets"].items():
        for pin in pins:
            ref, _, num = pin.partition(".")
            if ref not in parts:
                errs.append(f"{net}: ไม่มีชิ้นส่วน {ref}")
                continue
            fp = FP[parts[ref][1]]
            npin = fp.get("n", 2)
            if not num.isdigit() or not 1 <= int(num) <= npin:
                errs.append(f"{net}: {pin} ขาไม่มีอยู่จริง (มี 1-{npin})")
            if pin in seen:
                errs.append(f"{pin} อยู่สองเน็ต: {seen[pin]} และ {net}")
            seen[pin] = net
    # ขาที่ไม่ได้ต่ออะไรเลย
    for ref, (val, fpn) in parts.items():
        for i in range(1, FP[fpn].get("n", 2) + 1):
            if f"{ref}.{i}" not in seen:
                errs.append(f"{ref}.{i} ({val}) ไม่ได้ต่อเน็ตไหนเลย")
    return errs


if __name__ == "__main__":
    for name, fn in BOARDS.items():
        b = fn()
        e = check(b)
        n_pin = sum(len(v) for v in b["nets"].values())
        print(f"{name:6s} ชิ้นส่วน {len(b['parts']):2d} · เน็ต {len(b['nets']):2d} · "
              f"ขา {n_pin:3d} · {'ผ่าน' if not e else 'ผิด ' + str(len(e))}")
        for x in e:
            print("   -", x)


# ================================================ บอร์ดรวม: STM32 + 3 TX + 8 RX
# ขาของ Blue Pill อ่านจากตัวหนังสือบนบอร์ดจริง (รูปที่ผู้ใช้ส่งมา)
# แถวบนกับแถวล่างห่างกัน 17.78 มม. · 20 ขาต่อแถว · ระยะขา 2.54 มม.
BP_TOP = ["G", "G", "3.3", "R", "B11", "B10", "B1", "B0", "A7", "A6",
          "A5", "A4", "A3", "A2", "A1", "A0", "C15", "C14", "C13", "VB"]
BP_BOT = ["B12", "B13", "B14", "B15", "A8", "A9", "A10", "A11", "A12", "A15",
          "B3", "B4", "B5", "B6", "B7", "B8", "B9", "5V", "G", "3.3"]

# แมปขา Blue Pill -> เน็ตบนบอร์ด · ขาที่ไม่อยู่ในนี้จะกลายเป็นเน็ตเดี่ยว (ไม่ต่อไปไหน)
#   A0..A7 = ADC12_IN0..IN7 ทั้งแปดอยู่ติดกันบนแถวบน ลายจึงไม่ต้องข้ามบอร์ด
#   B6/B7/B8 = TIM4_CH1/2/3 สร้างคลื่น 40 kHz ด้วยฮาร์ดแวร์ จังหวะเป๊ะกว่าสั่งด้วยโค้ด
#   B3/B4/B5 เลี่ยงไว้ เพราะค่าเริ่มต้นเป็นขา JTAG ต้องปลดก่อนใช้เป็น GPIO
BP_NET = {"G": "GND", "3.3": "MCU3V3", "5V": "P5V",
          "A0": "RX1_ADC", "A1": "RX2_ADC", "A2": "RX3_ADC", "A3": "RX4_ADC",
          "A4": "RX5_ADC", "A5": "RX6_ADC", "A6": "RX7_ADC", "A7": "RX8_ADC",
          "B6": "TX1_IN", "B7": "TX2_IN", "B8": "TX3_IN"}

# MCP6024 DIP-14 (ขาเหมือน MCP6004 ทุกประการ ต่างที่ GBW 10 MHz ไม่ใช่ 1 MHz)
OPA = {"A": (2, 3, 1), "B": (6, 5, 7), "C": (9, 10, 8), "D": (13, 12, 14)}


def _mcu_rows(parts, nets, nc):
    """เฮดเดอร์ตัวเมียสองแถวสำหรับเสียบ Blue Pill"""
    for ref, row in (("JA", BP_TOP), ("JB", BP_BOT)):
        parts.append((ref, "Blue Pill", "HDR20"))
        for i, lab in enumerate(row, start=1):
            pin = f"{ref}.{i}"
            net = BP_NET.get(lab)
            if net:
                nets.setdefault(net, []).append(pin)
            else:
                # ขาที่ไม่ได้ใช้ ต้องมีรูและแป้นเหมือนกัน แต่ห้ามต่อถึงกัน
                # จึงให้เน็ตเดี่ยวของตัวเอง ไม่ใช่รวมเป็นเน็ต NC ก้อนเดียว
                nc[f"NC_{ref}_{lab}"] = [pin]


def _tx_stage(parts, nets, i, gin, gmid_a, gmid_b, gout):
    """หนึ่งหัวส่ง = สองเกตต่ออนุกรม + ตัวต้านทานอนุกรม + เทอร์มินอล"""
    parts += [(f"RT{i}", "100R", "R_AXIAL"), (f"T{i}", f"TX{i}", "TERM2")]
    nets[f"TX{i}_IN"] = nets.get(f"TX{i}_IN", []) + [f"U1.{gin}"]
    nets[f"TX{i}_MID"] = [f"U1.{gmid_a}", f"U1.{gmid_b}"]
    nets[f"TX{i}_OUT"] = [f"U1.{gout}", f"RT{i}.1"]
    nets[f"TX{i}_DRV"] = [f"RT{i}.2", f"T{i}.1"]
    return [f"T{i}.2"]


def _rx_stage(parts, nets, i, chip, sa, sb):
    """หนึ่งช่องรับ = ออปแอมป์สองสเตจ · คืน (ขาที่ลง GND, ขาที่ไป VREF)"""
    n1, p1, o1 = OPA[sa]
    n2, p2, o2 = OPA[sb]
    U = f"U{chip}"
    parts += [(f"K{i}", f"RX{i}", "TERM2"),
              (f"C{i}I", "10nF", "C_DISC"), (f"R{i}B", "1M", "R_AXIAL"),
              (f"R{i}F1", "33k", "R_AXIAL"), (f"R{i}G1", "1k", "R_AXIAL"),
              (f"C{i}G1", "100nF", "C_DISC"),
              (f"R{i}F2", "33k", "R_AXIAL"), (f"R{i}G2", "1k", "R_AXIAL"),
              (f"C{i}G2", "100nF", "C_DISC"),
              (f"R{i}O", "1k", "R_AXIAL"), (f"C{i}O", "1nF", "C_DISC")]
    nets.update({
        f"RX{i}_IN": [f"K{i}.1", f"C{i}I.1"],
        f"RX{i}_G": [f"C{i}I.2", f"R{i}B.1", f"{U}.{p1}"],
        f"RX{i}_S1O": [f"{U}.{o1}", f"R{i}F1.1", f"{U}.{p2}"],
        f"RX{i}_S1N": [f"{U}.{n1}", f"R{i}F1.2", f"R{i}G1.1"],
        f"RX{i}_G1": [f"R{i}G1.2", f"C{i}G1.1"],
        f"RX{i}_S2O": [f"{U}.{o2}", f"R{i}F2.1", f"R{i}O.1"],
        f"RX{i}_S2N": [f"{U}.{n2}", f"R{i}F2.2", f"R{i}G2.1"],
        f"RX{i}_G2": [f"R{i}G2.2", f"C{i}G2.1"],
    })
    nets.setdefault(f"RX{i}_ADC", []).extend([f"R{i}O.2", f"C{i}O.1"])
    gnd = [f"K{i}.2", f"C{i}G1.2", f"C{i}G2.2", f"C{i}O.2"]
    return gnd, [f"R{i}B.2"]


def main8():
    """บอร์ดรวมรุ่นแรก — STM32F103C8T6 + ส่ง 3 หัว + รับ 8 ช่อง บนแผ่นเดียว

    ทำไมย้ายจาก ESP32 มา STM32F103C8T6:
      ESP32 มีขา ADC1 ที่ใช้กับโหมด DMA ต่อเนื่องได้แค่ 6 ขา ต่อ 8 ช่องไม่ได้เลย
      STM32F103 มี ADC สองตัว ขาแอนะล็อก 10 ขา (PA0-PA7, PB0, PB1) และรวมกัน
      ได้เร็วราว 1.4 MHz แปลว่า 8 ช่องยังได้ ~175 kHz ต่อช่อง เร็วกว่าที่ ESP32
      ทำได้กับ 4 ช่อง (66 kHz) ถึง 2.6 เท่า

    ไฟแอนะล็อกกรองแยกจากไฟดิจิทัล:
      3.3V มาจากเรกูเลเตอร์บน Blue Pill ซึ่งเลี้ยง MCU ที่วิ่ง 72 MHz อยู่ด้วย
      วงจรรับขยาย 1156 เท่า สัญญาณรบกวนบนรางไฟจึงถูกขยายไปด้วย
      จึงคั่นด้วย R 10 โอห์ม + C 100uF ก่อนเข้าภาครับ
      ที่ 40 kHz คาปามีอิมพีแดนซ์ 0.04 โอห์ม กรองได้ดี แลกกับแรงดันตก 0.2V
      ที่กระแส 20 mA ซึ่งเหลือ 3.1V ยังเกินขั้นต่ำของ MCP6024 (2.5V)
    """
    parts, nets, nc = [], {}, {}
    _mcu_rows(parts, nets, nc)

    # ---- ภาคส่ง: 74HCT04 หนึ่งตัว 6 เกต = 3 หัว
    parts += [("U1", "74HCT04", "DIP14"),
              ("CD1", "100nF", "C_DISC"), ("CB1", "10uF", "C_ELEC"),
              ("JP", "5V IN", "TERM2")]
    gnd = ["U1.7", "CD1.2", "CB1.2", "JP.2"]
    nets["P5V"] = nets.get("P5V", []) + ["U1.14", "CD1.1", "CB1.1", "JP.1"]
    gnd += _tx_stage(parts, nets, 1, 1, 2, 3, 4)      # เกต 1,2
    gnd += _tx_stage(parts, nets, 2, 13, 12, 11, 10)  # เกต 6,5
    gnd += _tx_stage(parts, nets, 3, 5, 6, 9, 8)      # เกต 3,4

    # ---- ไฟแอนะล็อกที่กรองแล้ว
    parts += [("RF", "10R", "R_AXIAL"), ("CF", "100uF", "C_BULK")]
    nets["MCU3V3"] = nets.get("MCU3V3", []) + ["RF.1"]
    nets["A3V3"] = ["RF.2", "CF.1"]
    gnd.append("CF.2")

    # ---- ภาครับ 8 ช่อง: ชิปละ 2 ช่อง (A+B และ C+D)
    vref = []
    for i in range(1, 9):
        chip = 2 + (i - 1) // 2                      # U2..U5
        sa, sb = ("A", "B") if i % 2 else ("C", "D")
        g, v = _rx_stage(parts, nets, i, chip, sa, sb)
        gnd += g
        vref += v
    for u in range(2, 6):
        parts += [(f"U{u}", "MCP6024", "DIP14"),
                  (f"CD{u}", "100nF", "C_DISC")]
        nets["A3V3"] += [f"U{u}.4", f"CD{u}.1"]
        gnd += [f"U{u}.11", f"CD{u}.2"]

    # ---- VREF: ตัวแบ่งครึ่ง แล้วบัฟเฟอร์ · section ที่เหลือผูกเป็นตัวตาม ไม่ปล่อยลอย
    parts += [("U6", "MCP6024", "DIP14"), ("CD6", "100nF", "C_DISC"),
              ("RV1", "10k", "R_AXIAL"), ("RV2", "10k", "R_AXIAL"),
              ("CV1", "100nF", "C_DISC"), ("CB2", "10uF", "C_ELEC")]
    nets["A3V3"] += ["U6.4", "CD6.1", "RV1.1"]
    gnd += ["U6.11", "CD6.2", "RV2.2", "CV1.2", "CB2.2"]
    nets["MID"] = ["RV1.2", "RV2.1", "U6.3"]
    nets["VREF"] = vref + ["U6.1", "U6.2", "CV1.1", "CB2.1",
                           "U6.5", "U6.6", "U6.7",
                           "U6.10", "U6.9", "U6.8",
                           "U6.12", "U6.13", "U6.14"]
    # ต้องรวมเข้ากับของเดิม ไม่ใช่กำหนดทับ — ขา G ของเฮดเดอร์ถูกใส่ไว้ตั้งแต่
    # _mcu_rows แล้ว ถ้ากำหนดทับจะหายไปสามขา (ตัวตรวจ netlist จับได้)
    nets["GND"] = nets.get("GND", []) + gnd
    nets.update(nc)
    return dict(name="main8", title="STM32F103 + 3 TX + 8 RX", parts=parts, nets=nets)


BOARDS["main8"] = main8
