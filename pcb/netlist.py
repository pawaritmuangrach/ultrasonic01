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
    "R_AXIAL": dict(kind="axial", pitch=10.16, drill=0.9, pad=1.8),
    "C_DISC": dict(kind="axial", pitch=5.08, drill=0.9, pad=1.8),
    # แป้น 1.7 ไม่ใช่ 2.0 — ที่ระยะขา 2.54 แป้น 2.0 เหลือช่องว่างแค่ 0.54 มม.
    # ซึ่งบางเกินกติกา 0.4 มม. เมื่อปัดลงตาราง · 1.7 ได้ 0.84 มม. สบาย ๆ
    # และยังเหลือเนื้อทองแดงรอบรู 0.4 มม. ตามเกณฑ์
    "C_ELEC": dict(kind="axial", pitch=2.54, drill=0.9, pad=1.7),
    "HDR2": dict(kind="header", n=2, pitch=2.54, drill=1.0, pad=1.8),
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
