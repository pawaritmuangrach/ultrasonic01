#!/usr/bin/env python3
"""
scope_view.py - serial scope viewer for the ESP32 ultrasonic front-end.

Reads capture frames from esp32_selftest.ino (and later esp32_scope.ino) and
plots three panels: raw waveform, envelope, and FFT.

Runs standalone with --simulate, so the whole plotting/analysis path can be
verified before any hardware exists. The simulated trace is what a *correct*
capture should look like: transmitter ringing at t=0, then an echo whose
position follows t = 2d/c. Learn to read that plot now; it is the reference
you will compare every real capture against.

Frame format produced by the firmware:
    ASCII line : "#FRAME rate=<hz> ch=<n> n=<samples_per_ch> us=<elapsed> order=<c0,c1,..>\n"
    4 bytes    : magic "USC1"
    payload    : n*ch uint16 little-endian, channel-interleaved
    2 bytes    : uint16 little-endian sum of payload bytes (mod 65536)

Examples:
    python tools/scope_view.py --simulate
    python tools/scope_view.py --simulate --distance-cm 50
    python tools/scope_view.py --port COM3
    python tools/scope_view.py --port COM3 --repeat --save captures/
"""

import argparse
import pathlib
import struct
import sys
import time

import numpy as np

MAGIC = b"USC1"
C_AIR = 343.0  # m/s at 20 C; recalibrate per Stage 2


# --------------------------------------------------------------------------
# signal helpers (numpy only - scipy deliberately not required)
# --------------------------------------------------------------------------


def _padlen(n):
    """Next power of two at least 2n, so FFT filtering is linear not circular.

    Without this the transmitter burst at t=0 wraps around and reappears at
    the end of the record, where it masquerades as a distant echo.
    """
    return 1 << int(np.ceil(np.log2(max(2 * n, 2))))


def envelope(x):
    """Magnitude of the analytic signal (Hilbert envelope), via FFT."""
    n = len(x)
    if n == 0:
        return x
    npad = _padlen(n)
    xp = np.zeros(npad)
    xp[:n] = x
    spec = np.fft.fft(xp)
    h = np.zeros(npad)
    h[0] = h[npad // 2] = 1.0
    h[1 : npad // 2] = 2.0
    return np.abs(np.fft.ifft(spec * h))[:n]


def bandpass(x, fs, lo_hz, hi_hz, taper=0.25):
    """Zero-phase bandpass in the frequency domain.

    Zero-phase matters: any phase shift would bias the time-of-flight we are
    trying to measure. The transition bands are raised-cosine rather than a
    brick wall, because a brick wall has a sinc impulse response that rings
    across the whole record and smears the transmit burst into the echo.
    """
    n = len(x)
    npad = _padlen(n)
    xp = np.zeros(npad)
    xp[:n] = x - np.mean(x)

    spec = np.fft.rfft(xp)
    f = np.fft.rfftfreq(npad, d=1.0 / fs)

    w_lo = max(lo_hz * taper, 1.0)
    w_hi = max(hi_hz * taper, 1.0)
    gain = np.zeros_like(f)
    gain[(f >= lo_hz) & (f <= hi_hz)] = 1.0

    rise = (f > lo_hz - w_lo) & (f < lo_hz)
    gain[rise] = 0.5 * (1 - np.cos(np.pi * (f[rise] - (lo_hz - w_lo)) / w_lo))
    fall = (f > hi_hz) & (f < hi_hz + w_hi)
    gain[fall] = 0.5 * (1 + np.cos(np.pi * (f[fall] - hi_hz) / w_hi))

    return np.fft.irfft(spec * gain, n=npad)[:n]


def peak_frequency(freqs, spec):
    """Sub-bin peak frequency by parabolic interpolation on the log spectrum.

    An FFT bin is fs/N wide, so a fixed-length window gets coarser as the
    sample rate rises - at 2 MS/s with 1024 samples the bins are ~1.9 kHz
    apart, and one steady tone then reads differently at every sample rate.
    Fitting a parabola through the peak bin and its two neighbours recovers
    the true peak to a fraction of a bin.

    This is the same interpolation Stage 2 applies to the cross-correlation
    peak to get sub-sample Δt; worth getting familiar with here, where the
    right answer is already known.

    Returns (peak_hz, bin_width_hz).
    """
    bin_hz = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    k = int(np.argmax(spec))
    if k <= 0 or k >= len(spec) - 1:
        return float(freqs[k]), bin_hz

    y0, y1, y2 = (np.log(max(spec[k - 1], 1e-30)),
                  np.log(max(spec[k], 1e-30)),
                  np.log(max(spec[k + 1], 1e-30)))
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    return (k + delta) * bin_hz, bin_hz


def counts_to_volts(counts, vref=3.3, bits=12):
    return counts.astype(np.float64) * (vref / ((1 << bits) - 1))


def find_echo(env, fs, blank_s, pulse_s=200e-6, simple=False):
    """Index of the most likely echo, past the transmit blanking window.

    Two approaches that look reasonable and are not:

      * Largest peak after blanking. Fails at range - the transmitter's
        ringing tail stays bigger than a distant echo well past blanking.
      * Weighting by t^2 to undo 1/r^2 spreading. Overcorrects, because it
        amplifies the flat noise floor at the end of the record until noise
        wins outright.

    What actually separates the two is shape, not amplitude: ringing decays
    monotonically from t=0, whereas an echo is a *local* bump standing above
    its own neighbourhood. So compare every sample against the background
    either side of it, skipping guard cells so the echo cannot contaminate
    its own reference. This is cell-averaging CFAR, and on a decaying tail it
    scores below 1 by convexity, which rejects the ringing for free.
    """
    n = len(env)
    i0 = min(int(blank_s * fs), max(n - 1, 0))
    if simple:
        return i0 + int(np.argmax(env[i0:])) if n > i0 else 0

    guard = max(1, int(pulse_s * fs))
    ref = max(2, 4 * guard)

    cumsum = np.concatenate([[0.0], np.cumsum(env)])

    def window_mean(lo, hi):
        lo = np.clip(lo, 0, n)
        hi = np.clip(hi, 0, n)
        return (cumsum[hi] - cumsum[lo]) / np.maximum(hi - lo, 1)

    idx = np.arange(n)
    background = 0.5 * (window_mean(idx - guard - ref, idx - guard)
                        + window_mean(idx + guard + 1, idx + guard + 1 + ref))

    stat = env / np.maximum(background, 1e-15)
    stat[:i0] = 0.0
    stat[max(n - guard, 0):] = 0.0  # right reference window runs off the end
    return int(np.argmax(stat))


def spectrum(x, fs, center, width):
    """FFT of a window around `center`.

    Windowing the whole capture is wrong for a pulsed signal: a Hanning
    window is zero at the edges, so a burst sitting near t=0 gets erased and
    the FFT ends up showing nothing but noise. Analyse a slice around the
    event instead, and drop the DC bin so a residual offset cannot masquerade
    as the peak.
    """
    n = len(x)
    width = min(width, n)
    start = int(np.clip(center - width // 2, 0, max(0, n - width)))
    stop = start + width

    seg = x[start:stop]
    seg = (seg - np.mean(seg)) * np.hanning(len(seg))

    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
    if len(spec) > 1:
        spec[0] = 0.0  # kill DC so it cannot win the peak search
    return freqs, spec, start, stop


# --------------------------------------------------------------------------
# frame acquisition
# --------------------------------------------------------------------------


def send_command(ser, text, settle=0.4):
    """Send one firmware command and echo whatever it replies."""
    ser.write((text + "\n").encode())
    time.sleep(settle)
    while ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            print(f"  esp32: {line}")


def read_frame(ser, verbose=True):
    """Block until one complete frame arrives. Returns a dict or None."""
    header = None
    deadline = time.time() + 10.0
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        try:
            line = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            continue
        if line.startswith("#FRAME"):
            header = line
            break
        if line and verbose:
            print(f"  esp32: {line}")

    if header is None:
        return None

    meta = {}
    for token in header.split()[1:]:
        if "=" not in token:
            continue
        key, val = token.split("=", 1)
        meta[key] = val

    rate = int(meta.get("rate", 0))
    nch = int(meta.get("ch", 1))
    nsamp = int(meta.get("n", 0))
    elapsed_us = int(meta.get("us", 0))
    order = [int(c) for c in meta.get("order", "0").split(",") if c != ""]

    magic = ser.read(4)
    if magic != MAGIC:
        print(f"!! bad magic {magic!r} - frame dropped, resyncing", file=sys.stderr)
        return None

    nbytes = nsamp * nch * 2
    payload = ser.read(nbytes)
    if len(payload) != nbytes:
        print(f"!! short read {len(payload)}/{nbytes} - frame dropped", file=sys.stderr)
        return None

    chk_raw = ser.read(2)
    if len(chk_raw) == 2:
        want = struct.unpack("<H", chk_raw)[0]
        got = sum(payload) & 0xFFFF
        if want != got:
            print(f"!! checksum {got:#06x} != {want:#06x} - data corrupt", file=sys.stderr)
            return None

    flat = np.frombuffer(payload, dtype="<u2").astype(np.float64)
    data = flat.reshape(nsamp, nch).T  # -> [channel][sample]

    # The firmware reports the rate it actually achieved, not the one requested.
    actual_rate = rate
    if elapsed_us > 0 and nsamp > 1:
        actual_rate = (nsamp - 1) / (elapsed_us * 1e-6)

    return {
        "data": data,
        "rate": actual_rate,
        "rate_requested": rate,
        "channels": order,
        "elapsed_us": elapsed_us,
        # gap between neighbouring channels inside one cycle, measured by the
        # firmware. Zero for single-channel captures, where it has no meaning.
        "skew_us": float(meta.get("skew_us", 0.0)),
        "burst_us": float(meta.get("burst", 0)) / max(rate, 1) * 1e6 * max(nch, 1),
    }


def synth_frame(fs=1_000_000.0, duration_ms=20.0, f0=40_000.0,
                distance_cm=30.0, cycles=8, snr_db=20.0, seed=0):
    """Synthesise the capture a healthy Stage-1 rig should produce."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration_ms * 1e-3)
    t = np.arange(n) / fs

    def burst(t0, amp, tau):
        """40 kHz tone burst with the exponential tail a high-Q transducer has."""
        rel = t - t0
        gate = (rel >= 0) & (rel < cycles / f0 + 6 * tau)
        drive = np.where(rel < cycles / f0, 1.0, np.exp(-(rel - cycles / f0) / tau))
        return np.where(gate, amp * drive * np.sin(2 * np.pi * f0 * np.clip(rel, 0, None)), 0.0)

    # Transmitter ringing: huge, saturates the front end, decays over ~0.4 ms.
    signal = burst(0.0, 1.30, 80e-6)

    # Echo: round trip 2d/c, amplitude falls roughly as 1/r^2 for a flat plate.
    t_echo = 2.0 * (distance_cm / 100.0) / C_AIR
    echo_amp = 0.045 * (0.30 / (distance_cm / 100.0)) ** 2
    signal += burst(t_echo, echo_amp, 120e-6)

    noise_rms = echo_amp / (10 ** (snr_db / 20.0))
    signal += rng.normal(0.0, noise_rms, n)

    # Convert to what the ADC would report: 1.65 V bias, 12-bit, clipped at rails.
    volts = np.clip(1.65 + signal, 0.0, 3.3)
    counts = np.round(volts / 3.3 * 4095.0)

    print(f"  simulated: echo at t = {t_echo * 1e3:.3f} ms for d = {distance_cm:.1f} cm")
    return {
        "data": counts.reshape(1, n),
        "rate": fs,
        "rate_requested": fs,
        "channels": [6],
        "elapsed_us": int(n / fs * 1e6),
        "simulated": True,
        "t_echo_s": t_echo,
    }


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------


def plot_frame(frame, args):
    import matplotlib.pyplot as plt

    fs = frame["rate"]
    data = frame["data"]
    nch = data.shape[0]
    n = data.shape[1]
    t_ms = np.arange(n) / fs * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    title = f"fs = {fs / 1e3:.1f} kS/s   n = {n}   channels = {frame['channels']}"
    if frame.get("simulated"):
        title = "SIMULATED   " + title
    fig.suptitle(title)

    for ci in range(nch):
        counts = data[ci]
        volts = counts_to_volts(counts)
        ac = volts - np.mean(volts)
        label = f"ADC1_CH{frame['channels'][ci]}" if ci < len(frame["channels"]) else f"ch{ci}"

        filtered = bandpass(ac, fs, args.f0 * 0.6, args.f0 * 1.4)
        env = envelope(filtered)

        # Locate the echo, then analyse the spectrum right where it lives.
        echo_i = find_echo(env, fs, args.blank_ms * 1e-3, simple=args.simple_peak)
        t_echo = echo_i / fs
        dist_cm = t_echo * C_AIR / 2.0 * 100.0
        freqs, spec, w0, w1 = spectrum(ac, fs, echo_i, args.fft_window)

        axes[0].plot(t_ms, volts, lw=0.6, label=label)
        axes[1].plot(t_ms, env, lw=0.9, label=label)

        peak = np.max(spec) if np.max(spec) > 0 else 1.0
        spec_db = 20 * np.log10(np.maximum(spec, peak * 1e-6) / peak)
        axes[2].plot(freqs / 1e3, spec_db, lw=0.8, label=label)

        peak_hz, bin_hz = peak_frequency(freqs, spec)
        # Noise floor from the tail of the record, where neither the ringing
        # nor any echo should still be present.
        noise = np.median(env[int(n * 0.75):]) if n > 8 else 0.0
        snr_db = 20 * np.log10(env[echo_i] / noise) if noise > 0 else float("nan")

        print(f"  {label}: mean = {np.mean(volts):.3f} V  Vpp = {np.ptp(volts):.3f} V")
        print(f"  {label}: echo at t = {t_echo * 1e3:.3f} ms  ->  d = {dist_cm:.1f} cm  "
              f"(SNR {snr_db:.1f} dB)")
        print(f"  {label}: FFT peak = {peak_hz / 1e3:.2f} kHz "
              f"(bin {bin_hz:.0f} Hz, window {w0 / fs * 1e3:.2f}-{w1 / fs * 1e3:.2f} ms)")

        if ci == 0:
            axes[0].axvspan(w0 / fs * 1e3, w1 / fs * 1e3, color="orange", alpha=0.15,
                            label="FFT window")
        axes[1].plot(t_echo * 1e3, env[echo_i], "v", ms=8, label=f"echo {dist_cm:.1f} cm")

        if np.max(counts) >= 4094 or np.min(counts) <= 1:
            print(f"  {label}: !! CLIPPING at the rail - reduce gain")
        if abs(peak_hz - args.f0) > args.f0 * 0.1:
            print(f"  {label}: !! FFT peak is {peak_hz / 1e3:.1f} kHz, "
                  f"expected ~{args.f0 / 1e3:.0f} kHz")

    axes[0].set_ylabel("raw [V]")
    axes[0].set_xlabel("time [ms]")
    axes[0].axhline(1.65, color="grey", lw=0.5, ls="--")
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("envelope [V]")
    axes[1].set_xlabel("time [ms]")
    axes[1].grid(alpha=0.3)
    if frame.get("t_echo_s"):
        axes[1].axvline(frame["t_echo_s"] * 1e3, color="red", lw=0.8, ls="--",
                        label="true echo")

    axes[2].set_ylabel("magnitude [dB]")
    axes[2].set_xlabel("frequency [kHz]")
    axes[2].set_xlim(0, min(args.f0 * 4 / 1e3, fs / 2 / 1e3))
    axes[2].axvline(args.f0 / 1e3, color="red", lw=0.8, ls="--", label=f"{args.f0/1e3:.0f} kHz")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    if args.save:
        outdir = pathlib.Path(args.save)
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        np.savez(outdir / f"cap-{stamp}.npz", data=data, rate=fs,
                 channels=np.array(frame["channels"]))
        fig.savefig(outdir / f"cap-{stamp}.png", dpi=110)
        print(f"  saved {outdir / f'cap-{stamp}.npz'}")
    plt.show()


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. COM3")
    ap.add_argument("--baud", type=int, default=2000000)
    ap.add_argument("--simulate", action="store_true",
                    help="generate a synthetic capture; needs no hardware")
    ap.add_argument("--distance-cm", type=float, default=30.0,
                    help="simulated reflector distance")
    ap.add_argument("--f0", type=float, default=40_000.0, help="carrier frequency [Hz]")
    ap.add_argument("--blank-ms", type=float, default=0.6,
                    help="ignore this much of the record; it is transmitter ringing")
    ap.add_argument("--fft-window", type=int, default=1024,
                    help="samples around the echo to run the FFT over")
    ap.add_argument("--dc", action="store_true",
                    help="print the DC level at GPIO34 and exit (use as a voltmeter)")
    ap.add_argument("--simple-peak", action="store_true",
                    help="use plain largest-peak detection instead of CFAR")
    ap.add_argument("--rate", type=int,
                    help="ask the firmware for this ADC sample rate [Hz]")
    ap.add_argument("--samples", type=int,
                    help="ask the firmware for this many samples per capture")
    ap.add_argument("--probe", type=int,
                    help="probe pin to read: 34, 35, 32, 33, 36 or 39")
    ap.add_argument("--source", type=int,
                    help="set the firmware's test-source frequency [Hz]")
    ap.add_argument("--tone", type=int,
                    help="inject continuous DAC tone on GPIO26 at this Hz (RX gain probe, 'd' cmd)")
    ap.add_argument("--no-fire", action="store_true",
                    help="capture without firing TX (pair with --tone for a clean electrical probe)")
    ap.add_argument("--repeat", action="store_true", help="keep capturing until Ctrl-C")
    ap.add_argument("--save", help="directory to write .npz/.png captures into")
    args = ap.parse_args()

    if args.simulate:
        plot_frame(synth_frame(distance_cm=args.distance_cm, f0=args.f0), args)
        return 0

    if not args.port:
        ap.error("give --port COMx, or --simulate to run without hardware")

    try:
        import serial
    except ImportError:
        print("pyserial missing:  pip install -r tools/requirements.txt", file=sys.stderr)
        return 1

    print(f"opening {args.port} @ {args.baud}")
    with serial.Serial(args.port, args.baud, timeout=1.0) as ser:
        time.sleep(2.0)  # ESP32 resets when the port opens
        ser.reset_input_buffer()

        # Opening the port resets the board, so any rate/length settings have
        # to be re-sent here rather than persisting from a previous run.
        if args.probe:
            send_command(ser, f"c {int(args.probe)}")
        if args.rate:
            send_command(ser, f"f {int(args.rate)}")
        if args.samples:
            send_command(ser, f"n {int(args.samples)}")
        if args.source is not None:
            send_command(ser, f"s {int(args.source)}")
        if args.tone is not None:
            send_command(ser, f"d {int(args.tone)}")

        while True:
            ser.write(b"q\n" if args.no_fire else b"r\n")
            frame = read_frame(ser)
            if frame is None:
                print("no frame; is the firmware running? (send '?' for help)")
                if not args.repeat:
                    return 1
                continue
            if args.dc:
                # The ESP32's own ADC stands in for a voltmeter while probing.
                names = {6: "GPIO34", 7: "GPIO35", 4: "GPIO32"}
                for ci in range(frame["data"].shape[0]):
                    ch = frame["channels"][ci] if ci < len(frame["channels"]) else ci
                    volts = counts_to_volts(frame["data"][ci])
                    print(f"  {names.get(ch, f'ch{ch}'):8s} DC = {volts.mean():.3f} V   "
                          f"(min {volts.min():.3f}  max {volts.max():.3f}  "
                          f"ripple {np.ptp(volts) * 1000:.0f} mVpp)")
                return 0
            print(f"  requested {frame['rate_requested'] / 1e3:.0f} kS/s -> "
                  f"achieved {frame['rate'] / 1e3:.1f} kS/s")
            plot_frame(frame, args)
            if not args.repeat:
                return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped")
