#!/usr/bin/env python3
"""
Interactive CLI + live plot for a 3-sensor Honeywell RSC Teensy logger
plus a dual-channel WRPAS condensing particle counter

At startup, prompts you to select the Teensy and WRPAS serial ports.
Adds optional WRPAS debugging, CLI passthrough, disables flow control,
and plots pressures and particle concentrations on dual axes.

Commands:
  start [label]    – begin capture, create rsc_<label>_<YYYYmmdd_HHMMSS>.csv
  stop             – stop capture, close CSV, freeze plot
  wrpas <command>  – send arbitrary command to WRPAS and echo response
  debug on/off     – toggle WRPAS raw logging and parsed values
  exit, quit       – exit script

Usage:
  python3 pressure-particles-ingestion.py [--port /dev/cu...] [--wrpas /dev/cu...] [--baud 921600]
                       [--window 30] [--debug-wrpas]
"""
import argparse
import threading
import queue  
import datetime
import csv
import sys
import re
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as anim

CMD_PROMPT = ">>> "

# ──────────────────────────────────────────────────────────────
def choose_port(prompt, ports, allow_skip=False):
    if not ports:
        print(f"# No available ports for {prompt}")
        return None
    print(f"\nAvailable ports for {prompt}:")
    for i, p in enumerate(ports):
        print(f"  {i}: {p.device} — {p.description}")
    sel = input(f"Select {prompt} port by index{' (blank to skip)' if allow_skip else ''}: ").strip()
    if sel == "" and allow_skip:
        return None
    try:
        return ports[int(sel)].device
    except Exception:
        print("# Invalid selection, skipping.")
        return None

# ──────────────────────────────────────────────────────────────
class SerialWorker(threading.Thread):
    """Background thread: pump serial → rx queue"""
    def __init__(self, port, baud, rxq, stop_evt):
        super().__init__(daemon=True)
        # disable XON/XOFF and RTS/CTS flow control for compatibility
        self.ser = serial.Serial(
            port,
            baud,
            timeout=0.1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        self.rxq = rxq
        self.stop_evt = stop_evt
    def run(self):
        buf = b""
        while not self.stop_evt.is_set():
            data = self.ser.read(1024)
            if data:
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    self.rxq.put(line.decode(errors="ignore").strip())
        self.ser.close()

# ──────────────────────────────────────────────────────────────
def open_log(label: str):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"rsc_{label}_{ts}.csv" if label else f"rsc_{ts}.csv"
    fh = open(name, "w", newline="")
    csv_wr = csv.writer(fh)
    csv_wr.writerow([
        "t_us",
        "Pa_Global","Pa_Vertical","Pa_Horizontal",
        "raw_Global","raw_Vertical","raw_Horizontal",
        "mask_particles","ambient_particles"
    ])
    print(f"# logging → {name}")
    return fh, csv_wr, name

# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="Serial port for Teensy")
    ap.add_argument("--wrpas", help="Serial port for WRPAS")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--window", type=int, default=30,
                    help="seconds shown in live plot")
    ap.add_argument("--debug-wrpas", action="store_true",
                    help="Print raw and parsed WRPAS data to console")
    args = ap.parse_args()

    # Interactive port selection if flags missing
    available = list(serial.tools.list_ports.comports())
    if not args.port:
        args.port = choose_port("Teensy", available)
    if not args.wrpas:
        candidates = [p for p in available]
        args.wrpas = choose_port("WRPAS", candidates, allow_skip=True)

    if not args.port:
        sys.exit("No Teensy port selected. Exiting.")
    if args.wrpas:
        print(f"Using WRPAS port: {args.wrpas}")
    else:
        print("# WRPAS not selected; particle counts disabled.")

    # Start serial threads
    stop_evt = threading.Event()
    rxq = queue.Queue()
    worker = SerialWorker(args.port, args.baud, rxq, stop_evt)
    worker.start()

    wrpas_worker = None
    wrpas_q = None
    wrpas_evt = None
    dbg = args.debug_wrpas
    if args.wrpas:
        wrpas_evt = threading.Event()
        wrpas_q = queue.Queue()
        wrpas_worker = SerialWorker(args.wrpas, 115200, wrpas_q, wrpas_evt)
        wrpas_worker.start()
        if dbg:
            print("# WRPAS debug ON: raw lines and parsed values will be printed")

    print(f"Opened Teensy @ {args.port} @ {args.baud} baud")
    if wrpas_worker:
        print(f"Opened WRPAS @ {args.wrpas} @ 115200 baud")
    print("Type 'start', 'stop', 'wrpas <cmd>', 'debug on/off', or 'exit'\n")

    # Data state and buffers
    buf_sec = args.window
    ts_buf = []
    p_g = []; p_v = []; p_h = []
    conc1_buf = []
    conc2_buf = []
    capturing = False
    log_fh = None
    csv_wr = None
    conc1 = None
    conc2 = None

    # Set up plot with twin axes
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ln0, = ax.plot([], [], label="Global")
    ln1, = ax.plot([], [], label="Vertical")
    ln2, = ax.plot([], [], label="Horizontal")
    ln3, = ax2.plot([], [], linestyle='--', label="Conc1")
    ln4, = ax2.plot([], [], linestyle='--', label="Conc2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (Pa)")
    ax2.set_ylabel("Particle Conc.")
    ax.set_xlim(-buf_sec, 0)
    ax.set_ylim(-100, 100)
    ax2.set_ylim(0, 5000)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show(block=False)

    def update_plot(_):
        nonlocal ts_buf, p_g, p_v, p_h, conc1_buf, conc2_buf
        nonlocal capturing, log_fh, csv_wr, conc1, conc2

        # Drain and debug WRPAS
        if wrpas_worker:
            while not wrpas_q.empty():
                line = wrpas_q.get()
                if dbg:
                    print(f"[WRPAS RAW] {line}")
                m1 = re.search(r"Conc1:\s*([\d\.]+)", line)
                m2 = re.search(r"Conc2:\s*([\d\.]+)", line)
                if m1 and m2:
                    conc1 = float(m1.group(1))
                    conc2 = float(m2.group(1))
                    if dbg:
                        print(f"[WRPAS PARSED] Conc1={conc1}, Conc2={conc2}")

        # Drain Teensy
        while not rxq.empty():
            line = rxq.get()
            if line.startswith("#"):
                print(line)
                if line.strip() == "# streaming ON": capturing = True
                if line.strip() == "# streaming OFF": capturing = False
                continue

            parts = line.split(',')
            if len(parts) != 7:
                print(f"[WARN] Unexpected frame: {parts}")
                continue
            try:
                t = int(parts[0])
                vals = list(map(float, parts[1:4]))
                raws = list(map(int, parts[4:7]))
            except ValueError:
                print(f"[WARN] Parse error: {parts}")
                continue

            # Log to CSV
            if csv_wr and capturing:
                if dbg:
                    print(f"[WRPAS CSV] Writing Conc1={conc1}, Conc2={conc2}")
                csv_wr.writerow([
                    t, *vals, *raws,
                    conc1 if conc1 is not None else "",
                    conc2 if conc2 is not None else ""
                ])

            # Append data to buffers
            ts = t * 1e-6
            ts_buf.append(ts)
            p_g.append(vals[0]); p_v.append(vals[1]); p_h.append(vals[2])
            conc1_buf.append(conc1 if conc1 is not None else float('nan'))
            conc2_buf.append(conc2 if conc2 is not None else float('nan'))

            # Trim old data
            while ts_buf and ts_buf[-1] - ts_buf[0] > buf_sec:
                ts_buf.pop(0)
                p_g.pop(0); p_v.pop(0); p_h.pop(0)
                conc1_buf.pop(0); conc2_buf.pop(0)

        # Update plots
        if ts_buf:
            x0 = max(0, ts_buf[-1] - buf_sec)
            ln0.set_data(ts_buf, p_g); ln1.set_data(ts_buf, p_v); ln2.set_data(ts_buf, p_h)
            ax.set_xlim(x0, x0 + buf_sec); ax.relim(); ax.autoscale_view(scalex=False)
            ln3.set_data(ts_buf, conc1_buf); ln4.set_data(ts_buf, conc2_buf)
            ax2.relim(); ax2.autoscale_view(scalex=False)
        return ln0, ln1, ln2, ln3, ln4

    ani = anim.FuncAnimation(fig, update_plot, interval=100, blit=True)

    try:
        while True:
            cmd = input(CMD_PROMPT).strip()
            if not cmd: continue
            low = cmd.lower()

            if low.startswith("start"):
                if capturing:
                    print("# already capturing")
                    continue
                label = cmd.split(maxsplit=1)[1] if ' ' in cmd else ""
                log_fh, csv_wr, _ = open_log(label)
                ts_buf.clear(); p_g.clear(); p_v.clear(); p_h.clear()
                conc1_buf.clear(); conc2_buf.clear()
                if wrpas_worker: wrpas_worker.ser.write(b"CCT 1\r\n")
                worker.ser.write(b"start\r\n")
                continue

            if low == "stop":
                if not capturing:
                    print("# not capturing")
                    continue
                if wrpas_worker: wrpas_worker.ser.write(b"CCT 0\r\n")
                worker.ser.write(b"stop\r\n")
                capturing = False
                log_fh.close()
                print(f"# closed {log_fh.name}")
                continue

            if low.startswith("wrpas "):
                if not wrpas_worker:
                    print("# No WRPAS connected")
                    continue
                cmd_bytes = cmd.split(' ', 1)[1].encode() + b"\r\n"
                wrpas_worker.ser.write(cmd_bytes)
                continue

            if low == "debug on": dbg = True; print("WRPAS debug ON"); continue
            if low == "debug off": dbg = False; print("WRPAS debug OFF"); continue

            if low in ("exit", "quit"): break
            worker.ser.write((cmd + "\r\n").encode())

    except KeyboardInterrupt:
        print("\nExiting…")

    # Teardown
    if capturing and log_fh:
        log_fh.close(); print(f"# closed {log_fh.name}")
    stop_evt.set(); worker.join()
    if wrpas_worker: wrpas_evt.set(); wrpas_worker.join()
    plt.close(fig)

if __name__ == "__main__":
    main()
