#!/usr/bin/env python3
"""
Interactive CLI + live plot for 3‑sensor Honeywell RSC Teensy logger
-------------------------------------------------------------------
start [label]  – begin capture, create rsc_<label>_<YYYYmmdd_HHMMSS>.csv
stop            – stop capture, close CSV, freeze plot
(other fw cmds) – zero, rate <Hz>, info, dump, help, init <idx> …

Usage:
  python mask-logger-data-ingestion.py  COM4

Changes v2 (3‑sensor support)
-----------------------------
* Accepts 7‑field CSV frames:  t_us,P0,P1,P2,raw0,raw1,raw2
* Still tolerates legacy 5‑field frames (ignored with a warning)
* Logs & plots all three pressures (labels: S0, S1, S2)
* Header row in CSV now:  t_us,P0,P1,P2,raw0,raw1,raw2
"""
import argparse, threading, queue, datetime, csv, sys
import serial, serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as anim

CMD_PROMPT = ">>> "

# ──────────────────────────────────────────────────────────────
def find_default_port():
    for p in serial.tools.list_ports.comports():
        if "Teensy" in p.description or "USB Serial" in p.description:
            return p.device
    return None

# ──────────────────────────────────────────────────────────────
class SerialWorker(threading.Thread):
    """Background thread: pump serial → rx queue"""
    def __init__(self, port, baud, rxq, stop_evt):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, baud, timeout=0.1)
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
    fh   = open(name, "w", newline="")
    csv_wr = csv.writer(fh)
    csv_wr.writerow(["t_us","Pa_Global","Pa_Vertical","Pa_Horizontal","raw_Global","raw_Vertical","raw_Horizontal"])
    print(f"# logging → {name}")
    return fh, csv_wr, name

# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=find_default_port(),
                    help="Serial port (default: first Teensy found)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--window", type=int, default=30,
                    help="seconds shown in live plot")
    args = ap.parse_args()

    if not args.port:
        sys.exit("No port specified and no Teensy found.")

    stop_evt = threading.Event()
    rxq      = queue.Queue()
    worker   = SerialWorker(args.port, args.baud, rxq, stop_evt)
    worker.start()
    print(f"Opened {args.port} @ {args.baud} baud")
    print("Type 'start [label]' to begin, 'stop' to end, or firmware cmds (help/zero/rate …)\n")

    # ───── plot setup ─────
    buf_sec = args.window
    ts_buf, p_global_buf, p_vertical_buf, p_horizontal_buf = [], [], [], []
    fig, ax = plt.subplots()
    ln0, = ax.plot([], [], label="Global")
    ln1, = ax.plot([], [], label="Vertical")
    ln2, = ax.plot([], [], label="Horizontal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_xlim(-buf_sec, 0); ax.set_ylim(-100, 100)
    ax.legend()
    plt.show(block=False)

    # ───── capture state ─────
    capturing   = False
    log_fh      = None
    csv_wr      = None

    def update_plot(_):
        nonlocal ts_buf, p_global_buf, p_vertical_buf, p_horizontal_buf, capturing

        # ── drain serial queue ───────────────────────────
        while not rxq.empty():
            line = rxq.get()

            # ── show only firmware messages, not data ──
            if line.startswith("#"):
                print(line)

                # update capture state on stream start/stop
                if line.strip() == "# streaming ON":
                    capturing = True
                elif line.strip() == "# streaming OFF":
                    capturing = False
                continue                # done with this line

            parts = line.split(',')

            if len(parts) == 5:
                # Legacy 2‑sensor frame – ignore but warn once
                print("[WARN] Legacy 5‑field frame received – skipping")
                continue
            elif len(parts) != 7:
                print(f"[WARN] Unexpected field count ({len(parts)}): {parts}")
                continue

            # Parse seven‑field payload:  t_us,P0,P1,P2,raw0,raw1,raw2
            try:
                t_us  = int(parts[0])
                pa_global    = float(parts[1])
                pa_vertical    = float(parts[2])
                pa_horizontal    = float(parts[3])
                raw_global  = int(parts[4])
                raw_vertical  = int(parts[5])
                raw_horizontal  = int(parts[6])
            except ValueError:
                print("[WARN] couldn’t parse numbers:", parts)
                continue

            if csv_wr and capturing:
                csv_wr.writerow([t_us, pa_global, pa_vertical, pa_horizontal, raw_global, raw_vertical, raw_horizontal])

            # Scroll buffer regardless of ‘capturing’ so you always see live data
            ts_buf.append(t_us * 1e-6)
            p_global_buf.append(pa_global)
            p_vertical_buf.append(pa_vertical)
            p_horizontal_buf.append(pa_horizontal)

            while ts_buf and (ts_buf[-1] - ts_buf[0]) > buf_sec:
                ts_buf.pop(0); p_global_buf.pop(0); p_vertical_buf.pop(0); p_horizontal_buf.pop(0)

        # ── update plot ─────────────────────────────────
        if ts_buf:
            xmin = max(0, ts_buf[-1] - buf_sec)
            ln0.set_data(ts_buf, p_global_buf)
            ln1.set_data(ts_buf, p_vertical_buf)
            ln2.set_data(ts_buf, p_horizontal_buf)
            ax.set_xlim(xmin, xmin + buf_sec)
            ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
        return ln0, ln1, ln2

    ani = anim.FuncAnimation(fig, update_plot, interval=100, blit=True)

    # ───── CLI loop ─────
    try:
        while True:
            cmd_line = input(CMD_PROMPT).strip()
            if not cmd_line: continue

            if cmd_line.lower().startswith("start"):
                label = cmd_line.split(maxsplit=1)[1] if len(cmd_line.split()) > 1 else ""
                if capturing:
                    print("# already capturing – stop first"); continue
                log_fh, csv_wr, _ = open_log(label)
                ts_buf, p_global_buf, p_vertical_buf, p_horizontal_buf = [], [], [], []
                worker.ser.write(b"start\n")
                continue

            if cmd_line.lower() == "stop":
                if not capturing:
                    print("# not capturing"); continue
                worker.ser.write(b"stop\n")
                capturing = False
                log_fh.close()
                print(f"# closed {log_fh.name}")
                continue

            if cmd_line.lower() in ("exit", "quit"):
                break

            worker.ser.write((cmd_line + "\n").encode())

    except KeyboardInterrupt:
        print("\nCtrl-C received – exiting")

    # ───── teardown ─────
    if capturing and log_fh:
        log_fh.close(); print(f"# closed {log_fh.name}")
    stop_evt.set(); worker.join()
    plt.close(fig)

if __name__ == "__main__":
    main()
