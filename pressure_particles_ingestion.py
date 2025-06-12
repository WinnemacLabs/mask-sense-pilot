#!/usr/bin/env python3
"""
Interactive CLI + live plot for a 3-sensor Honeywell RSC Teensy logger
plus a dual-channel WRPAS condensing particle counter

At startup, prompts you to select the Teensy and WRPAS serial ports.
Adds optional WRPAS debugging, CLI passthrough, disables flow control,
and plots pressures and particle concentrations on dual axes.

Commands:
  start [label] [--participant P1] [--mask N95] [--fit leak|no_leak] [--exercise breathing]
                   – begin capture, create rsc_<label>_<YYYYmmdd_HHMMSS>.csv with metadata
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

__all__ = [
    "PressureParticlesSession",
    "choose_port",
    "main",
]

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
def open_log(label: str, participant: str = "unknown", mask: str = "unknown", leak_condition: str = "unknown", exercise: str = "unknown"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"data/rsc_{label}.csv" if label else f"data/rsc_{ts}.csv"
    fh = open(name, "w", newline="")
    csv_wr = csv.writer(fh)
    # write comments with metadata
    csv_wr.writerow([f"#Label, {label}"])
    csv_wr.writerow([f"#Participant, {participant}"])
    csv_wr.writerow([f"#Mask, {mask}"])
    csv_wr.writerow([f"#Leak Condition, {leak_condition}"])
    csv_wr.writerow([f"#Exercise, {exercise}"])
    csv_wr.writerow([f"#Date/time, {ts}"])
    csv_wr.writerow([
        "t_us",
        "Pa_Global","Pa_Vertical","Pa_Horizontal",
        "raw_Global","raw_Vertical","raw_Horizontal",
        "mask_particles","ambient_particles"
    ])
    print(f"# logging → {name}")
    return fh, csv_wr, name


class PressureParticlesSession:
    """Manage serial communication and live plotting for automated scripts."""

    def __init__(self, port=None, wrpas=None, baud=921600, window=30, debug_wrpas=False):
        # Interactive port selection if missing
        available = list(serial.tools.list_ports.comports())
        if not port:
            port = choose_port("Teensy", available)
        if not wrpas:
            candidates = [p for p in available]
            wrpas = choose_port("WRPAS", candidates, allow_skip=True)

        if not port:
            raise SystemExit("No Teensy port selected. Exiting.")
        if wrpas:
            print(f"Using WRPAS port: {wrpas}")
        else:
            print("# WRPAS not selected; particle counts disabled.")

        self.baud = baud
        self.debug_wrpas = debug_wrpas

        # Start serial threads
        self.stop_evt = threading.Event()
        self.rxq = queue.Queue()
        self.worker = SerialWorker(port, baud, self.rxq, self.stop_evt)
        self.worker.start()

        self.wrpas_worker = None
        self.wrpas_q = None
        self.wrpas_evt = None
        if wrpas:
            self.wrpas_evt = threading.Event()
            self.wrpas_q = queue.Queue()
            self.wrpas_worker = SerialWorker(wrpas, 115200, self.wrpas_q, self.wrpas_evt)
            self.wrpas_worker.start()
            if debug_wrpas:
                print("# WRPAS debug ON: raw lines and parsed values will be printed")

        print(f"Opened Teensy @ {port} @ {baud} baud")
        if self.wrpas_worker:
            print(f"Opened WRPAS @ {wrpas} @ 115200 baud")

        # Data state
        self.buf_sec = window
        self.ts_buf = []
        self.p_g = []; self.p_v = []; self.p_h = []
        self.conc1_buf = []; self.conc2_buf = []
        self.capturing = False
        self.log_fh = None
        self.csv_wr = None
        self.conc1 = None
        self.conc2 = None

        # Plot setup
        self.fig, self.ax = plt.subplots()
        self.ax2 = self.ax.twinx()
        self.ln0, = self.ax.plot([], [], label="Global")
        self.ln1, = self.ax.plot([], [], label="Vertical")
        self.ln2, = self.ax.plot([], [], label="Horizontal")
        self.ln3, = self.ax2.plot([], [], linestyle='--', label="Mask Particles")
        self.ln4, = self.ax2.plot([], [], linestyle='--', label="Ambient Particles")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pressure (Pa)")
        self.ax2.set_ylabel("Particle Conc.")
        self.ax.set_xlim(-self.buf_sec, 0)
        self.ax.set_ylim(-100, 100)
        self.ax2.set_ylim(0, 5000)
        self.ax.legend(loc='upper left')
        self.ax2.legend(loc='upper right')
        plt.show(block=False)

        self.ani = anim.FuncAnimation(self.fig, self._update_plot, interval=100, blit=True, cache_frame_data=False)

    # ---------------------------------------------
    def _update_plot(self, _):
        # Drain and debug WRPAS
        if self.wrpas_worker:
            while not self.wrpas_q.empty():
                line = self.wrpas_q.get()
                if self.debug_wrpas:
                    print(f"[WRPAS RAW] {line}")
                m1 = re.search(r"Conc1:\s*([\d\.]+)", line)
                m2 = re.search(r"Conc2:\s*([\d\.]+)", line)
                if m1 and m2:
                    self.conc1 = float(m1.group(1))
                    self.conc2 = float(m2.group(1))
                    if self.debug_wrpas:
                        print(f"[WRPAS PARSED] Conc1={self.conc1}, Conc2={self.conc2}")

        # Drain Teensy data
        while not self.rxq.empty():
            line = self.rxq.get()
            if line.startswith("#"):
                print(line)
                if line.strip() == "# streaming ON":
                    self.capturing = True
                if line.strip() == "# streaming OFF":
                    self.capturing = False
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

            if self.csv_wr and self.capturing:
                if self.debug_wrpas:
                    print(f"[WRPAS CSV] Writing Conc1={self.conc1}, Conc2={self.conc2}")
                self.csv_wr.writerow([
                    t, *vals, *raws,
                    self.conc1 if self.conc1 is not None else "",
                    self.conc2 if self.conc2 is not None else "",
                ])

            ts = t * 1e-6
            self.ts_buf.append(ts)
            self.p_g.append(vals[0]); self.p_v.append(vals[1]); self.p_h.append(vals[2])
            self.conc1_buf.append(self.conc1 if self.conc1 is not None else float('nan'))
            self.conc2_buf.append(self.conc2 if self.conc2 is not None else float('nan'))

            while self.ts_buf and self.ts_buf[-1] - self.ts_buf[0] > self.buf_sec:
                self.ts_buf.pop(0)
                self.p_g.pop(0); self.p_v.pop(0); self.p_h.pop(0)
                self.conc1_buf.pop(0); self.conc2_buf.pop(0)

        if self.ts_buf:
            x0 = max(0, self.ts_buf[-1] - self.buf_sec)
            self.ln0.set_data(self.ts_buf, self.p_g)
            self.ln1.set_data(self.ts_buf, self.p_v)
            self.ln2.set_data(self.ts_buf, self.p_h)
            self.ax.set_xlim(x0, x0 + self.buf_sec)
            self.ax.relim(); self.ax.autoscale_view(scalex=False)
            self.ln3.set_data(self.ts_buf, self.conc1_buf)
            self.ln4.set_data(self.ts_buf, self.conc2_buf)
            self.ax2.relim(); self.ax2.autoscale_view(scalex=False)

        return self.ln0, self.ln1, self.ln2, self.ln3, self.ln4

    # ---------------------------------------------
    def start_recording(self, label="", participant="unknown", mask_type="unknown", 
                       fit_condition="unknown", exercise="unknown"):
        if self.capturing:
            print("# already capturing")
            return
        self.log_fh, self.csv_wr, _ = open_log(
            label=label, 
            participant=participant, 
            mask=mask_type, 
            leak_condition=fit_condition, 
            exercise=exercise
        )
        self.ts_buf.clear(); self.p_g.clear(); self.p_v.clear(); self.p_h.clear()
        self.conc1_buf.clear(); self.conc2_buf.clear()
        if self.wrpas_worker:
            self.wrpas_worker.ser.write(b"CCT 1\r\n")
        self.worker.ser.write(b"start\r\n")
        self.capturing = True  # Set immediately after sending start

    def stop_recording(self):
        if not self.capturing:
            print("# not capturing")
            return
        if self.wrpas_worker:
            self.wrpas_worker.ser.write(b"CCT 0\r\n")
        self.worker.ser.write(b"stop\r\n")
        self.capturing = False  # Set immediately after sending stop
        if self.log_fh:
            self.log_fh.close()
            print(f"# closed {self.log_fh.name}")

    def send_teensy(self, cmd: str):
        self.worker.ser.write((cmd + "\r\n").encode())

    def close(self):
        if self.capturing:
            self.stop_recording()
        self.stop_evt.set(); self.worker.join()
        if self.wrpas_worker:
            self.wrpas_evt.set(); self.wrpas_worker.join()
        plt.close(self.fig)

    # ---------------------------------------------
    def run_cli(self):
        print("Type 'start [label]', 'stop', 'wrpas <cmd>', 'debug on/off', or 'exit'")
        print("Extended start syntax: start [label] [--participant P1] [--mask N95] [--fit leak|no_leak] [--exercise breathing]\n")
        try:
            while True:
                cmd = input(CMD_PROMPT).strip()
                if not cmd:
                    continue
                low = cmd.lower()

                if low.startswith("start"):
                    # Parse extended command format
                    parts = cmd.split()
                    label = parts[1] if len(parts) > 1 and not parts[1].startswith('--') else ""
                    
                    # Default values
                    participant = "unknown"
                    mask_type = "unknown"
                    fit_condition = "unknown"
                    exercise = "unknown"
                    
                    # Parse optional parameters
                    i = 2 if label else 1
                    while i < len(parts):
                        if parts[i] == "--participant" and i + 1 < len(parts):
                            participant = parts[i + 1]
                            i += 2
                        elif parts[i] == "--mask" and i + 1 < len(parts):
                            mask_type = parts[i + 1]
                            i += 2
                        elif parts[i] == "--fit" and i + 1 < len(parts):
                            fit_condition = parts[i + 1]
                            i += 2
                        elif parts[i] == "--exercise" and i + 1 < len(parts):
                            exercise = parts[i + 1]
                            i += 2
                        else:
                            i += 1
                    
                    self.start_recording(label, participant, mask_type, fit_condition, exercise)
                    continue
                if low == "stop":
                    self.stop_recording()
                    continue
                if low.startswith("wrpas "):
                    if not self.wrpas_worker:
                        print("# No WRPAS connected")
                        continue
                    cmd_bytes = cmd.split(' ', 1)[1].encode() + b"\r\n"
                    self.wrpas_worker.ser.write(cmd_bytes)
                    continue
                if low == "debug on":
                    self.debug_wrpas = True
                    print("WRPAS debug ON")
                    continue
                if low == "debug off":
                    self.debug_wrpas = False
                    print("WRPAS debug OFF")
                    continue
                if low in ("exit", "quit"):
                    break
                self.send_teensy(cmd)
        except KeyboardInterrupt:
            print("\nExiting…")
        finally:
            self.close()

    # ---------------------------------------------
    def wait_for_serial_message(self, substring, timeout=5):
        import time
        start = time.time()
        while time.time() - start < timeout:
            while not self.rxq.empty():
                line = self.rxq.get()
                if (line.startswith("#")):
                    print(line)
                if substring in line:
                    return True
            time.sleep(0.05)
        print(f"[WARN] Did not see '{substring}' in serial output within {timeout} seconds.")
        return False


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

    session = PressureParticlesSession(
        port=args.port,
        wrpas=args.wrpas,
        baud=args.baud,
        window=args.window,
        debug_wrpas=args.debug_wrpas,
    )
    session.run_cli()

if __name__ == "__main__":
    main()
