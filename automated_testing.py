# Automated protocol script for pressure + particle logging

import time
import textwrap
from pressure_particles_ingestion import PressureParticlesSession
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

RAINBOW = textwrap.dedent("""
                        When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. 
                        The rainbow is a division of white light into many beautiful colors. 
                        These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. 
                        There is, according to legend, a boiling pot of gold at one end. 
                        People look, but no one ever finds it. 
                        When a man looks for something beyond reach, his friends say he is looking for the pot of gold at the end of the rainbow.
""")

def beep(num_times=2):
    """Beep sound to signal important events."""
    for _ in range(num_times):
        os.system('tput bel') 
        time.sleep(0.2) 

def clear_screen():
    sys.stdout.write('\033c')
    sys.stdout.flush()

def countdown(seconds, message=""):
    """Simple textual countdown timer that allows GUI updates with minimal flicker."""
    import matplotlib.pyplot as plt
    for remaining in range(seconds, 0, -1):
        print(f"{message} {remaining}s remaining", end="\r")
        plt.pause(1)  # Only pause once per second
    print(" " * 40, end="\r")


def run_mask(session, participant, mask_label, with_leak=False):
    import shutil
    import sys
    leak_tag = "_leak" if with_leak else ""
    term_width = shutil.get_terminal_size((80, 20)).columns
    border = "=" * term_width
    mask_display = f"Mask: {mask_label}{leak_tag}"
    part_display = f"Participant: {participant}"
    header = f"{part_display} | {mask_display}"
    print("\n" * 2)
    print(border)
    print(header.center(term_width))
    print(border)
    print()

    def prompt(msg, submsg=None, prompt_msg="\nPress Enter to continue...", prompt_enter=True):
        print("\n" + "-" * term_width)
        print(msg.center(term_width))
        if submsg:
            print(submsg.center(term_width))
        print("-" * term_width)
        if prompt_enter:
            print(prompt_msg)
            return input()
        else:
            return

    def safe_countdown(seconds, message=""):
        import sys
        import matplotlib.pyplot as plt

        print("\n" + "=" * term_width)
        print(message.center(term_width))
        print("=" * term_width)

        for remaining in range(seconds, 0, -5):
            print(f"{remaining}s REMAINING   ", end="\r")
            # make progress bar
            bar_length = 30
            filled_length = int(bar_length * (seconds - remaining) / seconds)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"[{bar}] {remaining}s remaining", end="\r")
            # allow plot updates
            plt.pause(5)
            # Check for user input to quit
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.readline().strip()
                if user_input.lower() == 'q':
                    print("\n[INFO] Quitting test early. Sending stop command to Teensy...")
                    session.stop_recording()
                    session.wait_for_serial_message("# streaming OFF", timeout=5)
                    print("[INFO] Teensy stopped. Exiting test.")
                    return
        print(" " * 60, end="\r")

    def verify_zero(participant, mask_label, leak_tag, duration=5):
        """Capture breath-hold data to verify zeroing, saving data, results, and plots only on PASS."""

        folder_path = f"data/P{participant}/zeroing"
        os.makedirs(folder_path, exist_ok=True)

        while True:
            prompt("Hold your breath again for zero verification. Press enter when holding...", prompt_msg="")
            lbl = f"P{participant}_{mask_label}{leak_tag}_zero_check"

            session.start_recording(lbl)
            session.wait_for_serial_message("# streaming ON", timeout=5)
            countdown(duration, "Verifying zero")
            session.stop_recording()
            session.wait_for_serial_message("# streaming OFF", timeout=5)

            if not session.p_g:
                print("[WARN] No data captured for verification.")
                continue

            t = np.array(session.ts_buf) - session.ts_buf[0]
            p = np.array(session.p_g)
            mean = float(p.mean())
            std = float(p.std())

            filename_base = f"P{participant}_{mask_label}{leak_tag}_zero_check"

            print(f"Zero check mean: {mean:.3f} Pa | stdev: {std:.3f} Pa")

            if abs(mean) <= 0.2 and std <= 0.5:
                # Save raw data
                raw_data_path = os.path.join(folder_path, f"{filename_base}_raw.npy")
                np.save(raw_data_path, {'time': t, 'pressure': p})

                # Save results
                results_path = os.path.join(folder_path, f"{filename_base}_results.txt")
                with open(results_path, 'w') as f:
                    f.write(f"Mean: {mean:.3f} Pa\nStandard Deviation: {std:.3f} Pa\n")

                # Plot and save graph
                plt.figure()
                plt.plot(t, p)
                plt.xlabel("Time (s)")
                plt.ylabel("Pressure (Pa)")
                plt.title("Zero Verification")
                plot_path = os.path.join(folder_path, f"{filename_base}_plot.png")
                plt.savefig(plot_path)
                plt.show(block=False)

                print("\nZero verification PASS. You can breathe again!\n")
                time.sleep(3)
                return
            else:
                print("\nZero verification FAIL — re-zeroing\n")
                input("\nHold your breath and press Enter to re-zero...")
                session.send_teensy("zero")
                session.wait_for_serial_message("# new zeros:", timeout=5)


    import select
    try:
        clear_screen()
        beep(2)
        prompt("Fit the mask on your face now.", "Adjust for comfort and seal.")
        lbl = f"P{participant}_{mask_label}{leak_tag}_zeroing"
        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)

        clear_screen()
        beep(2)
        prompt("Hold your breath for sensor zeroing. Remain still and silent. Press enter when you've started holding your breath.", prompt_msg="")
        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)
        session.send_teensy("zero")
        print("\nZeroing (keep holding breath)\n")
        session.wait_for_serial_message("# new zeros:", timeout=5)
        print("\nSensors zeroed. You can breathe now!\n")
        time.sleep(3)

        beep(1)

        clear_screen()
        verify_zero(participant, mask_label, leak_tag)

        clear_screen()
        beep(2)
        prompt("Start quiet breathing.", "Breathe normally for 3 minutes.", prompt_msg="Press Enter to start.")
        lbl = f"P{participant}_{mask_label}{leak_tag}_quiet_breathing"
        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)
        safe_countdown(3*60, "Quiet breathing")
        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)
        print("\nQuiet breathing complete.\n")

        clear_screen()
        beep(2)
        prompt("Start deep breathing.", "Breathe deeply (not hyperventilating) for 1 minute.")
        lbl = f"P{participant}_{mask_label}{leak_tag}_deep_breathing"
        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)
        safe_countdown(60, "Deep breathing")
        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)
        print("\nDeep breathing complete.\n")

        clear_screen()
        beep(2)
        prompt("Read the Rainbow Passage aloud.", "Read clearly and at a natural pace.", prompt_enter=False)
        time.sleep(5)
        lbl = f"P{participant}_{mask_label}{leak_tag}_rainbow"
        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)
        print("\n" + "-" * term_width)
        print("RAINBOW PASSAGE:".center(term_width))
        print("-" * term_width)
        print(RAINBOW)
        print("-" * term_width)
        prompt("\nPress Enter when finished reading...", prompt_msg = "")
        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)
        print("\nReading complete.\n")

        clear_screen()
        beep(2)
        prompt("Functional Residual Capacity (FRC) Reset.", "Take a gentle breath in, then exhale passively and completely without forcing additional air out.", prompt_enter=False)
        time.sleep(5)
        lbl = f"P{participant}_{mask_label}{leak_tag}_frc_reset"

        session.start_recording(f"P{participant}_{mask_label}{leak_tag}_frc_reset_init")
        session.wait_for_serial_message("# streaming ON", timeout=5)

        prompt("\nPress enter after pausing at the end of your exhale for 2-3 seconds and resume quiet breathing when prompted\n", prompt_msg="")
        time.sleep(1)
        beep(1)
        prompt("\nResume quiet breathing\n", prompt_enter=False)

        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)

        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)
        
        safe_countdown(30, "\nQuiet breathing from FRC reset. Perform another FRC reset when prompted.\n")
        beep(1)
        prompt("\n\n\nPerform final FRC reset. Take a gentle breath in, then exhale passively and completely without forcing additional air out. Pause for 2-3 seconds and press Enter.\n", prompt_msg="")

        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)

        print("\nFRC reset and quiet breathing complete.\n")

    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user. Teensy has been stopped.")
        return


def main():
    participant = input("Participant number: ").strip()
    mask1 = input("Mask 1 label: ").strip() or "mask1"
    mask2 = input("Mask 2 label: ").strip() or "mask2"

    session = PressureParticlesSession()
    # Stop teensy at the start of the session
    print("Resetting Teensy...\n")
    session.send_teensy("stop")
    session.wait_for_serial_message("OFF", timeout=5)
    session.send_teensy("rate 1000")
    session.wait_for_serial_message("1000", timeout=5)
    time.sleep(1) 

    print("\nConfirm setup:")
    input("\n1. WRPAS wick saturated?")
    input("\n2. Tubes connected to correct ports?")

    print("Beginning test!")

    try:
        run_mask(session, participant, mask1, with_leak=False)
        print("Apply polyfil leak to right side of mask 1 and press Enter.")
        input()
        run_mask(session, participant, mask1, with_leak=True)

        print("Switch to mask 2 and press Enter when ready.")
        input()
        run_mask(session, participant, mask2, with_leak=False)
        print("Apply polyfil leak to right side of mask 2 and press Enter.")
        input()
        run_mask(session, participant, mask2, with_leak=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
