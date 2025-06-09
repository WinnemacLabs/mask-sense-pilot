# Automated protocol script for pressure + particle logging

import time
import textwrap
from pressure_particles_ingestion import PressureParticlesSession

RAINBOW = textwrap.dedent("""
                        When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. 
                        The rainbow is a division of white light into many beautiful colors. 
                        These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. 
                        There is, according to legend, a boiling pot of gold at one end. 
                        People look, but no one ever finds it. 
                        When a man looks for something beyond reach, his friends say he is looking for the pot of gold at the end of the rainbow.
""")


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

    def prompt(msg, submsg=None):
        print("\n" + "-" * term_width)
        print(msg.center(term_width))
        if submsg:
            print(submsg.center(term_width))
        print("-" * term_width)
        return input("\nPress Enter to continue...")

    def safe_countdown(seconds, message=""):
        import sys
        import matplotlib.pyplot as plt

        for remaining in range(seconds, 0, -5):
            print(f"{message} {remaining}s remaining   ", end="\r")
            plt.pause(5)  # Allow GUI updates
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.readline().strip()
                if user_input.lower() == 'q':
                    print("\n[INFO] Quitting test early. Sending stop command to Teensy...")
                    session.stop_recording()
                    session.wait_for_serial_message("# streaming OFF", timeout=5)
                    print("[INFO] Teensy stopped. Exiting test.")
                    session.close()
                    sys.exit(0)
        print(" " * 60, end="\r")

    import select
    try:
        prompt("Fit the mask on your face now.", "Adjust for comfort and seal.")
        lbl = f"P{participant}_{mask_label}{leak_tag}_zeroing"
        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)
        input("Hold your breath for sensor zeroing. Remain still and silent. Press enter when you've started holding your breath.")
        session.send_teensy("zero")
        print("\nZeroing (keep holding breath)\n")
        session.wait_for_serial_message("# new zeros:", timeout=5)
        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)
        print("\nSensors zeroed.\n")

        prompt("Start quiet breathing.", "Breathe normally for 3 minutes.")
        lbl = f"P{participant}_{mask_label}{leak_tag}_quiet_breathing"
        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)
        safe_countdown(3*60, "Quiet breathing")
        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)
        print("\nQuiet breathing complete.\n")

        prompt("Start deep breathing.", "Breathe deeply (not hyperventilating) for 1 minute.")
        lbl = f"P{participant}_{mask_label}{leak_tag}_deep_breathing"
        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)
        safe_countdown(60, "Deep breathing")
        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)
        print("\nDeep breathing complete.\n")

        prompt("Read the Rainbow Passage aloud.", "Read clearly and at a natural pace.")
        lbl = f"P{participant}_{mask_label}{leak_tag}_rainbow"
        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)
        print("\n" + "-" * term_width)
        print("RAINBOW PASSAGE:".center(term_width))
        print("-" * term_width)
        print(RAINBOW)
        print("-" * term_width)
        input("\nPress Enter when finished reading...")
        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)
        print("\nReading complete.\n")

        prompt("Prepare for Functional Residual Capacity (FRC) Reset.", "When instructed, take a gentle breath in, then exhale passively and completely without forcing additional air out.")
        lbl = f"P{participant}_{mask_label}{leak_tag}_frc_reset"

        print("Take a gentle breath in, then exhale passively and completely without forcing additional air out. Pause for 2-3 seconds, press enter, and then resume quiet breathing when prompted.")
        
        session.start_recording(f"P{participant}_{mask_label}{leak_tag}_frc_reset_init")
        session.wait_for_serial_message("# streaming ON", timeout=5)

        input("Press enter after pausing 2-3 seconds")
        time.sleep(1)
        prompt("Resume quiet breathing")

        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)

        session.start_recording(lbl)
        session.wait_for_serial_message("# streaming ON", timeout=5)
        
        safe_countdown(30, "Quiet breathing from FRC reset. Perform another FRC reset when prompted.")
        print("Perform final FRC reset. Take a gentle breath in, then exhale passively and completely without forcing additional air out. Pause for 2-3 seconds.")
        input("Press Enter after pausing 2-3 seconds.")

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
