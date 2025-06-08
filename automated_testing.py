# Automated protocol script for pressure + particle logging

import time
import textwrap
from pressure_particles_ingestion import PressureParticlesSession

RAINBOW = textwrap.dedent("""
When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it.When a man looks for something beyond reach, his friends say he is looking for the pot of gold at the end of the rainbow.
""")


def countdown(seconds, message=""):
    """Simple textual countdown timer."""
    for remaining in range(seconds, 0, -1):
        print(f"{message} {remaining}s remaining", end="\r")
        time.sleep(1)
    print(" " * 40, end="\r")


def run_mask(session, participant, mask_label, with_leak=False):
    """Guide participant through recordings for one mask."""
    leak_tag = "_leak" if with_leak else ""

    print("Participant: Fit the mask on your face now.")
    input("Press Enter when the mask feels comfortable...")

    input("Hold your breath and press Enter to begin sensor zeroing...")
    session.send_teensy("zero")
    countdown(3, "Zeroing—keep holding")
    print("Sensors zeroed.\n")

    input("Press Enter to start quiet breathing...")
    lbl = f"P{participant}_{mask_label}{leak_tag}_quiet_breathing"
    session.start_recording(lbl)
    print("Breathe quietly for 3 minutes.")
    countdown(3 * 60, "Quiet breathing")
    session.stop_recording()
    print("Quiet breathing complete.\n")

    input("Press Enter to start deep breathing...")
    lbl = f"P{participant}_{mask_label}{leak_tag}_deep_breathing"
    session.start_recording(lbl)
    print("Now breathe deeply for 1 minute (do not hyperventilate).")
    countdown(60, "Deep breathing")
    session.stop_recording()
    print("Deep breathing complete.\n")

    input("Press Enter to start reading the rainbow passage...")
    lbl = f"P{participant}_{mask_label}{leak_tag}_rainbow"
    session.start_recording(lbl)
    print("Please read the following passage aloud:\n")
    print(RAINBOW)
    input("Press Enter when finished reading...")
    session.stop_recording()
    print("Reading complete.\n")


def main():
    participant = input("Participant number: ").strip()
    mask1 = input("Mask 1 label: ").strip() or "mask1"
    mask2 = input("Mask 2 label: ").strip() or "mask2"

    session = PressureParticlesSession()

    try:
        run_mask(session, participant, mask1, with_leak=False)
        print("Apply polyfilla leak to right side of mask 1 and press Enter.")
        input()
        run_mask(session, participant, mask1, with_leak=True)

        print("Switch to mask 2 and press Enter when ready.")
        input()
        run_mask(session, participant, mask2, with_leak=False)
        print("Apply polyfilla leak to right side of mask 2 and press Enter.")
        input()
        run_mask(session, participant, mask2, with_leak=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
