#!/usr/bin/env python3
"""
Simplified automated testing protocol for mask fit testing.
Handles randomized mask/leak/exercise order with zeroing and verification.
"""

import time
import os
import random
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pressure_particles_ingestion import PressureParticlesSession

# Color constants for enhanced UI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
def colored_text(text, color):
    """Return colored text for terminal output."""
    return f"{color}{text}{Colors.END}"

# Exercise definitions
EXERCISES = {
    "quiet_breathing": {
        "name": "Quiet breathing",
        "instructions": "Breathe normally for 3 minutes",
        "duration": 180  # 3 minutes
    },
    "deep_breathing": {
        "name": "Deep breathing", 
        "instructions": "Breathe deeply (not hyperventilating) for 1 minute",
        "duration": 60
    },
    "rainbow": {
        "name": "Rainbow Passage",
        "instructions": "Read clearly and at a natural pace",
        "duration": None,
        "text": """When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. 
The rainbow is a division of white light into many beautiful colors. 
These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. 
There is, according to legend, a boiling pot of gold at one end. 
People look, but no one ever finds it. 
When a man looks for something beyond reach, his friends say he is looking for the pot of gold at the end of the rainbow."""
    },
    "frc_reset": {
        "name": "FRC Reset",
        "instructions": "Take a gentle breath in, then exhale passively and completely",
        "duration": 30
    }
}

def clear_screen():
    """Clear terminal screen with professional header."""
    if os.name == 'nt':
        os.system('cls')
    else:
        print('\033c', end='', flush=True)
    
    # Display professional header
    print(colored_text("═" * 80, Colors.HEADER))
    print(colored_text("                    MASK FIT TESTING PROTOCOL v2.0                    ", Colors.HEADER + Colors.BOLD))
    print(colored_text("             Automated Breath-by-Breath Analysis System              ", Colors.HEADER))
    print(colored_text("═" * 80, Colors.HEADER))
    print()

def beep():
    """Enhanced notification with visual feedback."""
    if os.name == 'nt':
        import winsound
        winsound.Beep(1000, 200)
    else:
        os.system('tput bel')
    print(colored_text("🔔 Notification!", Colors.YELLOW))

def print_status(message, status_type="info"):
    """Print formatted status messages."""
    symbols = {
        "info": ("ℹ️", Colors.BLUE),
        "success": ("✅", Colors.GREEN),
        "warning": ("⚠️", Colors.YELLOW),
        "error": ("❌", Colors.RED),
        "progress": ("⏳", Colors.CYAN)
    }
    symbol, color = symbols.get(status_type, ("ℹ️", Colors.BLUE))
    print(f"{symbol} {colored_text(message, color)}")

def print_section_header(title, subtitle=None):
    """Print a formatted section header."""
    print()
    print(colored_text("─" * 60, Colors.CYAN))
    print(colored_text(f"📋 {title.upper()}", Colors.CYAN + Colors.BOLD))
    if subtitle:
        print(colored_text(f"   {subtitle}", Colors.CYAN))
    print(colored_text("─" * 60, Colors.CYAN))
    print()

def prompt_user(message, wait_for_enter=True, prompt_type="info"):
    """Enhanced user prompt with visual formatting."""
    print()
    print_status(message, prompt_type)
    if wait_for_enter:
        input(colored_text("👉 Press Enter to continue...", Colors.BOLD))
    print()

def safe_countdown(seconds, message="", exercise_name=""):
    """Enhanced countdown with progress bar and better formatting."""
    print()
    print_section_header(f"{exercise_name} In Progress", message)
    
    total_bars = 40
    for remaining in range(seconds, 0, -1):
        progress = (seconds - remaining) / seconds
        filled_bars = int(total_bars * progress)
        bar = '█' * filled_bars + '░' * (total_bars - filled_bars)
        
        # Time formatting
        mins, secs = divmod(remaining, 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        print(f"\r🕐 [{colored_text(bar, Colors.GREEN)}] {colored_text(time_str, Colors.BOLD)} remaining", end="", flush=True)
        time.sleep(1)
    
    print(f"\r✅ [{colored_text('█' * total_bars, Colors.GREEN)}] {colored_text('COMPLETE!', Colors.GREEN + Colors.BOLD)}")
    print()

# Removed reset_session_state - not needed, PressureParticlesSession handles this naturally

def display_exercise_instructions(exercise_name, exercise_data, mask_label, with_leak):
    """Display formatted exercise instructions."""
    clear_screen()
    
    # Condition display
    leak_status = colored_text("LEAK CONDITION", Colors.RED + Colors.BOLD) if with_leak else colored_text("NO LEAK", Colors.GREEN + Colors.BOLD)
    mask_display = colored_text(mask_label, Colors.BLUE + Colors.BOLD)
    
    print_section_header(exercise_data['name'], f"Mask: {mask_display} | Status: {leak_status}")
    
    # Instructions box
    print(colored_text("📝 INSTRUCTIONS:", Colors.YELLOW + Colors.BOLD))
    print(colored_text("┌" + "─" * 58 + "┐", Colors.YELLOW))
    
    instructions = exercise_data['instructions']
    for line in instructions.split('\n'):
        print(colored_text(f"│ {line:<56} │", Colors.YELLOW))
    
    print(colored_text("└" + "─" * 58 + "┘", Colors.YELLOW))
    
    # Special handling for Rainbow Passage
    if exercise_name == "rainbow":
        print()
        print(colored_text("📖 RAINBOW PASSAGE TEXT:", Colors.CYAN + Colors.BOLD))
        print(colored_text("┌" + "─" * 78 + "┐", Colors.CYAN))
        
        text_lines = exercise_data['text'].split('. ')
        for line in text_lines:
            if line.strip():
                wrapped_line = line.strip() + ('.' if not line.endswith('.') else '')
                print(colored_text(f"│ {wrapped_line:<76} │", Colors.CYAN))
        
        print(colored_text("└" + "─" * 78 + "┘", Colors.CYAN))
    
    print()
    print_status(f"Ready to begin {exercise_data['name']}?", "progress")
    input(colored_text("👉 Press Enter to start recording...", Colors.GREEN + Colors.BOLD))

def display_condition_overview(mask_label, with_leak, exercise_count=4):
    """Display an overview of the current test condition."""
    clear_screen()
    
    leak_status = "LEAK CONDITION" if with_leak else "NO LEAK CONDITION"
    status_color = Colors.RED if with_leak else Colors.GREEN
    
    print()
    print(colored_text("🎯 " + "CURRENT TEST CONDITION".center(70) + " 🎯", Colors.HEADER + Colors.BOLD))
    print()
    print(colored_text("┌" + "─" * 70 + "┐", Colors.BLUE))
    print(colored_text(f"│ MASK: {mask_label.center(60)} │", Colors.BLUE + Colors.BOLD))
    print(colored_text(f"│ STATUS: {colored_text(leak_status.center(56), status_color + Colors.BOLD)} │", Colors.BLUE))
    print(colored_text(f"│ EXERCISES TO COMPLETE: {str(exercise_count).center(48)} │", Colors.BLUE))
    print(colored_text("└" + "─" * 70 + "┘", Colors.BLUE))
    print()
    
    prompt_user("All equipment ready for this condition?", True, "progress")

def record_data(session, participant, mask_label, with_leak, exercise_name, duration=None):
    """Record data for a specific exercise with enhanced feedback."""
    fit_condition = "leak" if with_leak else "no_leak"
    
    print_status("Starting data recording...", "progress")
    
    session.start_recording(
        participant=f"P{participant}",
        mask_type=mask_label,
        fit_condition=fit_condition,
        exercise=exercise_name
    )
    session.wait_for_serial_message("# streaming ON", timeout=5)
    
    print_status("📊 Recording active - sensors are collecting data", "success")
    
    if duration:
        safe_countdown(duration, f"Recording {exercise_name}", EXERCISES.get(exercise_name, {}).get('name', exercise_name))
    else:
        print()
        print_status("Complete the exercise at your own pace", "info")
        input(colored_text("👉 Press Enter when finished...", Colors.GREEN + Colors.BOLD))
    
    # 10s extension for particle data
    print_status("⏱️  Capturing additional particle data (10 seconds)...", "progress")
    time.sleep(10)
    
    session.stop_recording()
    session.wait_for_serial_message("# streaming OFF", timeout=5)
    
    print_status("✅ Data recording complete!", "success")
    time.sleep(1)

def verify_zero(session, participant, mask_label, with_leak):
    """Verify zero reading with enhanced visual feedback - using in-memory buffer like original."""
    fit_condition = "leak" if with_leak else "no_leak"

    print_section_header("Zero Verification", "Hold your breath to verify sensor calibration")
    prompt_user("Take a deep breath and hold it steady", True, "warning")
    
    print_status("🫁 Recording verification data...", "progress")
    
    # Use the old label format like automated_testing.py
    leak_tag = "_leak" if with_leak else ""
    lbl = f"P{participant}_{mask_label}{leak_tag}_zero_check"

    session.start_recording(lbl, participant=f"P{participant}", mask_type=mask_label, fit_condition=fit_condition, exercise="zero_check")
    session.wait_for_serial_message("# streaming ON", timeout=5)
    
    print_status("📊 Data stream active - collecting 5 seconds of data...", "success")
    
    # Simple countdown like the original
    for remaining in range(5, 0, -1):
        print(f"Verifying zero {remaining}s remaining", end="\r")
        time.sleep(1)
    print(" " * 40, end="\r")
    
    session.stop_recording()
    session.wait_for_serial_message("# streaming OFF", timeout=5)
    
    print_status("✅ Data collection complete", "success")
    
    # Check in-memory buffer like the original
    if not session.p_g:
        print_status("No data captured for verification.", "warning")
        return False
    
    # Use in-memory data like automated_testing.py
    t = np.array(session.ts_buf) - session.ts_buf[0]
    p = np.array(session.p_g)
    mean_p = float(p.mean())
    std_p = float(p.std())
    
    print_status(f"Analyzed {len(p)} pressure readings from buffer", "info")
    
    # Enhanced plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, p, color='blue', linewidth=1.5, alpha=0.8)
    plt.axhline(mean_p, color='green', linestyle='--', linewidth=2, label=f"Mean: {mean_p:.3f} Pa")
    plt.axhline(mean_p + std_p, color='red', linestyle=':', alpha=0.7, label=f"±1 Std: {std_p:.3f} Pa")
    plt.axhline(mean_p - std_p, color='red', linestyle=':', alpha=0.7)
    plt.fill_between(t, mean_p - std_p, mean_p + std_p, alpha=0.2, color='red')
    plt.title(f"Zero Verification: {mask_label} | {fit_condition.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show plot non-blocking
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    
    # Display results
    print()
    print(colored_text("📊 VERIFICATION RESULTS:", Colors.BLUE + Colors.BOLD))
    print(colored_text("┌" + "─" * 50 + "┐", Colors.BLUE))
    print(colored_text(f"│ Mean Pressure:     {mean_p:8.4f} Pa          │", Colors.BLUE))
    print(colored_text(f"│ Standard Deviation: {std_p:8.4f} Pa          │", Colors.BLUE))
    print(colored_text("│" + "─" * 48 + "│", Colors.BLUE))
    
    # Use original criteria (0.2 Pa mean, 0.5 Pa std)
    pass_criteria = abs(mean_p) <= 0.2 and std_p <= 0.5
    status = "PASS" if pass_criteria else "FAIL"
    status_color = Colors.GREEN if pass_criteria else Colors.RED
    
    print(colored_text(f"│ Result: {colored_text(status.center(36), status_color + Colors.BOLD)} │", Colors.BLUE))
    print(colored_text("└" + "─" * 50 + "┘", Colors.BLUE))
    
    print(f"Zero check mean: {mean_p:.3f} Pa | stdev: {std_p:.3f} Pa")
    
    if pass_criteria:
        print_status("Zero verification PASS! You can breathe again!", "success")
        beep()
        time.sleep(2)
        return True
    else:
        print_status("Zero verification FAIL - re-zeroing needed", "warning")
        print_status("Criteria: |Mean| ≤ 0.2 Pa AND Std Dev ≤ 0.5 Pa", "info")
        return False

def perform_zeroing_cycle(session, participant, mask_label, with_leak):
    """Perform zeroing and verification cycle with enhanced UI."""
    fit_condition = "leak" if with_leak else "no_leak"
    attempt = 1
    
    while True:
        clear_screen()
        
        leak_status = colored_text("LEAK CONDITION", Colors.RED + Colors.BOLD) if with_leak else colored_text("NO LEAK", Colors.GREEN + Colors.BOLD)
        
        print_section_header("Sensor Zeroing & Verification", 
                           f"Mask: {colored_text(mask_label, Colors.BLUE + Colors.BOLD)} | Status: {leak_status}")
        
        if attempt > 1:
            print_status(f"Attempt #{attempt} - Previous verification failed", "warning")
        
        print_status("🔧 Preparing to zero sensors...", "progress")
        
        session.start_recording(
            participant=f"P{participant}",
            mask_type=mask_label,
            fit_condition=fit_condition,
            exercise="zeroing"
        )
        
        print_status("⏳ Waiting for data stream to start...", "progress")
        session.wait_for_serial_message("# streaming ON", timeout=5)
        print_status("✅ Data stream confirmed active", "success")

        # Enhanced zeroing instructions
        print()
        print(colored_text("🫁 ZEROING INSTRUCTIONS:", Colors.YELLOW + Colors.BOLD))
        print(colored_text("┌" + "─" * 58 + "┐", Colors.YELLOW))
        print(colored_text("│ 1. Take a deep breath and hold it steady             │", Colors.YELLOW))
        print(colored_text("│ 2. Keep your breath held during the entire process   │", Colors.YELLOW))
        print(colored_text("│ 3. Do not move or speak                              │", Colors.YELLOW))
        print(colored_text("└" + "─" * 58 + "┘", Colors.YELLOW))
        
        prompt_user("Ready to hold your breath for sensor zeroing?", True, "warning")

        print_status("⏳ Stopping data recording...", "progress")
        session.stop_recording()
        session.wait_for_serial_message("# streaming OFF", timeout=5)
        print_status("✅ Data recording stopped", "success")
        
        print_status("⚙️ Zeroing sensors... keep holding your breath!", "progress")
        session.send_teensy("zero")
        session.wait_for_serial_message("# new zeros:", timeout=5)
        print_status("✅ Sensors have been zeroed!", "success")
        time.sleep(2)
        
        # Verify the zero
        if verify_zero(session, participant, mask_label, with_leak):
            print_status("🎉 Zeroing cycle complete! Ready to proceed with exercises.", "success")
            time.sleep(2)
            return  # Success, exit loop
        
        attempt += 1
        print()
        print_status("Need to repeat zeroing process", "warning")
        input(colored_text("👉 Press Enter to try zeroing again...", Colors.YELLOW + Colors.BOLD))

def run_exercise(session, participant, mask_label, with_leak, exercise_name):
    """Run a single exercise with enhanced UI."""
    exercise = EXERCISES[exercise_name]
    
    # Display exercise instructions
    display_exercise_instructions(exercise_name, exercise, mask_label, with_leak)
    
    if exercise_name == "rainbow":
        record_data(session, participant, mask_label, with_leak, exercise_name)
    elif exercise_name == "frc_reset":
        # Special handling for FRC reset with clear instructions
        print_section_header("FRC Reset - Part 1", "Initial breath and reset")
        print_status("Take a gentle breath in, then exhale passively and completely", "info")
        record_data(session, participant, mask_label, with_leak, "frc_reset_init", 5)
        
        print_section_header("FRC Reset - Part 2", "Resume normal breathing")
        print_status("Resume quiet, normal breathing", "info")
        record_data(session, participant, mask_label, with_leak, "frc_reset_quiet", 30)
    else:
        record_data(session, participant, mask_label, with_leak, exercise_name, exercise['duration'])
    
    # Exercise completion
    print()
    print_status(f"🎯 {exercise['name']} completed successfully!", "success")
    time.sleep(1)

def handle_mask_transition(current_mask, current_leak, prev_mask, prev_leak):
    """Handle mask/leak transitions with enhanced visual guidance."""
    if prev_mask is None:
        # First condition
        action = f"Apply polyfil leak to {current_mask}" if current_leak else f"Ensure no leak on {current_mask}"
        icon = "🔴" if current_leak else "🟢"
    elif current_mask != prev_mask:
        leak_action = "apply polyfil leak" if current_leak else "ensure no leak"
        action = f"Switch to {current_mask} and {leak_action}"
        icon = "🔄"
    elif current_leak != prev_leak:
        action = f"{'Apply' if current_leak else 'Remove'} polyfil leak on {current_mask}"
        icon = "🔴" if current_leak else "🟢"
    else:
        return  # No change needed
    
    clear_screen()
    print_section_header("Equipment Setup", "Mask and leak configuration")
    
    print(colored_text("🎭 SETUP INSTRUCTIONS:", Colors.BLUE + Colors.BOLD))
    print(colored_text("┌" + "─" * 58 + "┐", Colors.BLUE))
    print(colored_text(f"│ {icon} {action:<52} │", Colors.BLUE))
    print(colored_text("│                                                    │", Colors.BLUE))
    print(colored_text("│ Notes:                                             │", Colors.BLUE))
    print(colored_text("│ • Apply leak on the RIGHT side of the mask        │", Colors.BLUE))
    print(colored_text("│ • Ensure proper mask fit and comfort              │", Colors.BLUE))
    print(colored_text("│ • Check all connections are secure                │", Colors.BLUE))
    print(colored_text("└" + "─" * 58 + "┘", Colors.BLUE))
    
    prompt_user("Equipment setup complete and ready to proceed?", True, "progress")

def main():
    """Main protocol execution with enhanced UI."""
    clear_screen()
    
    # Welcome and setup
    print_section_header("Protocol Setup", "Initialize testing parameters")
    
    print(colored_text("👥 PARTICIPANT INFORMATION:", Colors.GREEN + Colors.BOLD))
    participant = input(colored_text("Enter participant number: ", Colors.GREEN)).strip()
    
    print()
    print(colored_text("🎭 MASK CONFIGURATION:", Colors.BLUE + Colors.BOLD))
    mask1 = input(colored_text("Enter Mask 1 label: ", Colors.BLUE)).strip() or "mask1"
    mask2 = input(colored_text("Enter Mask 2 label: ", Colors.BLUE)).strip() or "mask2"
    
    # System initialization
    print()
    print_status("🔧 Initializing data collection system...", "progress")
    session = PressureParticlesSession()
    
    print_status("📡 Connecting to Teensy controller...", "progress")
    session.send_teensy("stop")
    session.wait_for_serial_message("OFF", timeout=5)
    session.send_teensy("rate 1000")
    session.wait_for_serial_message("1000", timeout=5)
    time.sleep(1)
    print_status("✅ System initialization complete!", "success")
    
    # Setup verification
    clear_screen()
    print_section_header("Equipment Verification", "Pre-test system check")
    
    print(colored_text("✅ VERIFICATION CHECKLIST:", Colors.YELLOW + Colors.BOLD))
    print(colored_text("┌" + "─" * 58 + "┐", Colors.YELLOW))
    print(colored_text("│ □ WRPAS wick is properly saturated                   │", Colors.YELLOW))
    print(colored_text("│ □ All tubes are connected correctly                   │", Colors.YELLOW))
    print(colored_text("│ □ Pressure sensors are responsive                     │", Colors.YELLOW))
    print(colored_text("│ □ Particle counter is operational                     │", Colors.YELLOW))
    print(colored_text("│ □ Both masks are available and ready                  │", Colors.YELLOW))
    print(colored_text("└" + "─" * 58 + "┘", Colors.YELLOW))
    
    prompt_user("All equipment verified and ready to begin?", True, "progress")
    
    # Generate test order
    print_status("🎲 Generating randomized test sequence...", "progress")
    mask_groups = [[(mask1, False), (mask1, True)], [(mask2, False), (mask2, True)]]
    random.shuffle(mask_groups)  # Randomize mask order
    
    exercises = list(EXERCISES.keys())
    
    # Display test plan
    clear_screen()
    print_section_header("Test Protocol Overview", "Randomized sequence generated")
    
    print(colored_text("📋 TEST SEQUENCE:", Colors.CYAN + Colors.BOLD))
    print(colored_text("┌" + "─" * 70 + "┐", Colors.CYAN))
    
    condition_num = 1
    for mask_group in mask_groups:
        random.shuffle(mask_group)  # Randomize leak order within mask
        for mask_label, with_leak in mask_group:
            leak_text = "LEAK" if with_leak else "NO LEAK"
            print(colored_text(f"│ Condition {condition_num}: {mask_label:<15} | {leak_text:<15} │", Colors.CYAN))
            condition_num += 1
    
    print(colored_text("│" + "─" * 68 + "│", Colors.CYAN))
    print(colored_text(f"│ Exercises per condition: {len(exercises):<5} (randomized order)      │", Colors.CYAN))
    print(colored_text(f"│ Total conditions: {len(mask_groups) * 2:<5}                               │", Colors.CYAN))
    print(colored_text("└" + "─" * 70 + "┘", Colors.CYAN))
    
    prompt_user("Ready to begin the automated testing protocol?", True, "success")
    
    # Execute test protocol
    prev_mask, prev_leak = None, None
    condition_count = 1
    total_conditions = len(mask_groups) * 2
    
    for mask_group in mask_groups:
        for mask_label, with_leak in mask_group:
            # Progress indicator
            clear_screen()
            progress_pct = (condition_count - 1) / total_conditions * 100
            print(colored_text(f"🚀 PROTOCOL PROGRESS: {progress_pct:.0f}% Complete", Colors.HEADER + Colors.BOLD))
            print(colored_text(f"    Condition {condition_count} of {total_conditions}", Colors.HEADER))
            print()
            
            # Handle transitions
            handle_mask_transition(mask_label, with_leak, prev_mask, prev_leak)
            prev_mask, prev_leak = mask_label, with_leak
            
            # Show current condition
            display_condition_overview(mask_label, with_leak, len(exercises))
            
            # Zeroing cycle
            perform_zeroing_cycle(session, participant, mask_label, with_leak)
            
            # Run exercises in random order
            exercise_order = random.sample(exercises, len(exercises))
            for i, exercise_name in enumerate(exercise_order, 1):
                print()
                print_status(f"Exercise {i} of {len(exercises)} for this condition", "info")
                run_exercise(session, participant, mask_label, with_leak, exercise_name)
            
            # Condition completion
            print()
            print_status(f"🎉 Condition {condition_count} completed successfully!", "success")
            
            condition_count += 1
            
            if condition_count <= total_conditions:
                time.sleep(2)
    
    # Protocol completion
    clear_screen()
    print()
    print(colored_text("🎊 " + "PROTOCOL COMPLETE!".center(60) + " 🎊", Colors.GREEN + Colors.BOLD))
    print()
    print(colored_text("┌" + "─" * 60 + "┐", Colors.GREEN))
    print(colored_text(f"│ Participant: {participant:<47} │", Colors.GREEN))
    print(colored_text(f"│ Conditions completed: {total_conditions:<39} │", Colors.GREEN))
    print(colored_text(f"│ Total exercises: {total_conditions * len(exercises):<43} │", Colors.GREEN))
    print(colored_text(f"│ Data files saved in: data/P{participant}/<files>        │", Colors.GREEN))
    print(colored_text("└" + "─" * 60 + "┘", Colors.GREEN))
    print()
    print_status("All data has been collected and saved successfully!", "success")
    print_status("Thank you for completing the mask fit testing protocol!", "success")
    
    session.close()

if __name__ == "__main__":
    main()
