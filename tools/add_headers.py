import os
import sys

def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split('_')

    participant = parts[1]
    mask_type = parts[2]

    fit_condition = 'no_leak'
    exercise_start_index = 3

    if parts[3] == 'leak':
        fit_condition = 'leak'
        exercise_start_index = 4

    exercise = '_'.join(parts[exercise_start_index:-2])
    date_part = parts[-2]
    time_part = parts[-1]

    return participant, mask_type, fit_condition, exercise, date_part, time_part

def prepend_header_to_csv(file_path, header_lines):
    with open(file_path, 'r', newline='') as f:
        content = f.read()
    new_content = '\n'.join(header_lines) + '\n' + content
    with open(file_path, 'w', newline='') as f:
        f.write(new_content)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 add_headers.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename.startswith('rsc_'):
            participant, mask_type, fit_condition, exercise, date, time = parse_filename(filename)
            file_path = os.path.join(directory, filename)
            header_lines = [
                f'# Participant: {participant}',
                f'# Mask Type: {mask_type}',
                f'# Fit Condition: {fit_condition}',
                f'# Exercise: {exercise}',
                f'# Date: {date}',
                f'# Time: {time}'
            ]
            prepend_header_to_csv(file_path, header_lines)
            print(f'Updated file: {filename}')

if __name__ == '__main__':
    main()
