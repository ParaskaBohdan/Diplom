import sys
import shutil

def repeat_csv(input_file, output_file, n):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for _ in range(n):
            with open(input_file, 'r', encoding='utf-8') as in_f:
                shutil.copyfileobj(in_f, out_f)

if __name__ == "__main__":
    # Example usage: python big_csv_generator.py all_together.csv output.csv 5
    if len(sys.argv) != 4:
        print("Usage: python big_csv_generator.py <input_csv> <output_csv> <n>")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    n = int(sys.argv[3])
    repeat_csv(input_csv, output_csv, n)