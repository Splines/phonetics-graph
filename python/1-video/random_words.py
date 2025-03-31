import csv
import random


INPUT_FILE = "data/graph/nodes-first-100000.csv"
OUTPUT_FILE = "data/graph/nodes-first-random.csv"


def main():
    rows = []

    # Read the CSV file
    with open(INPUT_FILE, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        rows = list(reader)

    # Randomly choose words
    rows = rows[:5000]
    rows = [row[1] for row in rows]
    random_words = random.sample(rows, 1000)

    # Output
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for word in random_words:
            writer.writerow([word])


if __name__ == "__main__":
    main()
