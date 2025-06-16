import os
import glob
import pandas as pd

# Поточна директорія (можеш змінити, якщо потрібно)
current_dir = os.path.dirname(__file__)

# Знаходимо всі CSV-файли
csv_files = glob.glob(os.path.join(current_dir, "*.csv"))

dfs = []

for file in csv_files:
    try:
        df = pd.read_csv(file, sep=';', quotechar='"', encoding='utf-8')
        
        # Ігноруємо порожні файли або файли без даних
        if not df.empty and len(df.columns) > 1:
            dfs.append(df)
        else:
            print(f"⚠️  Пропущено порожній або пошкоджений файл: {file}")
    except Exception as e:
        print(f"❌ Помилка при зчитуванні {file}: {e}")

# Об'єднуємо всі дані
if dfs:
    big_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(current_dir, "all_together.csv")
    big_df.to_csv(output_path, index=False, sep=';')
    print(f"✅ Об'єднано {len(dfs)} файлів у all_together.csv")
else:
    print("❌ Жодного валідного CSV-файлу не знайдено.")
