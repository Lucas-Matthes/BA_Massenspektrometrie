import os
import pandas as pd
import re

def normalize_tic(df, intensity_col):
    tic = df[intensity_col].sum()
    if tic > 0:
        df[intensity_col] = df[intensity_col] / tic
    return df

def sortedbyint(df, count):
    df_sorted = df.sort_values(by=df.columns[1], ascending=False).head(count)
    top_masses = df_sorted[df.columns[0]].tolist()
    top_intensities = df_sorted[df.columns[1]].tolist()
    while len(top_masses) < count:
        top_masses.append(float('nan'))
        top_intensities.append(float('nan'))
    result = []
    for mz, inten in zip(top_masses, top_intensities):
        result.append(mz)
        result.append(inten)
    return result

def sortedbymz(df, count):
    df_sorted = df.sort_values(by=df.columns[1], ascending=False).head(count)
    df_sorted = df_sorted.sort_values(by=df.columns[0], ascending=True)
    top_masses = df_sorted[df.columns[0]].tolist()
    top_intensities = df_sorted[df.columns[1]].tolist()
    while len(top_masses) < count:
        top_masses.append(float('nan'))
        top_intensities.append(float('nan'))
    result = []
    for mz, inten in zip(top_masses, top_intensities):
        result.append(mz)
        result.append(inten)
    return result

def extract_name_and_category(file_name):
    match = re.search(r'\d{4}_(.*?)_(\d)\.xlsx', file_name)
    if match:
        return match.group(1).lower(), int(match.group(2))
    return 'unknown', -1

def process_folder(input_folder, output_csv, count=10, bin_size=1, sheet="1", sort_method="mz"):
    all_data = []
    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".xlsx"):
            continue
        file_path = os.path.join(input_folder, file_name)
        try:
            df = pd.read_excel(file_path, sheet_name=sheet, skiprows=7)
        except Exception as e:
            print(f"Fehler beim Laden von {file_name}: {e}")
            continue
        if df.shape[1] < 2:
            continue
        # Bin
        df['binned_mz'] = ((df[df.columns[0]] / bin_size).round() * bin_size).round(4)
        df_binned = df.groupby('binned_mz', as_index=False)[df.columns[1]].sum()
        df_binned = normalize_tic(df_binned, df.columns[1])
        # Sortieren
        if sort_method == "mz":
            top_values = sortedbymz(df_binned, count)
        else:
            top_values = sortedbyint(df_binned, count)
        # Name und Kategorie anhängen
        name, category = extract_name_and_category(file_name)
        top_values.extend([name, category])
        all_data.append(top_values)
    # DataFrame speichern
    columns = [f"feature_{i+1}" for i in range(len(all_data[0])-2)] + ["molekuelname", "category"]
    df_out = pd.DataFrame(all_data, columns=columns)
    df_out.to_csv(output_csv, index=False)
    print(f"CSV gespeichert unter: {output_csv}")

if __name__ == "__main__":
    process_folder(
        input_folder="data/novel",
        output_csv="data/output/novel.csv",
        count=10,
        bin_size=1,
        sheet="200",
        sort_method="int"
    )
