from itertools import combinations
import logging
import pandas as pd
import numpy as np
import os
import sklearn
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_csv(dir):
    dataframes = {}

    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir, filename)
            
            logging.info(f"Membaca file: {filename} ...")

            try:
                df = pd.read_csv(file_path)
                dataframes[filename] = df
                logging.info(f"Berhasil membaca {filename} ({len(df)} baris)")
            except Exception as e:
                logging.error(f"Gagal membaca {filename}: {e}")

    return dataframes

def count_column_every_feature(dir):
    logging.info(f"Mulai membaca folder: {dir}")

    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir, filename)

            logging.info(f"Name file: {filename}")

            try:
                df = pd.read_csv(file_path)
                logging.info(f"Berhasil membaca {filename} ({len(df)} baris, {len(df.columns)} kolom)")

                column_info = {}
                
                for col in df.columns:
                    non_null = df[col].count()
                    unique = df[col].nunique()
                    missing = df[col].isnull().sum()

                    column_info[col] = {
                        "non_null_count": non_null,
                        "unique_count": unique,
                        "missing_count": missing
                    }

                    logging.info(
                        f"[{filename}] "
                        f"{col:<20} | "
                        f"NonNull: {non_null:<6} | "
                        f"Unique: {unique:<6} | "
                        f"Missing: {missing:<6}"
                    )
            except Exception as e:
                logging.error(f"Gagal membaca {filename}: {e}")

    logging.info("Selesai memproses semua file.")

def handle_missing_data(dir):
    logging.info(f"Mulai membaca folder: {dir}")
    
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir, filename)
            logging.info(f"Memproses file: {filename}")
            
            try:
                df = pd.read_csv(file_path)
                df = df.dropna()
                df = df.drop_duplicates()
                df.to_csv(file_path, index=False)
                logging.info(f"Selesai bersihkan: {filename}")

            except Exception as e:
                logging.error(f"Error pada file {filename}: {e}")


def calculate_linear_regression(dir):
    logging.info(f"Mulai membaca folder: {dir}")
    
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir, filename)
            logging.info(f"Memproses file: {filename}")
            
            try:
                df = pd.read_csv(file_path)
                df_numeric = df.select_dtypes(include=['number'])

                if df_numeric.shape[1] < 2:
                    logging.info(f"Tidak cukup kolom numerik di {filename}")
                    continue

                strong_relation_found = False
                for col_x, col_y in combinations(df_numeric.columns, 2):
                    pair_df = df_numeric[[col_x, col_y]].dropna()

                    # skip kalau data kosong setelah dropna
                    if pair_df.shape[0] < 2:
                        logging.info(f"{filename} | {col_x}-{col_y} skip (data kosong)")
                        continue

                    if pair_df[col_x].nunique() <= 1 or pair_df[col_y].nunique() <= 1:
                        logging.info(f"{filename} | {col_x}-{col_y} skip (konstan)")
                        continue

                    X = pair_df[[col_x]].values
                    y = pair_df[col_y].values

                    model = LinearRegression()
                    model.fit(X, y)  # type: ignore
                    corr = np.corrcoef(pair_df[col_x], pair_df[col_y])[0, 1]

                    logging.info(
                        f"[{filename}] {col_x:>15} ↔ {col_y:<15} | corr = {corr:>7.4f}"
                    )

                    if abs(corr) >= 0.8:
                        strong_relation_found = True
                        logging.info(
                            f"🚩 Strong correlation: {filename} | {col_x} ↔ {col_y} = {corr:.4f}"
                        )
                print(f"\n🔎 HASIL ANALISIS:")
                if strong_relation_found:
                    print(f"Dataset '{filename}' MEMILIKI hubungan linear kuat (|corr| ≥ 0.8) dengan korelasi f{corr}")
                else:
                    print(f"Dataset '{filename}' tidak memiliki hubungan linear kuat dengan korelasi f{corr}")
                print(f"{'='*60}\n")

            except Exception as e:
                logging.error(f"Error pada file {filename}: {e}")

# def shrink_column(dir):
#     logging.info(f"Mulai membaca folder: {dir}")
    
#     for filename in os.listdir(dir):
#         if filename.endswith(".csv"):
#             file_path = os.path.join(dir, filename)

#             logging.info(f"Memproses file: {filename}")

#             try:
#                 df = pd.read_csv(file_path)
#                 original_cols = len(df.columns)
#                 df_shrink = df.iloc[:, :5]
#                 df_shrink.to_csv(file_path, index=False)

#                 logging.info(
#                     f"{filename:<20} | Kolom sebelum: {original_cols:<3} | Kolom sesudah: {len(df_shrink.columns):<3} | Status: overwritten"
#                 )

#             except Exception as e:
#                 logging.error(f"Gagal memproses {filename}: {e}")


def check_correlation(dir,to_dir):
    logging.info(f"Mulai cek korelasi dari: {dir}")
    logging.info(f"Hasil akan disimpan ke: {to_dir}")
    
    for filename in os.listdir(dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(dir, filename)
            output_path = os.path.join(to_dir, f"corr_{filename}")

            logging.info(f"Proses file: {filename}")

            if os.path.exists(output_path):
                logging.info(f"File sudah ada, skip: {output_path}")
                continue

            try:
                df = pd.read_csv(input_path)
                df_numeric = df.select_dtypes(include=['number'])

                if df_numeric.shape[1] < 2:
                    logging.warning(f"Kolom numerik kurang dari 2 di {filename}, skip")
                    continue

                corr_matrix = df_numeric.corr()

                print(f"\nKorelasi - {filename}")
                print(corr_matrix)

                corr_matrix.to_csv(output_path)
                logging.info(f"Disimpan ke: {output_path}")

            except Exception as e:
                logging.error(f"Error di file {filename}: {e}")


if __name__=='__main__':
    # print(read_csv("./Datasets"))
    # count_column_every_feature("./Datasets")
    # # shrink_column("./Dataset")
    # handle_missing_data("./Datasets")
    check_correlation("./Datasets","./Correlations")
    calculate_linear_regression("./Datasets")