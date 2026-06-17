
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """
    Melakukan preprocessing pada dataset universitas Jepang.
    Parameter:
        df (pd.DataFrame): DataFrame mentah dari kagglehub
    Returns:
        pd.DataFrame: DataFrame yang sudah siap untuk pemodelan
    """
    # Copy agar tidak mengubah data asli
    df = df.copy()
    
    # 1. Handle Missing Values
    df['phone'] = df['phone'].fillna('Unknown')
    df['review_rating'] = df['review_rating'].fillna(df['review_rating'].median())
    df['review_count'] = df['review_count'].fillna(df['review_count'].median())
    df['difficulty_SD'] = df['difficulty_SD'].fillna(df['difficulty_SD'].median())
    mode_rank = df['difficulty_rank'].mode()[0]
    df['difficulty_rank'] = df['difficulty_rank'].fillna(mode_rank)
    
    # 2. Ekstraksi tahun dari kolom 'found'
    df['found_year'] = df['found'].str.extract(r'(\d{4})').astype(float)
    
    # 3. Encoding variabel kategorikal
    rank_order = ['S', 'A', 'B', 'C', 'D', 'E', 'F']
    rank_mapping = {rank: i for i, rank in enumerate(rank_order)}
    df['difficulty_rank_encoded'] = df['difficulty_rank'].map(rank_mapping)
    
    # One-hot encoding untuk type dan state
    df = pd.get_dummies(df, columns=['type', 'state'], prefix=['type', 'state'])
    
    # Boolean ke integer
    df['has_grad'] = df['has_grad'].astype(int)
    df['has_remote'] = df['has_remote'].astype(int)
    
    # 4. Standarisasi fitur numerik
    numerical_cols = ['faculty_count', 'department_count', 'review_rating', 
                      'review_count', 'difficulty_SD', 'latitude', 'longitude', 
                      'found_year']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # 5. Drop kolom yang tidak diperlukan
    cols_to_drop = ['Unnamed: 0', 'code', 'name', 'name_jp', 'type_jp', 
                    'address', 'postal_code', 'phone', 'state_jp', 'found', 
                    'difficulty_rank']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df

# Contoh penggunaan jika dijalankan sebagai script
if __name__ == "__main__":
    # Simulasi loading data (sesuaikan dengan cara Anda mengambil data)
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    
    file_path = "japanese_universities.csv"
    df_raw = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "webdevbadger/japanese-universities",
        file_path
    )
    df_clean = preprocess_data(df_raw)
    print("Preprocessing selesai. Shape data:", df_clean.shape)
    # Simpan ke CSV untuk keperluan workflow
    df_clean.to_csv("data_clean.csv", index=False)
    print("Data bersih disimpan ke data_clean.csv")
