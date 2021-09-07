import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

def features_encoder(df):
    
    df = df.copy()
    
    for col in df.columns: 
        if df[col].dtype == "object":
            
            encoded = LabelEncoder()
            encoded.fit(df[col])
            df[col] = encoded.transform(df[col])
            df[col] = df[col].astype("category")
    
    return df

def pca_reduction(df):
    
    pca = decomposition.PCA(n_components=0.99)

    df_norm_pca = pca.fit_transform(df_norm)
    
    return df_norm_pca

def normalize(df): 
    
    df = df.copy()
    for col in df.columns:
        scaler = StandardScaler()
        if str(df[col].dtypes) != "category":
            df[col] = scaler.fit_transform(df[col].to_numpy().reshape(-1, 1))
   
    return df
    
    


def split_data(df): 
    
    df = df.copy()
    
    normal_df = df[df["label"] == 1]
    abnormal_df = df[df["label"] == -1]
    
    train_normal, test_normal = train_test_split(normal_df, test_size=0.4, random_state=0)
    val_normal, test_normal = train_test_split(test_normal, test_size=0.5, random_state=0)
    
    train_abnormal, test_abnormal = train_test_split(abnormal_df, test_size=0.4, random_state=0)
    val_abnormal, test_abnormal = train_test_split(test_abnormal, test_size=0.5, random_state=0)
    
    train_df = pd.concat([train_normal, train_abnormal]).reset_index(drop=True)
    val_df = pd.concat([val_normal, val_abnormal]).reset_index(drop=True)
    test_df = pd.concat([test_normal, test_abnormal]).reset_index(drop=True)
    
    return train_df, val_df, test_df
    