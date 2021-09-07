import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def features_encoder(df):
    
    df = df.copy()
    
    for col in df.columns: 
        if df[col].dtype == "object":
            
            encoded = LabelEncoder()
            encoded.fit(df[col])
            df[col] = encoded.transform(df[col])
            df[col] = df[col].astype("category")
    
    return df

# TODO
def normalize_df(df): 
    return None

# TODO
def split_data(df): 
    
    df = df.copy()   
    
    nbr_label = pd.unique(df["label"])
    
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for label in nbr_label:
        
        subset = df[df["label"] == label]
        
        if subset.shape[0] > 5:
            
            train_temp,test_temp = train_test_split(subset, test_size=0.4, random_state=0)
            val_temp, test_temp = train_test_split(test_temp, test_size=0.5, random_state=0)
            
            train_df = pd.concat([train_df,train_temp])
            val_df = pd.concat([val_df,val_temp])
            test_df = pd.concat([test_df,test_temp])

    return train_df, val_df, test_df
    