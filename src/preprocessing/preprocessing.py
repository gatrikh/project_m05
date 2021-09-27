import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def features_encoder(df):
    
    for col in df.columns: 
        if df[col].dtype == "object":
            
            encoded = LabelEncoder()
            encoded.fit(df[col])
            df[col] = encoded.transform(df[col])
            df[col] = df[col].astype("category")
    
    return df

def normalize(df): 

    for col in df.columns:
        scaler = StandardScaler()
        if str(df[col].dtypes) != "category":
            df[col] = scaler.fit_transform(df[col].to_numpy().reshape(-1, 1))
   
    return df

def split_data(df): 
      
    nbr_label = pd.unique(df["label"])
    nbr_label = nbr_label.sort_values()
    
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for label in nbr_label:
        
        subset = df[df["label"] == label]
        
        if subset.shape[0] >= 4:
            
            train_temp, test_temp = train_test_split(subset, test_size=0.4, random_state=0)
            val_temp, test_temp = train_test_split(test_temp, test_size=0.5, random_state=0)
            
            train_df = pd.concat([train_df, train_temp])
            val_df = pd.concat([val_df, val_temp])
            test_df = pd.concat([test_df, test_temp])
        
        elif subset.shape[0] == 3: 

            train_df = train_df.append(subset.iloc[0])
            val_df = val_df.append(subset.iloc[1])
            test_df = test_df.append(subset.iloc[2])

        elif subset.shape[0] == 2:
            train_df = train_df.append(subset.iloc[0])
            test_df = test_df.append(subset.iloc[1])
            
        elif subset.shape[0] == 1: 
            train_df = train_df.append(subset.iloc[0])
        
        df.drop(subset.index, inplace=True)

    return train_df, val_df, test_df
    