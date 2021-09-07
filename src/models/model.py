from sklearn.ensemble import RandomForestClassifier

def train_model(train_df): 
    
    x = train_df.drop(["label"], axis="columns")
    y = train_df["label"]
    
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf.fit(x, y)
    
    return clf