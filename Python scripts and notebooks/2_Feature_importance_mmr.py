import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import shap

# upload file and prepare features to normalize
def upload_file(file_path='mmr.xlsx', sheet_name='car_sale'):
    file=pd.ExcelFile(file_path)
    print(file.sheet_names)
    df=pd.read_excel(file_path, sheet_name=sheet_name) # load dataset
    return df

# Replace placeholder odometer values (999999) in dataset with predicted values
def replace_placeholder(df, features=['age', 'condition'], target='odometer', placeholder=999999): 
    wrong_odo_mask=df[target]==placeholder
    df_bad_odo=df[wrong_odo_mask]
    df_good_odo=df[~wrong_odo_mask]
    x_train=df_good_odo[features]
    y_train=df_good_odo[target]
    x_predict=df_bad_odo[features]
    
    scaler=StandardScaler()
    x_train_s=scaler.fit_transform(x_train) # normalize input features
    x_predict_s=scaler.transform(x_predict)
    model=LinearRegression() 
    model.fit(x_train_s, y_train) # train using linear regression
    predicted_odometers=model.predict(x_predict_s) # predict placeholder odometer values
    predicted_odometers=np.round(predicted_odometers).astype(int) 
    
    print(predicted_odometers[:10]) # check first 10 predicted values
    df.loc[wrong_odo_mask, target]=predicted_odometers #replace placeholder values with predicted values
    print(df.columns.duplicated().sum()) # check if duplicate columns are not created in df
    print(df[target].max()) # check max value of odometer (target column)
    return df

# Normalize selected features
def normalize_features(df, features_to_normalize=['condition', 'odometer', 'age']):
    x=df[features_to_normalize].copy()
    scaler=MinMaxScaler()
    x_norm=scaler.fit_transform(x)
    x_norm_df=pd.DataFrame(x_norm, columns=[f"{col}_norm" for col in features_to_normalize])
    print(x_norm_df.head())
    
    df = df.drop(columns=['condition_norm', 'age_norm', 'odometer_norm'], errors='ignore') # drop these columns to avoid duplication on multiple runs
    df=pd.concat([df, x_norm_df], axis=1) # add normalized values of features to df
    df['quality']=df['condition_norm']+df['age_norm'] # combine age and condition features into a single feature: 'quality'
    print(df['quality'].head())
    return df
    
# Use Random Forest Regressor to find out feature importance for generating 'mmr' or 'sellingprice'
def feature_importance(df, target='mmr'):
    df['model_encoded']=df['model'].map(df['model'].value_counts()) # use frequency encoding for models
    df=pd.get_dummies(df, columns=['brand', 'body', 'color'], drop_first=True) # use one hot encoding for other features
    features=['quality', 'odometer_norm', 'model_encoded'] + [col for col in df.columns if col.startswith('brand_') or col.startswith('body_') or col.startswith('color_')]
    x=df[features]
    y=df[target]
    model=RandomForestRegressor(n_estimators=100, random_state=42) 
    model.fit(x,y)
    importances=pd.Series(model.feature_importances_, index=x.columns)
    importances.sort_values(ascending=False).head(20).plot(kind='barh', figsize=(10,8), title='Top 20 features importance for MMR')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    importances_df = pd.DataFrame({'feature': x.columns, 'importance': model.feature_importances_})
    importances_df['group'] = importances_df['feature'].str.extract(r'(^[^_]+)')  # gets 'brand', 'body', etc.
    group_importance = importances_df.groupby('group')['importance'].sum().sort_values(ascending=True)
    group_importance.plot(kind='barh', figsize=(8, 6), title='Grouped Feature Importance') # plot feature importance by group
    plt.tight_layout()
    plt.show()
    return importances_df