**Project Title**: Vehicle MMR Prediction

**Project Overview**: An open dataset for auctioned cars in U.S. was obtained from Kaggle. This dataset initially contained over 5.8 lakh+ samples. After cleaning, which involved removing entries with missing information 
or zero values for either the Manheim Market Report (MMR) or selling price, and eliminating odd samples (e.g. age of the car is one year but odometer rating exceeds 900,000 kms), the dataset was refined to 
approximately 5.3 lakh samples. The dataset provides comprehensive details for each sample, including:
Vehicle Specifications: Year of purchase, brand, model, body type, transmission type, and color.
Condition & Usage: Condition rating and odometer reading.
Location & Transaction Details: State of auction, selling price, and date of sale/auction.
Manheim Market Report (MMR): A crucial metric widely used in the U.S. for estimating a vehicle's wholesale price.

**Problem Statement**: Analyzing the dataset revealed that a significant portion of vehicles (2.71 lakh out of 5.3 lakh) were sold at auction for price lower than MMR, resulting in loss for the dealer.

**Project Objectives**: 
1.Determine key factors (e.g. vehicle specifications, condition, usage etc.) which influence vehicle’s wholesale (MMR) value
2.Develop a predictive model to forecast more accurate MMR value, thereby minimizing probability of loss on the sale of vehicle

**Data Source**: https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data 

**Exploratory Data Analysis**: To reduce dataset complexity, following feature engineering steps were taken: 
1. Car models with similar makes and price points were merged into single sub-categories. 
2. Body types with very low sample counts were consolidated into 'Others' sub-category. 
3. Age was derived by subtracting year of purchase from year of sale. 
4. Final features and number of sub-categories are as follows: Brand (53), Model (181), Body (44), State(39), Condition(Rating: 1 to 5), Odometer(continuous numerical series), Color(19), Age(1 to 25).  
5. During EDA it was observed that there is a consistent inverse correlation between Age and Condition (scatter plot below). Subsequently, both ‘Condition’ and ‘Age’ were scaled using MinMax Scaler in Python and 
combined into a single feature ‘Quality’.
- the odometer ratings were encoded into 8 categories, i.e., '0-5k’: 1, '5k-50k’: 2, '50k-100k’: 3, '100k-150k’: 4, '150k-200k’: 5, '200k-250k’: 6, '250k-500k’: 7, '>500k’: 8 and the feature was renamed to ‘Mileage’.
- Furthermore, the quality feature was encoded into 5 categories: '0-.4’: 1, '.4-.8’: 2, '.8-1.2’: 3, '1.2-1.6’: 4, '1.6-2’: 5.
- Brands were broadly classified into 8 categories: Mainstream, Premium, Luxury, Ultra-Luxury, Sports, Utility, Off-road and EV.
- ‘State’ was included as a new feature in the analysis. Both brand category and state were one hot encoded.
- Random Forest Regressor model was re-trained using 7 features as input and MMR as the dependent variable.
- Mileage (odometer) again emerged as the most significant contributor to MMR, followed by other features
- Weight distribution for each feature importance is as follows: mileage- 41%, brand  category-17%, body- 16%, model-15%, state-5%, color-4%, quality-2%. 

- To understand how strongly feature categories are associated with potential losses, Percentage Negative Deviation (PND) were calculated for each sub-category across all features by visualizing them in Power BI.
- PND represents the proportion of samples where the MMR value exceeded the actual sale price (indicating a negative deviation or potential loss). For example, if 39,000 out of 75,000 'Luxury' brand vehicles
  experienced a negative deviation, the PND for the 'Luxury' category would be 52%.
- These PND values were used in defining a 'risk score' for each category, a concept elaborated in the subsequent data preprocessing section.

**Data Pre-processing**: To quantify the risk associated with each category across all seven features, a risk score was computed by combining Percentage Negative Deviation (PND) and sample count.
The calculation involved two steps:
1. Calculate adjusted risk score: 
 -The raw sample count is first normalized by dividing it by 3,00,000.
 -This normalized value and category’s PND value is then summed to together to yield and adjusted score.
2. Find normalized value for risk scores:
- The normalized risk scores for each category were appended to their corresponding rows/samples using VLOOKUP function. A Composite Risk Score was then generated for each individual vehicle by multiplying each
category's risk score by its respective feature's weight

- This Composite Risk Score was instrumental in deriving a more optimized MMR value (termed 'MMR 2’) designed to mitigate potential losses on vehicle sales without excessively reducing the estimated wholesale price.
The adjustment was applied based on the following logic:
If the Composite Risk Score for a vehicle was greater than 0.9, 'MMR 2' was set to 91% of its existing MMR. If the Composite Risk Score was between 0.8 and 0.9, 'MMR 2' was set to 92% of its existing MMR.
And so forth for the remaining risk score ranges.

**Model Development**: A predictive model was developed to forecast a more accurate MMR value, referred to as MMR 2. The dataset was prepared with the following feature engineering and encoding techniques:
Model: Frequency encoding
Brand category, body, color, state: One hot encoding
Mileage, Quality: Binning
Risk Score and Original MMR: These continuous numerical features were used without encoding, as Risk score was already scaled between 0 and 1, and MMR served as a foundational variable for the target.

The dataset was split into an 80:20 ratio for training and testing, respectively. XGB Regressor was selected for model training and prediction, leveraging its performance on structured data.
The model's performance was assessed by measuring its impact on profitability and predictive accuracy. 
Key metrics included:
-Reduction in Negative Deviation: The change in the number of cases where the predicted MMR 2 exceeded the actual selling price, indicating a reduction in potential losses.
-Average Deviation: The mean difference between the selling price and the predicted value, evaluated before and after model application.
-Accuracy Metrics: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R-squared.

**Tools and libraries used**: Software: Python, Power BI, Excel
Libraries: xgboost, sklearn, pandas, numpy, matplotlib, shap
Regression models: XGBRegressor, Random Forest, Linear Regression
Scalers: Minmax scaler, Standard scaler

**Results & Assessment**: -The predictive model was tested on 20% of the dataset, comprising 106,096 samples. The results demonstrate a significant improvement in profitability and prediction accuracy.
A substantial 42.02% reduction was achieved in cases where the predicted MMR value exceeded the sale price, mitigating potential losses.
- The average deviation between selling price and MMR saw a remarkable shift from -$153.09 (using Original MMR) to $634.97 (using Predicted MMR 2) within the tested samples, indicating a substantial improvement
in average profit margin.
- MAE of 88.38 indicates a minimal deviation in predicted values, especially given the large-scale values of MMR. 
- While the RMSE of 860.98 is notably higher than MAE, which indicates there are some extreme outlier cases in the prediction. 
- An R-squared value of 0.9909 confirms that the model is highly accurate at predicting the target value, based on the provided input features.

**Next steps for improving the predictive model**: Following strategies can be used to further improve the predictive model:
- Integrate Macroeconomic and Market Dynamics: Incorporate features representing demand/supply trends and broader economic indicators specific to the time of sale. This will provide the model with crucial context
on market liquidity and sentiment, thereby refining the risk scoring.
- Outlier Analysis and Robustness: Conduct further analysis to identify and understand the characteristics of outlier cases that contribute to extreme deviations in predictions.
- Dealer-Specific Information: Introduce 'Dealer Name' as a feature in the dataset. Dealers are a significant factor influencing sale prices, and including this information will allow the model to capture
  dealer-specific effects, leading to improved risk score and a more accurate target MMR.

**Respository Contents**: 2 python scripts- "feature_importance_mmr.py" for finding feature importance and "predict_mmr.py" for predicting MMR in the folder 'Python scripts and notebooks'.
- 2 Notebooks for feature importance and predictive model are also included for reference in the folder 'Python scripts and notebooks'.
- 1 power bi dashboard showing the details of EDA for find percent negative deviation fro each feature (in the folder 'EDA dashboard').
- Slide set (MMR Predictive Model.pdf) to provide detailed discription of the project and the results (in the folder ;project description').
- The final dataset used for predictive model (mmr.xlsx) in folder 'final dataset'.

**How to run the model**: Pre-requisites: Pthon environmen- python 3.9+, an IDE like JupyterLab, MS Excel and MS Power BI.
1. Download the mmr.xlsx dataset file from the folder 'Final dataset'. Place this file into the python working directory
2. To determine feature importance: open the file 'python scripts and notebooks/feature_importance_mmr.py' in python IDE. Within the feature_importance_mmr.py script, locate and run the code blocks in the
   specified order: upload_files(), normalize_features(), feature_importance(). A simple plot of the feature importances will be displayed directly in your Python environment's console or plot window.
3. To run the predictive model: open the file 'python scripts and notebooks/predict_mmr.py' in the python IDE. Within the predict_mmr.py script, locate and run the code blocks: upload_files(), predict_mmr().
   metrics demonstrating model improvement, will be displayed in your Python environment's console.
4. To view the interactive Exploratory Data Analysis (EDA) dashboard. Navigate to the 'EDA dashboard' folder and open the file 'MMR_EDA.pbix' using Power BI.
5. For a detailed overview of the entire project, including in-depth EDA, model development methodologies, and full results: Navigate to the 'project details' folder and open the file 'MMR Predictive Model.pdf'.

**Acknowledgment**: Thankful to Kaggel and user named Syed Anwar for publishing the car_prices.csv dataset
