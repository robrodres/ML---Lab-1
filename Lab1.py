import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer



# CHOSEN DATASET: economy.csv


# 1. Explore and understand the dataset
#-----------------------------------------------------------------------------


print ("1. EXPLORING THE DATASET")

df = pd.read_csv("economy.csv")
print ("Dataset shape: ")
print (df.shape)

print ("\nColumn names: ")
print (df.columns.tolist())

print ("\nData types: ")
print (df.dtypes)

print ("\nFirst 5 rows: ")
print (df.head().to_string())

print ("\nBasic descriptive statistics: ")
print (df.describe(include="all").T.to_string())

# missing values count

print ("\nMissin values in each column: ")
miss_val = df.isnull().sum()
print (miss_val.to_string())
print ("\nTotal missing values: ")
print(miss_val.sum)

# duplicates

duplicates = df.duplicated().sum()
print ("\nDuplicated obervations: ")
print(duplicates)

if duplicates > 0 :
    print ("\nDuplicated rows: ")
    print (df[df.duplicated(keep=False)].to_string(index=False))

# remove duplicate observations

data = df.drop_duplicates().copy()
print ("\nShape after removin duplicates: ")
print (data.shape)

# Convert price -> numerical variable

data ["price"] = pd.to_numeric(
    data["price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip(),
    errors = "coerce"
)

print ("\nPrice conversion: ")
print ("Invalid price values: ", data["price"].isna().sum())
print ("Prices <= 0: ", (data["price"]<=0).sum())
print ("\nPrice statistics: ")
print(data["price"].describe().to_string())

# check DATES

data["date_parsed"] = pd.to_datetime(
    data["date"],
    format = "%d-%m-%Y",
    errors = "coerce"
)
print ("\nDate quality: ")
print ("Invalid dates: ", data["date_parsed"].isna().sum())
print ("Minimum date: ", data["date_parsed"].min())
print ("Maximum date: ", data["date_parsed"].max())
print ("Unique dates: ", data["date_parsed"].nunique())

# check DEPARTURES & ARRIVALS

time_pattern = r"^\d{1,2}:\d{2}$"

for column in ["dep_time", "arr_time"]:
    invalid = ~data[column].astype(str).str.match(
        time_pattern,
        na=False
    )

    print(
        f"Invalid {column} values: ",
        invalid.sum()
    )


# convert hh:mm ---> minutes after midnite

def clock_to_mins(value):
    parts = str(value).split(":")
    return int(parts[0]) * 60 + int(parts[1])

        #⬇⬇⬇ applies function to all values, no need to run through a loop to apply individually
data["dep_minutes"] = data["dep_time"].apply(clock_to_mins)
data["arr_minutes"] = data["arr_time"].apply(clock_to_mins)

print ("\nDEPARTURE time range: ", data["dep_minutes"].min(), "-", data["dep_minutes"].max())
print ("\nARRIVAL time range: ", data["arr_minutes"].min(), "-", data["arr_minutes"].max())


# convert fligth duration -> minutes

def duration_to_mins(value):

    value = str(value).strip()

    match = re.fullmatch(r"(\d+)h(?:\s*(\d+)m)", value)

    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2) or 0)

        return hours * 60 + minutes

    # checking for unusual decilam formatting
    match = re.fullmatch (r"(\d+)\.(\d{2})h\s*m", value)

    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))

        return hours * 60 + minutes

    return np.nan

        #⬇⬇⬇ apply
data ["duration_minutes"] = data["time_taken"].apply(duration_to_mins)
print ("\nDuration quality: ")
print ("Invalid duration values: ", data["duration_minutes"].isna().sum())
print ("Minimum duration: ", data["duration_minutes"].min(), "minutes")
print ("Maximum duration: ", data["duration_minutes"].max(), "minutes")



# clean the stop variable

data["stop_clean"] = (
    data["stop"]
    .astype(str)
    .str.replace(r"\s+", " ", regex = True)
    .str.strip()
)

print ("\nMost common stop values: ")

print(
    data["stop_clean"]
    .value_counts()
    .head(10)
    .to_string()
)


# convert stops ---> number of stops

def get_number_of_stops(value):

    if value == "non-stop":
        return 0

    match = re.match(r"(\d+)\+?-stop", value)

    if match:
        return int(match.group(1))

    return np.nan

        #⬇⬇⬇ apply
data["stops"] = data["stop_clean"].apply(get_number_of_stops)

print ("\nInvalid stop values: ", data["stops"].isna().sum())
print ("\nNumber of stops: ")
print (
    data["stops"]
    .value_counts()
    .sort_index()
    .to_string()
)


# check categorical values

for column in ["airline", "from", "to"]:

    print ( f"\n{column}: {data[column].nunique()} unique values")
    print (sorted(data[column].dropna().unique()))

# check identical departure->destintaion

same_origin_dest = (
    data["from"] == data["to"]
).sum()

print ("\nFlights with same Origin & Destination", same_origin_dest)


# check possible price outliers

q1 = data["price"].quantile(0.25)
q3 = data["price"].quantile(0.75)


iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

high_price_outliers = (
    data["price"]>upper_bound
).sum()

print ("\nPrice IQR outlier check: ")

print ("Q1", q1)
print ("Q3", q3)
print ("IQR", iqr)

print ("Lower bound: ", lower_bound)
print ("Upper bound: ", upper_bound)
print ("Potential price outliers: ", high_price_outliers)

            # tickets with high price may still represent reality, so not taken out



# Visualizations

plot_sample = data.sample(
    n = min(15000, len(data)),
    random_state = 42
)

# Visualization 1 - Price distribution
plt.figure(figsize=(10,5))

plt.hist(
    data["price"],
    bins = 60
)

plt.xlabel("Ticket price")
plt.ylabel("Frequency")

plt.title ("1.- Distribution of Economy Class Ticket Prices")

plt.tight_layout()
plt.show()

# Visualization 2 - Price & airline
plt.figure(figsize=(10, 5))

data.boxplot(
    column = "price",
    by = "airline",
    grid = False
)

plt.suptitle("")

plt.title ("2.- Ticket Price by Airline")

plt.xlabel ("Airline")
plt.ylabel ("Price")

plt.xticks(rotation = 30)

plt.tight_layout()
plt.show()

# Visualization 3 - flight duration & price
plt.figure(figsize=(10, 5))

plt.scatter(
    plot_sample["duration_minutes"],
    plot_sample["price"],
    alpha = 0.25,
    s = 10
)

plt.xlabel ("Flight duration (minutes)")
plt.ylabel ("Ticket price")
plt.title ("3.- Flight Duration vs Ticket Price")

plt.tight_layout()
plt.show()

# Visualization 4 - price by number of stops

plt.figure(figsize=(8, 5))

data.boxplot(
    column = "price",
    by = "stops",
    grid = False
)

plt.suptitle("")

plt.title ("4.- Ticket Price by Number of Stops")

plt.xlabel("Number of stopr")
plt.ylabel("Price")

plt.tight_layout()
plt.show()



# MAIN OBSERVATIONS
#-------------------------------


mean_by_airline = (
    data.groupby("airline")["price"]
    .mean()
    .sort_values(ascending = False)
)

mean_by_stops = (
    data.groupby("stops")["price"]
    .mean()
)

print ("\nMAIN OBSERVATIONS")
print ("------------------------")


print (f"""The dataset contains {len(data)} observations after
the removal of duplicated rows.
            
The target price ranges between:
- Minimum: {data['price'].min():,.0f}
- Maximum: {data['price'].max():,.0f}

The mean ticket price is:
- {data['price'].mean():,.2f}

{mean_by_airline.index[0]} has the highest average
ticket price.: {mean_by_airline.iloc[0]:,.2f}

{mean_by_airline.index[-1]} has the lowest average
ticket price.: {mean_by_airline.iloc[-1]:,.2f}
            
Flights without stop have an average price of:
{mean_by_stops.loc[0]:,.2f}
            
The visualizations show differences in ticket prices
depending on the airline used and the number of stops
of the flight, and the relationship between flight
duration and price.\n\n""")


if data["price"].skew() > 1:

    print(
        "The price distribution is right-skewed, with a "
        "relatively long upper tail."
    )

print(
    f"{duplicates} exact duplicate rows were found and removed."
)

print(
    "No missing values were found in the original dataset.\n\n"
)


# 2. Prepare the data
#-----------------------------------------------------------------------------

print ("2. PREPARING THE DATA")


data["month"] = data["date_parsed"].dt.month

data["day_of_month"] = (data["date_parsed"].dt.day)

data["day_of_week"] = (data["date_parsed"].dt.dayofweek)


#target variable
target = "price"


#model features
numeric_features = [
    "dep_minutes",
    "arr_minutes",
    "duration_minutes",
    "stops",
    "month",
    "day_of_month",
    "day_of_week"
]

categorical_features = [
    "airline",
    "from",
    "to"
]

feature_columns = (
    numeric_features + categorical_features
)

x = data[feature_columns].copy()
y = data[target].copy()



print ("\nSelected model features: ")

for feature in feature_columns:
    print("-", feature)

print(f"""\nTarget variable: {target}
Features shape: {x.shape}
Target shape: {y.shape}""")


# excluded and converted columns
print ("""\n\nExcluded columns and reasons:
- date: replaced by month, day_of_month and day_of_week
- ch_code: duplicates the airline information
- num_code: flight number is an identifier, not a numerical variable
- dep_time: converted to dep_minutes
- arr_time: also converted, to arr_minutes
- time_taken: also converted, to duration_minutes
- stop: converted to numeric value of stops""")


# PREPROCESSING
#---------------------------

numeric_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            numeric_preprocessor,
            numeric_features
        ),

        (
            "cat",
            categorical_preprocessor,
            categorical_features
        )
    ]
)

print ("""\nPreprocessing pipeline created.
\nNumerica variables:
    - Missing values: median imputation
    - Scaling: StandardScaler
\nCategorical variables:
    - Missing values: most frequent value
    - Encoding: One-Hot Encoding
\nThe preprocessing pipeline has NOT been fitted yet.
It will be fitted only using the training data
after the train/test split.""")


# 3. Create training and test data
#-----------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

# 80% testing, 20% training

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.20,
    random_state = 42
)

print (f"""
3. CREATING TRAINING AND TEST DATA


Training set:
    - x_train: {x_train.shape}
    - y_train: {y_train.shape}

Test set:
    - x_test: {x_test.shape}
    - y_test: {y_test.shape}""")




# 4. Build at least two regression models
#-----------------------------------------------------------------------------


# The chosen regression algorithms are:
# 1. Linear Regression
# 2. Rando Forest Regression

# Both use the same preprocessing strategy included inside the pipelines
# from "PREPROCESSING" section  ↑ ↑ ↑



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline



print ("4. REGRESSION MODELS")



def create_preprocessor():      # independent preprocessor for each model

    return ColumnTransformer(
        transformers = [
            (
                "num",
                numeric_preprocessor,
                numeric_features
            ),
            (
                "cat",
                categorical_preprocessor,
                categorical_features
            )
        ]
    )


# Linear Regression
linear_model = Pipeline (
    steps = [
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)

# Random Forest
rand_for_model  = Pipeline(
    steps = [
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators= 100,
                random_state= 42,
                n_jobs= -1
            )
        )
    ]
)


# TRAINING, same set
print ("\n\nCurrently training Linear Regression ⏳")

linear_model.fit(
    x_train,
    y_train
)

print ("LR Training finished")
print ("\nCurrently training Random Forest ⏳")

rand_for_model.fit(
    x_train,
    y_train
)

print ("RF Training finished")



# 5. Evaluate the models
#-----------------------------------------------------------------------------


# Mean Absolute Error (MAE) --> lower = smaller prediction error
# Root Mean Squared Error (RMSE) --> lower = smaller prediction error
# R-squared (R^2) --> higher = explains larger proportion of variation


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# predictions

linear_predictions = linear_model.predict (x_test)

rand_for_predictions = rand_for_model.predict (x_test)


# metrics calculations
def metrics_calc(model_name, actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )
    rmse = np.sqrt(
        mean_squared_error(
        actual,
        predicted
        )
    )
    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r2
    }

linear_results = metrics_calc(
    "Linear Regression",
    y_test,
    linear_predictions
)

rand_for_results = metrics_calc(
    "Random Forest Regression",
    y_test,
    rand_for_predictions
)

# show results
results = pd.DataFrame(
    [
        linear_results,
        rand_for_results
    ]
)

print ("\n\n5. MODEL EVALUATION\n")
print ({results.to_string(index=False)})




# 6. Improve and tune the models
#-----------------------------------------------------------------------------


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# 6.1 TUNING LR ----------

# new pipeline
tuned_linear_pipeline = Pipeline (
    steps = [
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)
# params
linear_parameters = {
    "model__fit_intercept": [
        True,
        False
    ]
}
# 5-fold cross-validation
linear_search = GridSearchCV(
    estimator=tuned_linear_pipeline,
    param_grid=linear_parameters,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    refit=True
)

linear_search.fit(
    x_train,
    y_train
)
# show
print ("\n\n6. IMPROVED MODELS RESULTS\n")
print("\nBest Linear Regression parameters:")
print(linear_search.best_params_)

print(
    "\nBest cross-validation RMSE:"
)

print(
    -linear_search.best_score_
)

# 6.2 TUNING RFR ---------

# new pipeline
tuned_rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)
#params
rf_parameters = {
    "model__n_estimators": [
        100,
        150
    ],

    "model__max_depth": [
        None,
        15,
        25
    ],

    "model__min_samples_split": [
        2,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2
    ],

    "model__max_features": [
        "sqrt",
        0.7,
        1.0
    ]
}
rf_search = RandomizedSearchCV(         # limits and randomizes the combinations
    estimator=tuned_rf_pipeline,
    param_distributions=rf_parameters,
    n_iter=4,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    refit=True
)

rf_search.fit(
    x_train,
    y_train
)
#show
print("\nBest Random Forest parameters:")
print(rf_search.best_params_)

print(
    "\nBest cross-validation RMSE:"
)

print(
    -rf_search.best_score_
)