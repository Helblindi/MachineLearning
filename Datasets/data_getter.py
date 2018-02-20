import pandas as pd


def get_data_uci_car_evaluation():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    print("Getting UCI Car Evaluation")
    df = pd.read_csv("../Datasets/UCI_Car_Evaluation.csv",
                      header=None,
                      names=headers,
                      index_col=False)

    # replace each column 1 by 1
    df['buying'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4.0, 3.0, 2.0, 1.0], inplace=True)
    df['maint'].replace(to_replace=['vhigh', 'high', 'med', 'low'], value=[4.0, 3.0, 2.0, 1.0], inplace=True)
    df['doors'].replace(to_replace=['2', '3', '4', '5more'], value=[1.0, 2.0, 3.0, 4.0], inplace=True)
    df['persons'].replace(to_replace=['2', '4', 'more'], value=[1.0, 2.0, 3.0], inplace=True)
    df['lug_boot'].replace(to_replace=['small', 'med', 'big'], value=[1.0, 2.0, 3.0], inplace=True)
    df['safety'].replace(to_replace=['low', 'med', 'high'], value=[1.0, 2.0, 3.0], inplace=True)
    df['class'].replace(to_replace=['unacc', 'acc', 'good', 'vgood'], value=[1.0, 2.0, 3.0, 4.0], inplace=True)

    # return the dataframe
    return df.values[:, :-1], df.values[:, -1]


def get_data_pima_indians_diabetes():
    headers = ["num_pregnant", "plasma_glucose_con", "diastolic_bp", "tri_thickness",
               "2hr_serum_insulin", "bmi", "diabetes_pedigree_function", "age", "class"]
    df = pd.read_csv("../Datasets/Pima_Indians_Diabetes.txt",
                     header=None,
                     names=headers,
                     index_col=False)

    # replace the null values with the mode of each column
    # returns the mode of each column
    modes = df.mode().values[0]

    df['plasma_glucose_con'].replace(to_replace=[0], value=[modes[1]], inplace=True)
    df['diastolic_bp'].replace(to_replace=[0], value=[modes[2]], inplace=True)
    df['tri_thickness'].replace(to_replace=[0], value=[modes[3]], inplace=True)
    df['2hr_serum_insulin'].replace(to_replace=[0], value=[modes[4]], inplace=True)
    df['bmi'].replace(to_replace=[0], value=[modes[5]], inplace=True)
    df['diabetes_pedigree_function'].replace(to_replace=[0], value=[modes[6]], inplace=True)
    df['age'].replace(to_replace=[0], value=[modes[7]], inplace=True)

    # return the dataframe
    return df.values[:, :-1], df.values[:, -1]


def get_data_automobile_mpg():
    headers = ["mpg", "cylinders", "displacement", "horsepower", "weight",
               "acceleration", "model_year", "origin", "car_name"]
    df = pd.read_csv("../Datasets/Automobile_MPG.txt",
                     header=None,
                     names=headers,
                     na_values='?',
                     delim_whitespace=True,
                     index_col=False)

    # drop the car_name column per it does not appear to contain any valuable information
    df = df.drop("car_name", axis=1)

    df = pd.get_dummies(df, columns=['origin'])

    # drop the rows with NA values as it only accounts for 2% of the data
    df.dropna(inplace=True)

    # return the dataframe
    return df


def get_data_movie_profit():
    headers = ["type", "plot", "stars", "profit"]
    df = pd.read_csv("../Datasets/Movie_Profit.csv",
                     header=None,
                     names=headers,
                     index_col=False)

    # return the dataframe
    return df