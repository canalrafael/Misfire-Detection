from pyEasyML import Settings
import os
settings = Settings(os.path.abspath(__file__))

import pandas as pd
from pyEasyML.Data.DataPreprocessing import DataPreprocessor
from pyEasyML.Data.FeatureSelection import FeatureSelector
from sklearn.feature_selection import r_regression
from pyEasyML.Classification.Classifier import Classifier

def save_cleaned_dataset(data_preprocessor:DataPreprocessor) -> None:
    dataset = data_preprocessor.read_dataset(data_preprocessor.DATASET_PATH)
    dataset = clean_dataset(dataset)
    data_preprocessor.save_dataset(dataset, data_preprocessor.DATASET_PATH)

def clean_dataset(df:pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by removing non-numeric columns and handling missing values.
    This is a standard implementation, but you can implement your own.

    Returns:
        tuple[pd.DataFrame, ...]: The cleaned dataset.
    """
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.select_dtypes(include=['number'])
    df = handle_missing_values(df)

    return df

def handle_missing_values(dataset:pd.DataFrame) -> pd.DataFrame:
    """Handles missing values by removing columns with all missing values and filling the rest with the mean of the column.
    This is a standard implementation, but you can implement your own.

    Args:
        dataset (pd.DataFrame): The dataset to be cleaned.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    X = dataset.columns.to_list()
    for x in X:
        if dataset[x].isna().all():
            dataset.drop(labels=[x], axis=1, inplace=True)
        elif dataset[x].isna().any():
            dataset[x] = dataset[x].fillna(dataset[x].mean())

    return dataset

def balance_dataset(train_val_dataset:pd.DataFrame) -> pd.DataFrame:
    """Balances the dataset due to rarity of missfire events. This is a standard implementation, but you can implement your own.

    Args:
        train_val_dataset (pd.DataFrame): The dataset to be balanced.

    Returns:
        pd.DataFrame: The balanced dataset.
    """
    healthy = train_val_dataset[train_val_dataset[settings.get_target_feature()] == 0]
    anomalous = train_val_dataset[train_val_dataset[settings.get_target_feature()] == 1]

    anomalous_size = len(anomalous)
    resized_healthy = healthy.sample(n=anomalous_size, random_state=42)

    concat_trained_val_dataset = pd.concat([resized_healthy, anomalous])

    train_val_dataset = concat_trained_val_dataset.sample(frac=1, random_state=42)
    
    return train_val_dataset

settings.set_target_feature("") # Place the name of your missfire column here.

data_preprocessor = DataPreprocessor()
data_preprocessor.DATASET_PATH = "" # Place the path of your dataset here.

#save_cleaned_dataset(data_preprocessor) # Uncomment this line if you want to clean your dataset. If dataset is already cleaned, you can skip this step.

train_val_dataset, test_dataset = data_preprocessor.get_train_val_test_datasets()

#train_val_dataset = balance_dataset(train_val_dataset) # Uncomment this line if you want to balance your dataset.

X_train, X_test, Y_train, Y_test = data_preprocessor.gen_train_test_datasets(dataset=train_val_dataset, columns=train_val_dataset.columns)

feature_selector = FeatureSelector(
    X_train, Y_train,
    model_name="GradientBoostingClassifier", # or "XGBClassifier", "GradientBoostingClassifier", "LogisticRegression", "SVC", "KNN"
    columns=train_val_dataset.columns
)

selected_features = feature_selector.select_k_best(k=5, func=r_regression) # or select_percentile, sequential_feature_selector, recursive_feature_elimination, select_from_model
print(selected_features)
settings.set_selected_features(selected_features)

X_train, X_test, Y_train, Y_test = data_preprocessor.gen_train_test_datasets(dataset=train_val_dataset)

classifier = Classifier(
                    model_name="GradientBoostingClassifier", # or "XGBClassifier", "GradientBoostingClassifier", "LogisticRegression", "SVC", "KNN"
                    default_columns=True,
                    default_target=True,
                    #...param_1 = value_1,...,param_n = value_n
                )
classifier.fit(X_train=X_train, Y_train=Y_train)

confusion_matrix = classifier.evaluate(X_test=X_test, Y_test=Y_test)
print(confusion_matrix)
