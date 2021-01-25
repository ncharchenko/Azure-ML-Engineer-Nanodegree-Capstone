from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    # Our problem space revolves around predicting game outcomes based on early game data, so we only focus on the early game (stats at 10 and 15 min).
    columns = ['side', 'result', 'firstblood', 'firstherald', 'firsttower', 'golddiffat10', 'xpdiffat10', 'golddiffat15', 'xpdiffat15']

    df = pd.DataFrame(data, columns=columns)

    # We are looking at the games from the lens of Blue side.
    df = df.loc[df['side'] == 'Blue']

    # Side no longer needed so we can delete this field.
    del df['side']

    # Remove any incomplete entries from the dataset.
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    x_df = df[indices_to_keep]

    y_df = x_df.pop("result")

    x_df.reset_index()

    return x_df, y_df

    
        
# Import player data CSV
ds = TabularDatasetFactory.from_delimited_files(path="https://github.com/ncharchenko/Azure-ML-Engineer-Nanodegree-Capstone/raw/master/OraclesElixir_lol_match_data_teams.csv")

player_data = ds.to_pandas_dataframe()

# Process dataframe
x, y = clean_data(player_data)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

run = Run.get_context()

def main():
    # Add arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Stronger regularization comes from smaller values.")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge.")

    args = parser.parse_args()

    run.log("Number of estimators:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()