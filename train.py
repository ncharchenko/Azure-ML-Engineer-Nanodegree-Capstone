from sklearn.ensemble import GradientBoostingClassifier
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
    columns = ['side', 'result', 'golddiffat10', 'xpdiffat10', 'golddiffat15', 'xpdiffat15']

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

x_train, x_test = train_test_split(x, test_size=0.2)

y_train, y_test = train_test_split(y, test_size=0.2)

run = Run.get_context()

def main():
    # Add arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument('--max_leaf_nodes', type=int, default=None, help="Maximum number of leaf nodes. This restricts tree growth.")

    args = parser.parse_args()

    run.log("Number of estimators:", np.int(args.n_estimators))
    run.log("Max leaf nodes:", np.int(args.max_leaf_nodes))

    model = GradientBoostingClassifier(n_estimators=args.n_estimators, max_leaf_nodes=args.max_leaf_nodes).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
