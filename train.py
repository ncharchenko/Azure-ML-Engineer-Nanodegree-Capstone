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
    # Filter by most significant metrics as well as results and sides.
    player_columns = ['side', 'position', 'result', 'golddiffat10', 'xpdiffat10', 'golddiffat15', 'xpdiffat15']

    df = pd.DataFrame(data, columns=player_columns)

    # We are looking at the stat differences from the lens of Blue side.
    df = df.loc[df['side'] == 'Blue']

    # Side no longer needed so we can delete this field.
    del df['side']

    # Create dummy variables for player positions.
    x_df = pd.get_dummies(df, columns=['position'])

    y_df = x_df.pop("result")
    
    # Positions no longer needed after feature engineering.
    del df['position']

    return x_df, y_df

    
        
# Import player data CSV
ds = TabularDatasetFactory.from_delimited_files(path="https://github.com/ncharchenko/Azure-ML-Engineer-Nanodegree-Capstone/raw/master/OraclesElixir_lol_match_data_players.csv")

player_data = ds.to_pandas_dataframe()

# Process dataframe
x, y = clean_data(player_data)

x_train, x_test = train_test_split(x, test_size=0.2)

y_train, y_test = train_test_split(y, test_size=0.2)

run = Run.get_context()

def main():
    # Add arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()