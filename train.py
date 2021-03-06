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
    columns = ['side', 'position', 'result', 'golddiffat10', 'xpdiffat10', 'golddiffat15', 'xpdiffat15']

    df = pd.DataFrame(data, columns=columns)

    # We are looking at the games from the lens of Blue side.
    df = df.loc[df['side'] == 'Blue']

    # Take only the team data.
    df = df[df.position.eq('team')]

    # Side no longer needed so we can delete this field.
    df.drop(['side'], inplace=True, axis=1, errors='ignore')
    # We also no longer need positions.
    df.drop(['position'], inplace=True, axis=1, errors='ignore')

    # Remove any incomplete entries from the dataset.
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    x_df = df[indices_to_keep]

    y_df = x_df.pop("result")

    x_df.reset_index()

    return x_df, y_df

    
        
# Import player data CSV
ds = TabularDatasetFactory.from_delimited_files(path="https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/2020_LoL_esports_match_data_from_OraclesElixir_20210126.csv")

player_data = ds.to_pandas_dataframe()

# Process dataframe
x, y = clean_data(player_data)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

run = Run.get_context()

def main():
    # Add arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Stronger regularization comes from smaller values.")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge.")

    args = parser.parse_args()

    run.log("C:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    test_df = model.predict_proba(x_test)[:,1]

    print("Probability model output.")
    print(test_df)

    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.pkl')

if __name__ == '__main__':
    main()