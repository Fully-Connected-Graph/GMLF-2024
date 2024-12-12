import pandas as pd
from sklearn.metrics import mean_absolute_error
import os
import numpy as np
from scipy.stats import entropy
import json
import matplotlib.pyplot as plt

def save_result(mae, experiment_name):
    
    # Create results.csv if it doesn't exist
    try:
        pd.read_csv('results.csv')
    except:
        # Create a df with the columns experiment_name and mae and the first row
        df = pd.DataFrame([{'experiment_name': experiment_name, 'mae': mae}])
        df.to_csv('results.csv', index=False)
        return
    
    # add a row to the results.csv file
    results = pd.read_csv('results.csv')
    new_row = pd.DataFrame([{'experiment_name': experiment_name, 'mae': mae}])
    results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv('results.csv', index=False)
    
def save_processed_data(df, filepath, new_folder):
    
    # Get the file name (after the last /)
    file_name = filepath.split('/')[-1]
    
    new_folder_path = f'data/{new_folder}'
    new_path = f'{new_folder_path}/{file_name}'
    
    print(f'Saving the file in {new_path}')

    # Create the folder if it does not exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        
    # Save as csv first
    new_path = new_path.replace('.pkl', '.csv')
    df.to_csv(new_path, index=False)
    
    # replace the file extension with .pkl
    new_path = new_path.replace('.csv', '.pkl')
    df.to_pickle(new_path)
    
def force_save_data(df, filepath):
    # If folder does not exist, create it
    folder = '/'.join(filepath.split('/')[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    print(f'Saving the file in {filepath}')
        
    # Save, no extension is provided
    df.to_pickle(f"{filepath}.pkl")
    df.to_csv(f"{filepath}.csv", index=False)

    
def make_cyclic_features(df, column, max_val):
    
    # Convert column type to float
    df[column] = df[column].astype(float)
    
    df[column + '_sin'] = np.sin(2 * np.pi * df[column]/max_val)
    df[column + '_cos'] = np.cos(2 * np.pi * df[column]/max_val)
    
    # remove the original column
    df.drop(column, axis=1, inplace=True)
    
    return df

def compute_kl_divergence(train_df, test_df):
    
    kl_divergences = {}
    
    # Skip categorical columns
    for column in train_df.columns:
        if column == 'target' or column == 'ID' or train_df[column].dtype.name == 'category':
            continue
        
        # Fill NaN values with the mean of the column
        train_df[column] = train_df[column].fillna(train_df[column].mean())
        test_df[column] = test_df[column].fillna(test_df[column].mean())
        
        # Compute probability distributions
        train_prob, _ = np.histogram(train_df[column], bins=50, density=True)
        test_prob, _ = np.histogram(test_df[column], bins=50, density=True)
        
        # Add a small value to avoid division by zero
        train_prob += 1e-10
        test_prob += 1e-10
        
        # Normalize the distributions
        train_prob /= train_prob.sum()
        test_prob /= test_prob.sum()
        
        # Compute KL divergence
        kl_divergences[column] = entropy(train_prob, test_prob)
        
    return kl_divergences

def compute_sample_weights(train_df, kl_divergences):
    weights = np.zeros(len(train_df))
    
    for column, kl_div in kl_divergences.items():
        if column == 'target':
            continue
        # Compute weights based on KL divergence
        weights += kl_div * (train_df[column] - train_df[column].mean())**2
    
    # Normalize weights
    weights = np.exp(-weights)
    weights /= weights.sum()
    
    return weights

def load_hyperparams(model_name, data_name):
    filepath = f'config/{data_name}/best_params_{model_name}.json'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        print(f"Loaded best hyperparameters for {model_name} from {filepath}.")
        return params
    else:
        print(f"No existing hyperparameter file for {model_name}. Proceeding with search.")
        return {}

# Helper function to save hyperparameters to JSON
def save_hyperparams(model_name, data_name, params):
    
    # Create the folder if it does not exist
    folder = f'config/{data_name}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = f'config/{data_name}/best_params_{model_name}.json'
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Saved best hyperparameters for {model_name} to {filepath}.")
    
def plot_temperature_over_time(df, column='source_1_temperature'):
    """
    Plot temperature over time using either measurement_time or index.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing temperature data
    column (str): Name of temperature column to plot
    """
    plt.figure(figsize=(12, 6))
    
    # Use measurement_time if available, otherwise use index
    if 'measurement_time' in df.columns:
        x_axis = pd.to_datetime(df['measurement_time'])
        x_label = 'Time'
    else:
        
        # Reset the index
        df = df.reset_index()
        
        x_axis = df.index
        x_label = 'Index'
    
    plt.plot(x_axis, df[column], label=column)
    plt.xlabel(x_label)
    plt.ylabel('Temperature')
    plt.title(f'{column} Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()