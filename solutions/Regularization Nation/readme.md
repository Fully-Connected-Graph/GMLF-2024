# Team Regularization Nation @ MlFortnight 2024 🔥

Hi! This is our attempt at predicting energy consumption for an unknown building in the Netherlands. We did a lot of stuff, here's fun stats and how to run the code.

## Stats 📊
- 1 year of data (Nov-Aug training, Aug-Nov testing) 
- 2 weeks to beat them all 🤯
- 1000+ features which were pruned to 43 🪓
- almost 200 models trained 🏃‍♂️💨
- 4 nights where I couldn't sleep ❌
- 1 broken laptop 💻🔨 

## Project Structure 🏗️

- `src/`: Source code directory
  - `data_pipeline/`: Data processing and feature engineering modules
- `notebooks/`: Jupyter notebooks for analysis and training
- `notebooks/recursive_leakage/`: Final model was trained here, taking the last best model's predictions as the target for the next model
- `data/`: Data directory (not included in repository)

## Requirements 🤝

The project requires Python 3.11+. To install the requirements, run `pip install -r requirements.txt` 🐍

## Running the code 🏃‍♂️💨
- The main training file is `notebooks/train_w_autogluon.ipynb`
- The main prediction file is `notebooks/predict.ipynb`

Yippee!
