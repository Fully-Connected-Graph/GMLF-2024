import os
import warnings

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

warnings.filterwarnings("ignore")
np.random.seed(123)


train_path = "./cbrt_hourcos_isweekend_pruned_3ma12_inter_-13_tempcos-cor.csv"
sub_path = "./cbrt_hourcos_isweekend_pruned_3ma12_inter_-13_tempcos-cor_submit.csv"

train_data = pd.read_csv(train_path)
sub_data = pd.read_csv(sub_path)


def recursive_leakage(
    train_data: pd.DataFrame,
    sub_data: pd.DataFrame,
    max_iterations: int = 22,
):
    for iteration in range(max_iterations):
        print(f"\n\n --------------------- {iteration} ---------------------\n\n")
        identifier = f"final_validation_{iteration}fold_best"
        if iteration == 0:
            sub_data["target"] = pd.read_csv(
                "/Users/akseljoonas/Documents/mlfortnight/notebooks/recursive_leakage/lol_validation_4fold_best.csv"
            )["target"]
        else:
            sub_data["target"] = pd.read_csv(
                f"lol_validation_{iteration-1}fold_best.csv"
            )["target"]

        tuning_data = sub_data.copy()
        sub_data = sub_data.drop(columns=["target"])

        train_data = pd.concat([train_data, tuning_data], ignore_index=True)

        label = "target"
        problem_type = "regression"

        predictor = TabularPredictor(
            label=label,
            problem_type=problem_type,
            eval_metric="mean_absolute_error",
            path=f"./AutogluonModels/{identifier}",
        )

        predictor = predictor.fit(
            presets="best",
            train_data=train_data,
            auto_stack=True,
            time_limit=1800,
        )

        preds = pd.DataFrame(predictor.predict(sub_data))
        preds["ID"] = pd.read_csv("test.csv")["ID"]

        # Save the predictions for the next iteration
        preds.to_csv(
            f"lol_validation_{iteration}fold_best.csv",
            columns=["ID", "target"],
            index=False,
        )


recursive_leakage(train_data, sub_data)
