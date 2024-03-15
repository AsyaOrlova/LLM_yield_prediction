import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from classifier.configuration import Configuration
from pathlib import Path
from sys import argv
import re

path = argv[1]

df = pd.read_json(f'{path}/results.json')

def check_clf(x):
    if 'High-yielding' in x[0]:
        x[0] = 'High-yielding'
    elif 'Not high-yielding' in x[0]:
        x[0] = 'Not high-yielding'
    else:
        x[0]
    return x

def check_reg(x):
    n = re.findall(r"\d+\.\d+", x[0])
    if n == []:
        n = ['101.0']
    return n

config_path = './config_asya'
config = Configuration.load(Path(config_path))

def string_to_continuous(income_data: list, classes: list) -> pd.DataFrame:

    """This function wraps model responses into pd.DataFrame format"""

    df = pd.DataFrame(0, columns=classes, index=range(len(income_data)))
    df[classes] = df.apply(
        lambda x: [1 if col in income_data[x.name] else 0 for col in classes],
        axis=1,
        result_type="expand",
    )
    return df

def string_to_continuous_reg(income_data: list, classes: list) -> pd.DataFrame:

    """This function wraps model responses into pd.DataFrame format"""
    df = pd.DataFrame(0, columns=classes.split(), index=range(len(income_data)))
    df[classes] = df.apply(
        lambda x: [float(i[0]) for i in income_data]
    )
    return df

def calculate_classification_metrics(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, path: str, config: dict
):

    """This func calculates standard classification metrics and saves it to ./experiments/your_name"""

    class_metrics = []

    for class_label in y_true.columns:
        precision = precision_score(
            y_true[class_label], y_pred[class_label], zero_division=1
        )

        recall = recall_score(y_true[class_label], y_pred[class_label], zero_division=1)

        f1 = f1_score(y_true[class_label], y_pred[class_label], zero_division=1)

        class_metrics.append(
            {
                "Class": class_label,
                "Accuracy": accuracy_score(y_true[class_label], y_pred[class_label]),
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=1)

    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=1)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)

    micro_precision = precision_score(
        y_true.values.flatten(), y_pred.values.flatten(), zero_division=1
    )

    micro_recall = recall_score(
        y_true.values.flatten(), y_pred.values.flatten(), zero_division=1
    )

    micro_f1 = f1_score(
        y_true.values.flatten(), y_pred.values.flatten(), zero_division=1
    )

    class_metrics.append(
        {
            "Class": "Macro-Average",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": macro_precision,
            "Recall": macro_recall,
            "F1": macro_f1,
        }
    )

    class_metrics.append(
        {
            "Class": "Micro-Average",
            "Accuracy": accuracy_score(
                y_true.values.flatten(), y_pred.values.flatten()
            ),
            "Precision": micro_precision,
            "Recall": micro_recall,
            "F1": micro_f1,
        }
    )

    df_metrics = pd.DataFrame(class_metrics)
    df_metrics.to_csv(
        f'{path}/metrics.csv',
        index=False,
        sep=";",
        header=True,
        encoding="utf-8",
    )
    
def calculate_regression_metrics(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, target_col_name: str, path: str, config: dict
):
    """This func calculates standard regression metrics and saves it to ./experiments/your_name"""
    
    r2 = r2_score(y_true[target_col_name], y_pred[target_col_name])
    mae = mean_absolute_error(y_true[target_col_name], y_pred[target_col_name])
    mse = mean_squared_error(y_true[target_col_name], y_pred[target_col_name])
    rmse = mean_squared_error(y_true[target_col_name], y_pred[target_col_name])**0.5
    
    df_metrics = pd.DataFrame({'R^2': [r2], 'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse]})
    df_metrics.to_csv(
        f'{path}/metrics.csv',
        index=False,
        sep=";",
        header=True,
        encoding="utf-8",
    )

if config['task']=='regression':
    df.target_classes = df.target_classes.apply(check_reg)
    df.predicted_classes = df.predicted_classes.apply(check_reg)
    calculate_regression_metrics(y_true=string_to_continuous_reg(income_data=df.target_classes,
                                                                    classes=config["target_column"]),
                                        y_pred=string_to_continuous_reg(income_data=df.predicted_classes,
                                                                    classes=config["target_column"]),
                                        target_col_name=config["target_column"],
                                        config=config,
                                        path=argv[1]
                                        )
else:
    df.target_classes = df.target_classes.apply(check_clf)
    df.predicted_classes = df.predicted_classes.apply(check_clf)
    calculate_classification_metrics(y_true=string_to_continuous(income_data=df.target_classes,
                                                                    classes=config["classes"]),
                                        y_pred=string_to_continuous(income_data=df.predicted_classes,
                                                                    classes=config["classes"]),
                                        config=config,
                                        path=argv[1]
                                        )