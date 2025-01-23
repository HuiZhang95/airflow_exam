from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import numpy as np

modelPool = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]

def prepare_data(task_instance, path_to_data='/app/clean_data/fulldata.csv'):
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for c in df['city'].unique():
        df_temp = df[df['city'] == c]

        # creating target
        df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)

        # creating features
        for i in range(1, 10):
            df_temp.loc[:, 'temp_m-{}'.format(i)
                        ] = df_temp['temperature'].shift(-i)

        # deleting null values
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # concatenating datasets
    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )

    # deleting date variable
    df_final = df_final.drop(['date'], axis=1)

    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)

    features = df_final.drop(['target'], axis=1)
    target = df_final['target']

    task_instance.xcom_push(key='features', value=features.to_json())
    task_instance.xcom_push(key='target', value=target.to_json())

def compute_model_score(task_instance, model):
    # Pull features and target from XCom
    features_json = task_instance.xcom_pull(task_ids='task1_repareData', key='features')
    target_json = task_instance.xcom_pull(task_ids='task1_repareData', key='target')

    # Convert JSON back to DataFrame and Series
    features = pd.read_json(features_json)
    target = pd.read_json(target_json, typ='series')

    # computing cross val
    cross_validation = cross_val_score(model, features, target, cv=3, scoring='neg_mean_squared_error')
    model_score = cross_validation.mean()
    task_instance.xcom_push(key='model_score',value=model_score)
    task_instance.xcom_push(key='model_name',value=str(model))

def select_bestModel(task_instance):
    scores = []
    models = []
    for count in range(len(modelPool)):
        score = task_instance.xcom_pull(task_ids=f'task234_training_models.task_model{count+1}', key='model_score')
        model = task_instance.xcom_pull(task_ids=f'task234_training_models.task_model{count+1}', key='model_name')
        scores.append(score)
        models.append(model)

    print("best model scores are{}".format(scores))
    clf = models[np.argmin(scores)]
    print("best model is{}".format(clf))
    task_instance.xcom_push(key="model_name",value=clf)

def retrain_bestModel(task_instance, path_to_model = '/app/clean_data/best_model.pickle'):
    model = eval(task_instance.xcom_pull(task_ids='task5_selectBestModel', key='model_name'))

    # Pull features and target from XCom
    features_json = task_instance.xcom_pull(task_ids='task1_repareData', key='features')
    target_json = task_instance.xcom_pull(task_ids='task1_repareData', key='target')

    # Convert JSON back to DataFrame and Series
    features = pd.read_json(features_json)
    target = pd.read_json(target_json, typ='series')

    # training the model
    model.fit(features, target)
    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)





with DAG(
    dag_id='task45_train_models_v2',
    tags=['trainModels', 'datascientest','task45'],
    schedule_interval=None, 
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
        },
    catchup=False
) as dag:

    task1 = PythonOperator(
        task_id = 'task1_repareData',
        python_callable = prepare_data,
        op_kwargs={'path_to_data':'/app/clean_data/fulldata.csv'}
    )

    with TaskGroup("task234_training_models") as task234:
        tasks = []
        count = 1
        for model in modelPool:
            task = PythonOperator(
                task_id = f'task_model{count}',
                python_callable = compute_model_score,
                op_kwargs={'model':model}
            )
            count+=1

    task5 = PythonOperator(
        task_id = 'task5_selectBestModel',
        python_callable = select_bestModel
    )

    task6 = PythonOperator(
        task_id = 'task6_retrain_bestModel',
        python_callable = retrain_bestModel,
        op_kwargs={'path_to_model':'/app/clean_data/best_model.pickle'}
    )

    task1 >> task234 >> task5 >> task6