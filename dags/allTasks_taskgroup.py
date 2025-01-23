from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime
import requests
import os
import json
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import numpy as np

n_files_task2 = 20
cities = ['paris', 'london', 'washington']
api_key = '33293be55c88248d0f290a79731a58fc'
url = "https://api.openweathermap.org/data/2.5/weather"
save_raw_foler = '/app/raw_files'
modelPool = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]

def retrieve_data():
    allData = []
    for city in cities:
        response = requests.get(
            url=url,
            params={
                "q": city,
                "appid": api_key
            }
        )
        allData.append(response.json())

    filename = datetime.now().strftime("%d-%m-%Y %H:%M")
    with open(os.path.join(save_raw_foler, filename), 'w') as f:
        json.dump(allData, f)

def transform_data_into_csv(n_files=None, filename='data.csv'):
    parent_folder = '/app/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    if n_files:
        if len(files) > n_files:
            files = files[:n_files]


    dfs = []

    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)
        for data_city in data_temp:
            dfs.append(
                {
                    'temperature': data_city['main']['temp'],
                    'city': data_city['name'],
                    'pression': data_city['main']['pressure'],
                    'date': f.split('.')[0]
                }
            )

    df = pd.DataFrame(dfs)

    print('\n', df.head(10))

    df.to_csv(os.path.join('/app/clean_data', filename), index=False)


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
    features_json = task_instance.xcom_pull(task_ids='task45.task1_repareData', key='features')
    target_json = task_instance.xcom_pull(task_ids='task45.task1_repareData', key='target')

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
        score = task_instance.xcom_pull(task_ids=f'task45.task234_training_models.task_model{count+1}', key='model_score')
        model = task_instance.xcom_pull(task_ids=f'task45.task234_training_models.task_model{count+1}', key='model_name')
        scores.append(score)
        models.append(model)

    print("best model scores are{}".format(scores))
    clf = models[np.argmin(scores)]
    print("best model is{}".format(clf))
    task_instance.xcom_push(key="model_name",value=clf)

def retrain_bestModel(task_instance, path_to_model = '/app/clean_data/best_model.pickle'):
    model = eval(task_instance.xcom_pull(task_ids='task45.task5_selectBestModel', key='model_name'))

    # Pull features and target from XCom
    features_json = task_instance.xcom_pull(task_ids='task45.task1_repareData', key='features')
    target_json = task_instance.xcom_pull(task_ids='task45.task1_repareData', key='target')

    # Convert JSON back to DataFrame and Series
    features = pd.read_json(features_json)
    target = pd.read_json(target_json, typ='series')

    # training the model
    model.fit(features, target)
    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)

with DAG(
    dag_id='full_pipline',
    tags=['full_pipeline', 'datascientest','all_tasks'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False
) as dag:

    task1 = PythonOperator(
        task_id = 'task1_retrieveData',
        python_callable = retrieve_data
    )
    
    with TaskGroup("task23") as task23:
        task2_last20 = PythonOperator(
            task_id = 'task2_last20recodes',
            python_callable = transform_data_into_csv,
            op_kwargs={'n_files': None, 'filename': 'data.csv'}
        )

        task3_all = PythonOperator(
            task_id = 'task3_allrecodes',
            python_callable = transform_data_into_csv,
            op_kwargs={'n_files': n_files_task2, 'filename': 'fulldata.csv'}
        )

    with TaskGroup("task45") as task45:
        task45_1 = PythonOperator(
            task_id = 'task1_repareData',
            python_callable = prepare_data,
            op_kwargs={'path_to_data':'/app/clean_data/fulldata.csv'}
        )

        with TaskGroup("task234_training_models") as task45_234:
            tasks = []
            count = 1
            for model in modelPool:
                task = PythonOperator(
                    task_id = f'task_model{count}',
                    python_callable = compute_model_score,
                    op_kwargs={'model':model}
                )
                count+=1

        task45_5 = PythonOperator(
            task_id = 'task5_selectBestModel',
            python_callable = select_bestModel
        )

        task45_6 = PythonOperator(
            task_id = 'task6_retrain_bestModel',
            python_callable = retrain_bestModel,
            op_kwargs={'path_to_model':'/app/clean_data/best_model.pickle'}
        )

        task45_1 >> task45_234 >> task45_5 >> task45_6

    task1 >> task23 >> task45

