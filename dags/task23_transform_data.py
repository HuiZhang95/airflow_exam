from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from datetime import datetime

import requests
import os
import json
import pandas as pd

n_files_task2 = 20

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


with DAG(
    dag_id='transform_to_clean_data',
    tags=['transformData', 'datascientest','task23'],
    schedule_interval='* 07 * * *', 
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
        },
    catchup=False
) as dag:

    task1_last20 = PythonOperator(
        task_id = 'task2_last20recodes',
        python_callable = transform_data_into_csv,
        op_kwargs={'n_files': None, 'filename': 'data.csv'}
    )

    task2_all = PythonOperator(
        task_id = 'task3_allrecodes',
        python_callable = transform_data_into_csv,
        op_kwargs={'n_files': n_files_task2, 'filename': 'fulldata.csv'}
    )