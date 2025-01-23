from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.models import Variable
import requests
import os
import json

Variable.set(key="cities", value=['paris', 'london', 'washington'])
cities = ['paris', 'london', 'washington']
api_key = '33293be55c88248d0f290a79731a58fc'
url = "https://api.openweathermap.org/data/2.5/weather"
save_raw_foler = '/app/raw_files'

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



with DAG(
    dag_id='retrieve_data_from_web',
    tags=['retrieve_data', 'datascientest','task1'],
    schedule_interval='* * * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False
) as dag:

    python_retrieveData = PythonOperator(
        task_id = 'task1_retrieveData',
        python_callable = retrieve_data
    )









