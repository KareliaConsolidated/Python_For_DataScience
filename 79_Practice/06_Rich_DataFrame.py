from rich_dataframe import prettify
import pandas as pd

df = pd.read_csv('data/Housing.csv')    

prettify(df)
