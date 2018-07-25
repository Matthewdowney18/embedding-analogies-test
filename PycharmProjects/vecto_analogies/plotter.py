import pandas
import matplotlib.pyplot as plt
import re
import os


def plot_data(dataframes):
    df = pandas.concat(dataframes)
    df = df.set_index('filename')
    df = df.transpose()
    df = df.drop(index='citation')
    df = df.drop(index='description')
    df = df.drop(index='embedding')
    df = df.astype(float)
    df_results = df
    df_missing = df
    for index2, row in df.iterrows():
        searchobj = re.search('results', str(index2))
        searchobj2 = re.search('missing', str(index2))
        if searchobj:
            df_missing = df_missing.drop(index=index2)
        elif searchobj2:
            df_results = df_results.drop(index=index2)
    df_results.plot(title='Results', kind='bar', figsize = (10,10), linewidth = 10)
    plt.show()

def main(directory):
    dfs = []
    for filename in os.listdir(directory):
        df = pandas.read_csv(directory+filename)
        df['filename'] = filename
        dfs.append(df)
    plot_data(dfs)

main('/home/downey/PycharmProjects/vecto_analogies/tables/')