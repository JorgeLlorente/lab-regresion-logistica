import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder


import pickle

import sys
sys.path.append("../")

import src.support as sp


# --------------------------------------------- OUTLIERS ---------------------------------------------

def detectar_outliers(lista_columnas, dataframe):

    dicc_indices = {}

    for col in lista_columnas:

        Q1 = np.nanpercentile(dataframe[col], 25)
        Q3 = np.nanpercentile(dataframe[col], 75)

        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outliers_data = dataframe[(dataframe[col] < Q1 - outlier_step) | (dataframe[col] > Q3 + outlier_step)]

        if outliers_data.shape[0] > 0:

            dicc_indices[col] = (list(outliers_data.index))

    return dicc_indices


def opcion(dataframe):

    # Esta función devuelve el número de outliers por el total de combinaciones de nuestro dataframe.

    # Hago el cálculo de outliers del dataframe general

    df_num = dataframe.select_dtypes(include=np.number)
    listacol = df_num.columns
    out = detectar_outliers(listacol, df_num)
    valores_general = list(out.values())
    valores_general = len([indice for sublista in valores_general for indice in sublista])
    print(f"Los outliers sin tocar nada son {valores_general}")

    # Creo mi diccionario para poder iterar sobre las distintas opciones

    df_base = dataframe.select_dtypes(include=object)
    cols = df_base.columns
    dicc_base = {}
    for col in cols:
        dicc_base[col] = df_base[col].unique().tolist()

    # Itero por ellas
    for k, v in dicc_base.items():
        outliers = 0
        count_outliers = 0
        for i in v:
            df1 = dataframe[dataframe[k] == i]
            df_num = df1.select_dtypes(include=np.number)
            cols = df_num.columns
            a = sp.detectar_outliers(cols, df_num)
            valores = list(a.values())
            valores = len([indice for sublista in valores for indice in sublista])
            if valores > outliers:
                outliers = valores
            else:
                pass
            count_outliers +=outliers
        print(f"Los outliers agrupando por {k} son {count_outliers}")


def sustituir_outliers(dict, dataframe):
    for k, v in dict.items():
        mediana = dataframe[k].median()
        for i in v:
            dataframe.loc[i,k] = mediana



# ------------------------------------------- ENCODING -------------------------------------------

def ordinal_encoder(orden, df, columna):
    ordinal = OrdinalEncoder(categories = [orden], dtype = int)
    transformados_oe = ordinal.fit_transform(df[[columna]])
    df[columna] = transformados_oe

    with open(f'datos/encoding{columna}.pkl', 'wb') as s:
        pickle.dump(ordinal, s)
    return df

def label_encoder(df, columnas):
    le = LabelEncoder()
    for col in df[columnas].columns:
        nuevo_nombre = col + "_encoded"
        df[nuevo_nombre] = le.fit_transform(df[col])
    return df

def one_hot_encoder(dff, columnas):

    '''
    función: hace un encoding de tipo one hot encoder
    args:
        - dff: dataframe sobre el que hacemos el encoding
        - columnas: columnas (en formato lista) a las que hacemos el encoding
    '''

    oh = OneHotEncoder()

    transformados = oh.fit_transform(dff[columnas])

    oh_df = pd.DataFrame(transformados.toarray(), columns = oh.get_feature_names_out(), dtype = int)

    dff[oh_df.columns] = oh_df

    dff.drop(columnas, axis = 1, inplace = True)

    return dff


def ordinal_map(df, columna, orden_valores):

    ordinal_dict = {}

    for i, valor in enumerate(orden_valores):
        ordinal_dict[valor] = i

    nuevo_nombre = columna + "_mapeada"

    df[nuevo_nombre] = df[columna].map(ordinal_dict)

    return df


