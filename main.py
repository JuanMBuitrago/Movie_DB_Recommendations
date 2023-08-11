#!/usr/bin/env python
# coding: utf-8

# In[46]:


import os
import pandas as pd
import numpy as np
import ast
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


# In[47]:


app = FastAPI()

# In[48]:


df_peliculas = pd.read_csv('output.csv')


# In[49]:


df_peliculas = df_peliculas.drop_duplicates(subset='id').reset_index(drop=True)
df_peliculas['overview'] = df_peliculas['overview'].fillna('')


# In[50]:


columnas_diccionario = ['belongs_to_collection', 'genres']
columnas_lista = ['production_companies', 'production_countries', 'spoken_languages']
columnas_anidadas = columnas_diccionario + columnas_lista
for col in columnas_anidadas:
    df_peliculas[col].fillna('{}', inplace=True)

for col in columnas_anidadas:
    df_peliculas[col] = df_peliculas[col].apply(lambda x: ast.literal_eval(x))


# In[51]:


df_peliculas['release_date'].fillna('')
fechas_erradas = []
for idx, fecha in enumerate(df_peliculas['release_date']):
    try:
        df_peliculas.at[idx, 'release_date'] = pd.to_datetime(fecha, format='%Y-%m-%d')
    except Exception as e:
        print(f"Error in cell {idx}: {fecha}. Error message: {e}")
        fechas_erradas.append(idx)
df_peliculas.drop(index=fechas_erradas, inplace=True)
df_peliculas['release_year'] = pd.to_datetime(df_peliculas['release_date']).dt.year


# In[52]:


columnas_monetarias = ['revenue', 'budget']

for col in columnas_monetarias:
    df_peliculas[col] = df_peliculas[col].fillna(0).apply(pd.to_numeric)
df_peliculas['return'] = np.where(df_peliculas['budget'] != 0, df_peliculas['revenue'] / df_peliculas['budget'], np.nan)


# In[53]:


columnas_innecesarias = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']
#df_peliculas.drop(columnas_innecesarias, axis=1, inplace=True)


# In[54]:


@app.get('/peliculas_idioma/{idioma}')
def peliculas_por_idioma(idioma: str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''
    idioma = idioma.lower()
    total_idiomas = df_peliculas['original_language'] == idioma
    cantidad = len(df_peliculas['original_language'][total_idiomas])
    if any(total_idiomas):
        return {'idioma': idioma, 'cantidad': cantidad}
    else:
        return (f"Idioma {idioma} no fue encontrada en la base de datos.")


# In[55]:


#@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    '''Ingresas la pelicula, retornando la duracion y el año'''
    pelicula_l = pelicula.lower()
    fila = df_peliculas['title'].str.lower() == pelicula_l
    if any(fila):
        duracion, anno = df_peliculas.loc[fila, ['runtime', 'release_year']].iloc[0]
        return {'pelicula':pelicula.title(), 'duracion':duracion, 'anio':anno}
    else:
        return (f"{pelicula.title()} no fue encontrada en la base de datos.")


# In[72]:


df_peliculas['franquicia'] = df_peliculas['belongs_to_collection'].apply(lambda x: x.get('name', None))
df_peliculas['franquicia'] = df_peliculas['franquicia'].apply(lambda x: x.lower() if x is not None else None)
franquicia_df = df_peliculas.dropna(subset=['belongs_to_collection'])
franquicia_df = franquicia_df[franquicia_df['belongs_to_collection'].apply(lambda x: len(x) > 0)]
app.get('/franquicia/{franquicia}')
def franquicia( franquicia: str ):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    franquicia = franquicia.lower()
    num_franquicia = len(franquicia_df[franquicia_df['franquicia'] == franquicia])
    gan_franquicia = franquicia_df['revenue'][franquicia_df['franquicia'] == franquicia].sum()
    if num_franquicia > 0:
        gan_promedio = gan_franquicia / num_franquicia
        return {'franquicia':franquicia.title(), 'cantidad':num_franquicia, 'ganancia_total':gan_franquicia, 'ganancia_promedio':gan_promedio}
    else:
        return (f"Franquicia {franquicia.title()} no fue encontrada en la base de datos.")


# In[74]:


def obtener_nombres(columna_extraer):
  lista_nombres = []
  for coleccion in columna_extraer:
    lista_nombres.append(coleccion['name'].lower())
  return lista_nombres


# In[83]:


#  Se ingresa un país (como están escritos en el dataset, no hay que traducirlos!), retornando la cantidad de peliculas producidas en el mismo.
#                     Ejemplo de retorno: Se produjeron X películas en el país X

df_peliculas['paises'] = df_peliculas['production_countries'].apply(obtener_nombres)

@app.get('/peliculas_pais/{pais}')
def peliculas_pais( pais: str ):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    pais = pais.lower()
    respuesta = len(df_peliculas[df_peliculas['paises'].apply(lambda x: pais in x)])
    return {'pais':pais.title(), 'cantidad':respuesta}  


# In[88]:


df_peliculas['productoras'] = df_peliculas['production_companies'].apply(obtener_nombres)
df_peliculas['productoras'] = df_peliculas['productoras'].apply(lambda x: [i.lower() for i in x])
@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas( productora: str ):
    '''Ingresas la productora, entregandote el revunue total y la cantidad de peliculas que realizo '''
    productora = productora.lower()
    revenue_productora = df_peliculas['revenue'][df_peliculas['productoras'].apply(lambda x: productora in x)].sum()
    peliculas_productora = len(df_peliculas[df_peliculas['productoras'].apply(lambda x: productora in x)])
    return {'productora':productora.title(), 'revenue_total': revenue_productora,'cantidad':peliculas_productora}


# In[89]:


# df_creditos = pd.read_csv('credits.csv')
# df_creditos['equipo_tecnico'] = df_creditos['crew'].apply(lambda x: ast.literal_eval(x))
# directores = []
# for idx, pelicula in enumerate(df_creditos['equipo_tecnico']):
#   dir_pelicula = []
#   for miembro in pelicula:
#     if miembro['job'] == 'Director':
#       dir_pelicula.append(miembro['name'].lower())
#   directores.append(dir_pelicula)
# df_creditos['directores'] = directores
# df_peliculas['id'] = df_peliculas['id'].astype(int)
# df_peliculas = df_peliculas.merge(df_creditos[['id', 'directores']], on='id', how='left')


# In[131]:

df_peliculas['directores'] = df_peliculas['directores'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df_peliculas['directores'] = df_peliculas['directores'].apply(lambda x: ast.literal_eval(x.lower()) if isinstance(x, str) else x)
def director_presente(pelicula, nombre_director: str):
    if isinstance(pelicula, list):
        return nombre_director in pelicula
    return False

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    nombre_director = nombre_director.lower()
    df_director = df_peliculas[df_peliculas['directores'].apply(lambda x: director_presente(x, nombre_director))]
    num_peliculas = len(df_director)
    if num_peliculas == 0:
        return {'message': 'Director not found'}
    ingresos_pelicula = df_director['revenue']
    budget_pelicula = df_director['budget']
    retorno_pelicula = ingresos_pelicula / budget_pelicula  # Corrected
    retorno_total_director = ingresos_pelicula.sum() / budget_pelicula.sum()  # Corrected
    return {
        'director': nombre_director.title(),
        'retorno_total_director': retorno_total_director,
        'peliculas': df_director['title'].tolist(),
        'anio': df_director['release_year'].tolist(),
        'retorno_pelicula': retorno_pelicula.tolist(),
        'budget_pelicula': budget_pelicula.tolist(),
        'revenue_pelicula': ingresos_pelicula.tolist()
    }

# In[133]:




# In[143]:


df_peliculas['title'] = df_peliculas['title'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df_peliculas['popularity'] = pd.to_numeric(df_peliculas['popularity'], errors='coerce')
peliculas_populares = df_peliculas[df_peliculas['popularity'] > 10]
peliculas_populares = peliculas_populares.reset_index(drop=True)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(peliculas_populares['overview'])
similitudes_coseno = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[144]:


@app.get('/get_movie_recommendations/{titulo}')
def get_movie_recommendations(titulo: str, limit: int = 5):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    titulo = titulo.lower()
    try:
        indice_pelicula = peliculas_populares[peliculas_populares['title'] == titulo].index[0]
    except:
        return {f'Pelicula {titulo.title()} no encontrada en la base de datos'}
    peliculas_similares = list(enumerate(similitudes_coseno[indice_pelicula]))
    peliculas_similares = sorted(peliculas_similares, key=lambda x: x[1], reverse=True)
    peliculas_similares = peliculas_similares[1:limit + 1]
    recomendaciones = []
    for idx, puntuacion in peliculas_similares:
        recomendaciones.append(df_peliculas['title'].iloc[idx].title())  # Convert to title case 
    return {'lista recomendada': recomendaciones}

