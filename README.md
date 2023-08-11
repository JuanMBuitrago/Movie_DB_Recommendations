# Proyecto de Análisis Exploratorio de Datos (EDA) y Transformación (ETL) - MLOps Engineer

## Descripción del Problema 
### Contexto
Se obtuvo una base de datos con datos faltantes, datos anidados y sin formatos consistentes. Estos datos eran inutilizables para el análisis sin un procesamiento previo. Por lo tanto, los datos se procesaron utilizando técnicas de ETL y EDA. El proceso ETL extrajo los datos de la base de datos original, los transformó para que estuvieran en un formato consistente y sin errores, y luego los cargó en una nueva base de datos. El proceso EDA analizó los datos utilizando técnicas de visualización de datos y análisis estadístico para identificar patrones y tendencias. Los resultados del análisis exploratorio de los datos se utilizaron para crear funciones básicas con capacidad de API para permitir a los usuarios acceder a los datos de forma segura y eficiente.

El proceso de procesamiento de datos, EDA y creación de funciones básicas se completó con éxito. Los datos se han limpiado y transformado, y ahora están disponibles para ser utilizados para análisis y visualización. Las funciones básicas permiten a los usuarios acceder a los datos de forma segura y eficiente.

El resultado de este proceso es un conjunto de datos limpio, consistente y listo para el análisis. Este conjunto de datos puede utilizarse para una variedad de propósitos, como el desarrollo de modelos de aprendizaje automático, la identificación de tendencias de mercado y la toma de decisiones informadas.

## Contenido
El repositorio está contiene los siguientes archivos:

output.csv: Archivos de datos procesados por el proceso ETL. Usados por la API para la ejecución de funciones.
main.py: Scripts en lenguaje Python, define la funciones para la API, la mayoria de las transformaciones siguen presentes inhibias como comentarios.
Movie_EDA.ipynb: Jupyter notebook  que almacena resultados, gráficos y visualizaciones generados durante el EDA.

##Transformaciones
- Se desanido los campos anidados como "belongs_to_collection" y "production_companies", y se los uní nuevamente con el conjunto de datos para futuras consultas de la API.
- Se rellenaron los valores nulos en los campos "revenue" y "budget" con el número 0.
- Se eliminaron los valores nulos en el campo "release_date".
- Se formatearon las fechas al estilo AAAA-mm-dd y se creó la columna "release_year" para extraer el año de estreno.
- Se creó la columna "return" para calcular el retorno de inversión (revenue / budget). Se establecieron los valores faltantes en 0.
- Se eliminaron las columnas no utilizadas: "video", "imdb_id", "adult", "original_title", "poster_path" y "homepage".
- Se desarrolló una API utilizando el framework FastAPI para acceder a los datos de la empresa. Las consultas propuestas incluyen:
    peliculas_idioma(Idioma: str): Devuelve la cantidad de películas producidas en un idioma específico.
    peliculas_duracion(Pelicula: str): Proporciona la duración y el año de una película dada.
    franquicia(Franquicia: str): Ofrece la cantidad de películas, ganancia total y promedio de una franquicia específica.
    peliculas_pais(Pais: str): Indica la cantidad de películas producidas en un país determinado.
    productoras_exitosas(Productora: str): Muestra los ingresos totales y la cantidad de películas de una productora específica.
    get_director(nombre_director): Muestra el éxito de un director y detalles de sus películas.
- Se desplegó la API utilizando servicios como Render o Railway para que la API fuera accesible desde la web.

## Análisis Exploratorio de Datos (EDA)
Se realizó un análisis exploratorio para comprender las relaciones entre las variables, identificar patrones y posibles anomalías. Se evitaron las librerías automáticas de EDA para aplicar los conceptos y tareas manualmente.

## Sistema de Recomendación
Una vez que los datos fueron accesibles a través de la API y el EDA proporcionó una comprensión sólida, se entrenó un modelo de machine learning para crear un sistema de recomendación de películas. Se utilizó la similitud de puntuación para recomendar películas similares basadas en una película dada.
