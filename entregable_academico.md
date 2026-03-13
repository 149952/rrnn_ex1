# Examen 1
## Borrador académico del entregable

**Título sugerido**
Clasificación del desempeño académico estudiantil mediante un perceptrón multicapa y Random Forest

**Planteamiento**
El problema consiste en predecir la clase de calificación académica de estudiantes de preparatoria a partir de variables demográficas, hábitos de estudio, apoyo parental, actividades extracurriculares y desempeño académico. El conjunto de datos empleado corresponde a 2392 estudiantes y fue obtenido del repositorio Students Performance Dataset de Kaggle. La tarea se formuló como un problema de clasificación multiclase, donde la variable objetivo fue `GradeClass`.

La relevancia del problema radica en que la predicción del desempeño académico puede apoyar estrategias de seguimiento escolar, intervención temprana y análisis de riesgo académico. En un contexto aplicado, este tipo de modelos puede contribuir a detectar estudiantes con trayectorias de bajo rendimiento y, de manera indirecta, apoyar acciones orientadas a la disminución de la deserción escolar.

**Marco teórico y conceptual**
Un perceptrón multicapa es una red neuronal de propagación hacia adelante compuesta por una capa de entrada, varias capas ocultas y una capa de salida. Cada capa oculta transforma la representación de los datos mediante combinaciones lineales y funciones no lineales de activación. En este trabajo, la activación seleccionada fue ReLU, debido a su simplicidad computacional, a su capacidad para reducir el problema del desvanecimiento del gradiente y a su uso extendido en arquitecturas profundas para clasificación.

Las capas `Dropout` introducen una regularización estocástica al desactivar aleatoriamente una fracción de neuronas durante el entrenamiento. Su propósito es reducir el sobreajuste, impedir que la red dependa excesivamente de subconjuntos específicos de neuronas y mejorar la capacidad de generalización del modelo.

La capa de salida utiliza la función `softmax`, la cual transforma los valores de activación finales en probabilidades asociadas a cada clase posible. Dado que el problema es multiclase, la combinación de `softmax` con la función de pérdida `sparse_categorical_crossentropy` es adecuada para el entrenamiento de la red neuronal.

Como modelo comparativo se utilizó `Random Forest`, un método de aprendizaje automático basado en un conjunto de árboles de decisión entrenados sobre subconjuntos de datos y variables. Este enfoque es especialmente útil en datos tabulares, permite modelar relaciones no lineales, es robusto ante ruido y además proporciona una medida de importancia de variables, lo cual resulta valioso para interpretar qué atributos contribuyen más a la predicción.

**Fundamento de las capas del modelo principal**
- Capa de entrada. Su tamaño depende del número de variables resultantes después del preprocesamiento, en particular tras el escalamiento de variables continuas y la codificación de variables categóricas.
- Capa oculta densa. Cada capa densa aprende combinaciones de atributos y patrones intermedios útiles para la clasificación.
- Activación ReLU. Se usa para introducir no linealidad y mejorar la capacidad de aprendizaje del modelo.
- Dropout con tasa 0.2. Se emplea como regularización para mejorar la generalización.
- Capa de salida con `softmax`. Convierte la salida final en probabilidades de pertenencia a cada clase.

**Pipeline realizado**
1. Carga del conjunto de datos en formato CSV.
2. Inspección inicial de estructura, tipos de datos, valores nulos y distribución de clases.
3. Exclusión de `StudentID` por tratarse de un identificador sin valor predictivo.
4. Conservación de `GPA` dentro de los atributos, dado que forma parte del conjunto de datos original utilizado en el escenario final de evaluación.
5. Separación de variables continuas y categóricas.
6. Escalamiento de variables continuas con `StandardScaler`.
7. Codificación de variables categóricas con `OneHotEncoder`.
8. Partición estratificada en entrenamiento, validación y prueba con proporción `70/15/15`.
9. Búsqueda de hiperparámetros para el MLP principal.
10. Búsqueda de hiperparámetros para `Random Forest`.
11. Evaluación final en el conjunto de prueba con métricas y matrices de confusión normalizadas.
12. Exportación de tablas y figuras para su incorporación al reporte y a Overleaf.

**Metodología utilizada**
El dataset contiene 2392 registros y 15 columnas, incluyendo atributos como edad, género, grupo étnico, educación parental, tiempo de estudio semanal, ausencias, tutorías, apoyo parental, participación extracurricular, deportes, música, voluntariado, GPA y la clase objetivo. No se detectaron valores nulos, por lo que no fue necesario un proceso de imputación. El principal tratamiento sobre los datos consistió en la identificación de atributos útiles, la exclusión de la matrícula del estudiante y el preprocesamiento diferenciado de variables continuas y categóricas.

El modelo principal se construyó respetando la arquitectura exigida por el enunciado:
- Capa oculta 1 de 200 unidades con activación ReLU.
- Dropout de 0.2.
- Capa oculta 2 de 200 unidades con activación ReLU.
- Dropout de 0.2.
- Capa oculta 3 de 150 unidades con activación ReLU.
- Dropout de 0.2.
- Capa oculta 4 de 200 unidades con activación ReLU.
- Dropout de 0.2.

La capa de entrada quedó determinada por el número de variables generadas tras el preprocesamiento y la capa de salida por el número de clases presentes en `GradeClass`.

La búsqueda de hiperparámetros del MLP consideró:
- Optimizadores: `Adam`, `RMSprop` y `SGD`.
- Tasas de aprendizaje: `0.0001`, `0.0005` y `0.001`.
- Tamaños de lote: `32`, `64` y `128`.
- Épocas: `50`, `100` y `150`.

Para el segundo modelo se evaluó `Random Forest` con búsqueda de:
- Número de árboles: `200`, `400` y `700`.
- Profundidad máxima: `None`, `10`, `20` y `30`.
- Mínimo de muestras por hoja: `1`, `2` y `4`.
- Esquema de peso por clase: `None`, `balanced` y `balanced_subsample`.

El criterio principal de selección del mejor modelo fue el puntaje F1 macro en validación, complementado con la exactitud de validación. Las métricas finales se reportaron sobre el conjunto de prueba.

**Fundamentación de decisiones de configuración**
- Se utilizó la arquitectura base solicitada por el profesor para el modelo principal, respetando exactamente el número de capas y unidades ocultas establecidas.
- Se aplicó `Dropout(0.2)` en cada bloque oculto, tal como se indicó en la consigna.
- Se mantuvo `GPA` en el escenario final debido a que el objetivo práctico del ejercicio era clasificar la calificación a partir del dataset disponible y este atributo forma parte del conjunto original utilizado en la práctica.
- Se utilizó partición estratificada porque el profesor indicó explícitamente que debía conservarse la proporción de clases en entrenamiento y prueba.
- Se utilizó `Random Forest` como modelo adicional porque es un enfoque ampliamente utilizado en problemas de rendimiento académico y deserción, además de adaptarse bien a datos tabulares y aportar interpretabilidad.

**Atributos incluidos**
Se emplearon como atributos predictivos:
- Edad
- Género
- Grupo étnico
- Educación parental
- Tiempo de estudio semanal
- Ausencias
- Tutorías
- Apoyo parental
- Actividades extracurriculares
- Deportes
- Música
- Voluntariado
- GPA

Se excluyó:
- Matrícula del estudiante (`StudentID`)

**Herramientas agregadas**
- Python 3.12
- JupyterLab
- NumPy
- pandas
- scikit-learn
- TensorFlow / Keras
- matplotlib
- seaborn

**Técnicas extras implementadas**
- Partición estratificada de datos.
- Escalamiento de variables continuas.
- Codificación `one-hot` para variables categóricas en el MLP.
- Búsqueda sistemática de hiperparámetros.
- Detención temprana en el MLP para evitar sobreajuste.
- Exportación de figuras y tablas para documentación y reporte.
- Análisis de importancia de variables con `Random Forest`.

**Descripción del modelo comparativo**
`Random Forest` fue seleccionado como modelo comparativo adicional debido a su buen desempeño reportado en literatura para tareas de predicción de rendimiento académico y deserción escolar. Se trata de un ensamble de árboles de decisión que mejora la estabilidad y la capacidad de generalización mediante muestreo de observaciones y selección aleatoria de variables. Este enfoque fue útil para contrastar el comportamiento del MLP principal frente a un método clásico de aprendizaje automático sobre datos tabulares.

**Resultados principales**
En la evaluación final sobre el conjunto de prueba se obtuvieron los siguientes resultados:
- MLP principal:
  - Exactitud: `0.8607`
  - Precisión macro: `0.8270`
  - Sensibilidad macro: `0.7405`
  - Puntaje F1 macro: `0.7643`
- Random Forest:
  - Exactitud: `0.9331`
  - Precisión macro: `0.9147`
  - Sensibilidad macro: `0.8363`
  - Puntaje F1 macro: `0.8604`

El mejor MLP se obtuvo con:
- Optimizador: `Adam`
- Tasa de aprendizaje: `0.0001`
- Tamaño de lote: `32`
- Épocas solicitadas: `50`

El mejor `Random Forest` se obtuvo con:
- `class_weight = balanced`
- `max_depth = None`
- `min_samples_leaf = 2`
- `n_estimators = 200`

**Interpretación y discusión**
Los resultados muestran que ambos modelos alcanzaron un desempeño alto, consistente con la expectativa expresada por el profesor de que el dataset permitía superar el 80% de exactitud. El MLP principal cumplió con la arquitectura exigida y produjo un desempeño robusto, lo cual valida la pertinencia de la búsqueda de hiperparámetros y del preprocesamiento aplicado.

Sin embargo, `Random Forest` superó claramente al MLP en todas las métricas principales. Esto sugiere que, para este problema particular y para la estructura tabular del conjunto de datos, el modelo de ensamble fue más efectivo al capturar relaciones entre variables. La diferencia fue especialmente visible en el puntaje F1 macro, lo cual indica un mejor equilibrio entre clases.

La importancia de variables calculada por `Random Forest` muestra que `GPA` fue el atributo más influyente, seguido por `Absences` y `StudyTimeWeekly`. Este hallazgo es coherente con el propósito del dataset y con la naturaleza del problema, ya que el rendimiento académico final y el ausentismo escolar son variables estrechamente ligadas a la clase de calificación del estudiante.

Las matrices de confusión muestran que ambos modelos clasifican con alta precisión la clase mayoritaria, pero `Random Forest` mejora de forma clara el comportamiento sobre las clases intermedias y eleva también el rendimiento de la clase minoritaria. Esto fortalece la conclusión de que, en este caso, el segundo modelo no solo cumple la exigencia de comparación, sino que constituye una alternativa más fuerte que la red neuronal base para este conjunto de datos.

**Figuras y tablas recomendadas para el reporte**
- `grafica_distribucion_clases.png`
- `matriz_correlacion.png`
- `boxplots_variables_por_clase.png`
- `arquitectura_mlp_principal.png`
- `comparacion_hiperparametros_mlp.png`
- `comparacion_hiperparametros_random_forest.png`
- `matrices_confusion_comparadas.png`
- `comparacion_metricas_modelos.png`
- `importancia_variables_random_forest.png`
- `comparacion_modelos_final.csv`

**Conclusión sugerida**
El trabajo permitió resolver un problema de clasificación del desempeño académico estudiantil a partir de un conjunto de datos tabular con variables demográficas, académicas y contextuales. El modelo principal, basado en una red neuronal de tipo perceptrón multicapa con la arquitectura exigida por el examen, alcanzó un desempeño alto y metodológicamente consistente. No obstante, el modelo comparativo `Random Forest`, seleccionado con respaldo en literatura y ajustado mediante búsqueda de hiperparámetros, obtuvo un rendimiento superior en exactitud y puntaje F1 macro.

En consecuencia, el estudio no solo cumplió con la implementación del modelo solicitado, sino que además demostró experimentalmente que, para este problema y este conjunto de datos, un método de aprendizaje automático clásico puede superar a la arquitectura neuronal base. Esto aporta valor al análisis, fortalece la discusión técnica del reporte y justifica la inclusión de un segundo modelo sustentado en trabajos previos.
