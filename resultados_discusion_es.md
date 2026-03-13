# Resultados y Discusión

Se evaluaron cinco enfoques sobre el mismo conjunto de datos y bajo un esquema de partición estratificada `70/15/15`: el perceptrón multicapa (MLP) base exigido por el enunciado, un modelo de K vecinos más cercanos (KNN), una variante del MLP con ajustes de entrenamiento (`pesos por clase`, `normalización por lotes` y ajuste fino de hiperparámetros), una variante del MLP con balanceo mediante `SMOTE`, y un `Random Forest` como modelo adicional respaldado por literatura para datos tabulares. Todas las comparaciones se realizaron excluyendo `GPA` y `StudentID` de las variables de entrada, con el fin de evitar fuga de información y asegurar una evaluación realista del problema de clasificación.

El análisis visual del conjunto de datos mostró una distribución de clases claramente desbalanceada, con predominio de la clase `4` y muy pocos ejemplos de la clase `0`. Esta condición afecta directamente la dificultad del problema, ya que un modelo puede obtener una exactitud aceptable favoreciendo la clase mayoritaria y aun así fallar en las clases minoritarias. La matriz de correlación y las gráficas exploratorias indicaron además que `Absences` presenta la señal más fuerte respecto a `GradeClass`, mientras que otras variables tienen una contribución menor. Esto sugiere que la capacidad predictiva sin `GPA` es limitada y que el problema, por naturaleza, es más difícil de lo que aparentaría si se incluyera dicha variable.

En las comparaciones iniciales, el MLP base obtuvo el mejor equilibrio entre cumplimiento del enunciado y rendimiento global, superando al modelo KNN. KNN funcionó como un modelo base de comparación simple, pero mostró menor capacidad para separar adecuadamente las clases, especialmente las minoritarias. El MLP ajustado con pesos por clase y normalización por lotes no logró mejorar el comportamiento en el conjunto de prueba. Aunque este enfoque elevó algunas métricas en validación y redistribuyó el error entre clases, su capacidad de generalización final fue inferior a la del MLP base. Este resultado confirma que introducir técnicas de balanceo y normalización no garantiza una mejora real cuando el conjunto de datos es pequeño, desbalanceado y con señal predictiva limitada.

La estrategia con `SMOTE` resultó más interesante. Aunque no necesariamente maximizó la exactitud, sí mostró capacidad para mejorar el puntaje F1 macro, lo cual implica un tratamiento más equilibrado entre clases. Este comportamiento es relevante porque el objetivo no debe interpretarse únicamente en términos de exactitud global, sino también en la capacidad del modelo para reconocer clases poco representadas. Desde el punto de vista del reporte, esta variante puede presentarse como una mejora orientada al balance de desempeño, incluso si el MLP base sigue siendo preferible cuando se prioriza la exactitud.

El `Random Forest` se incorporó como modelo comparativo adicional por tratarse de un algoritmo ampliamente validado en la literatura de predicción de desempeño académico y, además, especialmente adecuado para datos tabulares. Su inclusión es metodológicamente valiosa porque permite contrastar el MLP con un enfoque de ensamble clásico. Las gráficas de importancia de variables derivadas de este modelo aportan además un elemento interpretativo importante: permiten identificar qué atributos tienen mayor peso en la decisión, reforzando la discusión sobre la relevancia de `Absences` y de otras variables académicas o contextuales.

Las curvas de entrenamiento permiten observar diferencias en convergencia y estabilidad entre las variantes del MLP. El modelo base presentó una evolución relativamente estable, mientras que las variantes ajustadas mostraron comportamientos más sensibles a la configuración de entrenamiento. Esto apoya la idea de que, en este problema particular, una arquitectura más compleja o un esquema de entrenamiento más agresivo no necesariamente conduce a una mejor generalización. Por otra parte, las matrices de confusión normalizadas facilitan identificar en qué clases concentra el error cada modelo y sirven para argumentar que las clases minoritarias siguen siendo el principal desafío del problema.

En síntesis, los resultados indican que el MLP base constituye una solución sólida y consistente para cumplir con el enunciado, mientras que `SMOTE` representa la estrategia adicional más prometedora cuando se desea mejorar el equilibrio entre clases. KNN queda como modelo base de comparación simple y `Random Forest` como modelo base de comparación robusto respaldado por literatura. En términos de discusión, la principal conclusión es que el límite del rendimiento no parece estar únicamente en la arquitectura, sino en la estructura del conjunto de datos: fuerte desbalance, baja señal en varias variables y dependencia marcada de unos pocos atributos relevantes. Por ello, cualquier mejora futura probablemente dependerá más de estrategias de manejo de clases, selección de variables e ingeniería de características que de aumentar la complejidad de la red neuronal.

## Convención terminológica sugerida

- `accuracy`: **exactitud**
- `precision`: **precisión**
- `recall`: **sensibilidad**
- `f1-score`: **puntaje F1**
- `macro average`: **promedio macro**
- `weighted average`: **promedio ponderado**
- `confusion matrix`: **matriz de confusión**
- `normalized confusion matrix`: **matriz de confusión normalizada**
- `training`: **entrenamiento**
- `validation`: **validación**
- `test`: **prueba**
- `batch size`: **tamaño de lote**
- `epochs`: **épocas**
- `learning rate`: **tasa de aprendizaje**
- `class weight`: **pesos por clase**
- `feature importance`: **importancia de variables**
- `hyperparameter tuning`: **búsqueda de hiperparámetros**
- `overfitting`: **sobreajuste**
- `underfitting`: **subajuste**
