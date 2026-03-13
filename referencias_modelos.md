# Referencias para Modelo Comparativo

## Estrategia recomendada

Para la comparacion con un modelo "validado por terceros", conviene usar un modelo
clasico y ampliamente reportado en literatura de prediccion de rendimiento
academico. `RandomForest` es una buena opcion porque:

- funciona bien en datos tabulares;
- suele competir fuertemente contra redes neuronales en datasets pequenos o medianos;
- aparece con frecuencia en trabajos de prediccion de desempeno estudiantil.

## Como reportarlo

En el informe IEEE, la comparacion mas rigurosa es:

1. Ejecutar `RandomForest` localmente sobre el mismo dataset y el mismo split.
2. Citar trabajos previos donde `RandomForest` o ensambles similares fueron usados
   exitosamente en prediccion de desempeno academico.
3. Aclarar que la comparacion numerica directa con literatura externa puede no ser
   estrictamente equivalente si cambian dataset, definicion de clases o protocolo.

## Referencias utiles

- Smart Learning Environments (Springer): trabajos de prediccion de desempeno estudiantil con comparacion de modelos clasicos y metricas de clasificacion.
  Link: https://link.springer.com/article/10.1186/s40561-022-00192-z

- Applied Sciences (MDPI): estudios de prediccion academica donde se comparan ANN, Random Forest y otros modelos supervisados.
  Link: https://www.mdpi.com/2076-3417/12/19/9467

- Education and Information Technologies (Springer): clasificacion de desempeno academico con algoritmos de mineria de datos y aprendizaje supervisado.
  Link: https://link.springer.com/article/10.1007/s10639-024-12619-w

- Caso tecnico basado en el mismo dataset de Kaggle con `GradeClass`.
  Link: https://www.docs.isalos.novamechanics.com/files/Classification/Student%27s%20performance/Students%27%20Performance.pdf

## Redaccion sugerida

"Ademas del modelo MLP requerido y del baseline KNN, se incorporo un modelo
Random Forest como referencia adicional. La eleccion de este algoritmo se baso
en literatura previa de prediccion de desempeno academico, donde los metodos de
ensamble sobre datos tabulares han mostrado resultados competitivos y una buena
capacidad de generalizacion."
