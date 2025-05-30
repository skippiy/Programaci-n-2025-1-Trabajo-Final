# Programación-2025-1-Trabajo-Final
# Air Quality Prediction Using Machine Learning

## Descripción General del Proyecto

Este proyecto presenta una canalización completa de ciencia de datos para predecir concentraciones de contaminantes del aire utilizando algoritmos de aprendizaje automático. El conjunto de datos utilizado es *AirQualityUCI*, uno de los datasets públicos más completos basados en sensores para el monitoreo de la calidad del aire urbano. Incluye 9358 registros horarios provenientes de una matriz de 5 sensores químicos de óxidos metálicos integrados en un dispositivo multisensor desplegado en campo, recolectados entre marzo de 2004 y febrero de 2005 en una ciudad italiana contaminada a nivel de calle.

Como estudiante de Ingeniería Mecatrónica, elegí este conjunto de datos porque combina dos áreas que me apasionan: **sensado ambiental** y **aprendizaje automático**. El procesamiento de señales de sensores, la confiabilidad, el manejo de la deriva y la modelación de fenómenos del mundo real son desafíos centrales tanto en contextos académicos como de la mecatrónica. Este proyecto me permitió explorar todos ellos de una manera significativa y basada en datos.


---

## Por Qué Este Repositorio es Importante

Este repositorio representa un flujo de trabajo completo de aprendizaje automático que demuestra:

- Carga, limpieza y preprocesamiento de datos con deriva de sensores y anomalías del mundo real.
- Análisis exploratorio de datos (EDA) para identificar tendencias y valores atípicos.
- Selección de modelos, ajuste de hiperparámetros y evaluación del rendimiento.
- Curvas de aprendizaje para entender la capacidad de generalización del modelo.
- Análisis comparativo entre dos modelos utilizando validación cruzada y correlación de predicciones.

Esta implementación de extremo a extremo refleja no solo experiencia en aprendizaje automático, sino también una mentalidad de ingeniería orientada a la confiabilidad de señales, la robustez en condiciones ruidosas y el pensamiento a nivel de sistemas.


---

## Descripción del Conjunto de Datos

El conjunto de datos contiene:

- **9358 instancias** de respuestas promediadas por hora de 5 sensores de óxidos metálicos (CO, NMHC, Benceno, NOx, NO₂).
- **Ventana temporal:** Marzo de 2004 a febrero de 2005.
- **Lugar de muestreo:** Zona urbana contaminada a nivel de calle en Italia.
- **Datos de referencia:** Proporcionados por un analizador químico certificado y co-localizado.
- **Desafíos presentes en los sensores:** Sensibilidad cruzada, deriva de concepto y deriva del sensor (como se describe en De Vito et al., *Sensors and Actuators B*, Vol. 129, No. 2, 2008).
- **Datos faltantes:** Marcados con el valor `-200`.

**Nota:** El conjunto de datos está destinado exclusivamente a **fines de investigación**. El uso comercial está estrictamente prohibido.

---

##  Estructura del repositorio

```text
.
├── 01-exploración.py       
├── 02-preprocesado.py      
├── 03-modelo1.py           
├── 04-modelo2.py           
├── 05-comparación.py       
├── dataset.csv          
├── 07-informe.pdf          
└── README.md               
````
## Herramientas y recursos
- Python 3.10+
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn for modeling, validation, and evaluation
- Visual Studio Code


## Descripción de los Archivos del Repositorio

- **README.md**  
  Archivo Markdown con la descripción breve del proyecto.

- **01-exploración.ipynb**  
  Muestra cómo se carga el dataset y se realiza el análisis exploratorio de los datos.  
  Esta exploración no tiene que ser exhaustiva, solo debe mostrar que eres capaz de cargar los datos e inspeccionarlos.

- **02-preprocesado.ipynb**  
  Muestra cómo se carga el archivo de datos y se realizan las operaciones de limpieza y preprocesado necesarias.

- **03-modelo1.ipynb** y **04-modelo2.ipynb**  
  Muestran, respectivamente, las dos soluciones planteadas y la evaluación de los modelos (curvas de aprendizaje, matrices de confusión, etc.).

- **05-comparación.ipynb**  
  Muestra la estrategia de comparación entre los dos modelos implementados, usando análisis de concordancia.

- **06-dataset.csv**  
  Archivo con los datos utilizados.

- **07-informe.pdf**  
  Archivo que resume toda la estrategia de solución planteada, los resultados obtenidos y las conclusiones alcanzadas.

## Citas
``` text
De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008). On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario. Sensors and Actuators B: Chemical, 129(2), 750-757.
```
