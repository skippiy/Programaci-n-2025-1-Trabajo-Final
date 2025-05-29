# Programación-2025-1-Trabajo-Final
# Air Quality Prediction Using Machine Learning

##  Project Overview

This project presents a complete data science pipeline to predict air pollutant concentrations using machine learning algorithms. The dataset used is *AirQualityUCI*, one of the most comprehensive publicly available sensor-based datasets for urban air quality monitoring. It includes 9358 hourly records from an array of 5 metal oxide chemical sensors embedded in a field-deployed multisensor device, collected between March 2004 and February 2005 in a polluted Italian city at road level.

As a Mechatronics Engineering student, I chose this dataset because it combines two areas I'm passionate about: **environmental sensing** and **machine learning**. Sensor signal processing, reliability, drift handling, and modeling of real-world phenomena are central challenges in both academic and industrial contexts of mechatronics. This project allowed me to explore all of them in a meaningful, data-driven way.

---

##  Why This Repository Matters

This repository is a full-stack machine learning workflow that demonstrates:

- Data loading, cleaning, and pre-processing with real-world sensor drift and anomalies.
- Exploratory data analysis (EDA) to identify trends and outliers.
- Model selection, hyperparameter tuning, and performance evaluation.
- Learning curves to understand model generalization.
- Comparative analysis between two models using cross-validation and correlation of predictions.

This end-to-end implementation reflects not only machine learning expertise, but also an engineer’s mindset toward signal reliability, robustness under noisy conditions, and system-level thinking.

---

##  Dataset Description

The dataset contains:

- **9358 instances** of hourly-averaged responses from 5 metal oxide sensors (CO, NMHC, Benzene, NOx, NO₂).
- **Time window:** March 2004 to February 2005.
- **Sampling location:** Polluted urban area at road level in Italy.
- **Ground truth data:** Provided by a certified co-located chemical analyzer.
- **Sensor challenges present:** Cross-sensitivity, concept drift, and sensor drift (as described in De Vito et al., *Sensors and Actuators B*, Vol. 129, No. 2, 2008).
- **Missing data:** Tagged with value `-200`.

 **Note:** The dataset is intended exclusively for **research purposes**. Commercial use is strictly prohibited.

---

##  Repository Structure

```text
.
├── 01-exploración.py       # Exploratory data analysis
├── 02-preprocesado.py      # Data cleaning and preprocessing
├── 03-modelo1.py           # Random Forest Regressor model
├── 04-modelo2.py           # K-Nearest Neighbors Regressor model
├── 05-comparación.py       # Model comparison using metrics and correlation
├── 06-dataset.csv          # Final dataset used in modeling
├── 07-informe.pdf          # Project report with methodology, results, and conclusions
└── README.md               # Project documentation (this file)
````
## Tools and Technologies
- Python 3.10+
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn for modeling, validation, and evaluation
- Visual Studio Code

## Citation
``` text
De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008). On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario. Sensors and Actuators B: Chemical, 129(2), 750-757.
```
