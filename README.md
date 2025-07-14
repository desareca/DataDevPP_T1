# Predictor de Propinas para Taxis de NYC

## Descripción General del Proyecto

Este proyecto implementa una solución de machine learning para predecir propinas altas (>20% del costo del viaje) en los viajes de taxi de Nueva York. El modelo se entrena con datos de enero de 2020 y se evalúa en los meses siguientes para analizar su degradación de rendimiento a lo largo del tiempo.

### Características Principales

- **Modelo de Machine Learning**: Clasificador Random Forest para predicción binaria de propinas
- **Evaluación Temporal**: Evaluación sistemática mes a mes del modelo
- **Análisis de Data Drift**: Comparación estadística de distribuciones entre meses
- **Pipeline Automatizado**: Estructura de código modular para análisis reproducible

## Estructura Principal del Proyecto

```
├── data/               # Archivos de datos (no versionados en git)
├── docs/               # Documentación
├── models/             # Modelos entrenados
├── notebooks/          # Notebooks de Jupyter
├── references/         # Materiales de referencia
├── reports/            # Reportes generados e imagenes
├── src/                # Código fuente
│   ├── data/           # Carga y preprocesamiento de datos
│   ├── features/       # Ingeniería de características
│   ├── modeling/       # Entrenamiento y predicción
│   └── visualization/  # Utilidades de visualización
├── requirements.txt    # Dependencias del proyecto
└── README.md
```

## Comenzando

### Prerequisitos

- Python 3.11
- Git
- Herramienta de entorno virtual (ej: venv, conda)

### Instalación

1. Clonar el repositorio
```bash
git clone https://github.com/desareca/DataDevPP_T1
cd DataDevPP_T1
```

2. Crear y activar un entorno virtual
```bash
# Usando venv
python -m venv venv
# En Windows
venv\Scripts\activate
# En Unix o MacOS
source venv/bin/activate
```

3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### Configuración de Datos

El proyecto utiliza datos de viajes de la Comisión de Taxis y Limosinas de NYC (TLC) del 2020. Sigue estos pasos para configurar los datos:

1. Crea un directorio `data/raw`, `data/processed`, `data/interim` y `data/external` en tu carpeta del proyecto
2. Utiliza los datos de viajes de Yellow Taxi del 2020 (formato Parquet) desde el [sitio web de NYC TLC](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
3. Descarga y guarda los archivos descargados en el directorio `data/raw`

### Ejecutando el Análisis

1. Navega al directorio `notebooks`
2. Abre y ejecuta el notebook de Jupyter `01_csaquel_nyc_taxi_model.ipynb`


## Pipeline del Modelo

1. **Preprocesamiento de Datos** (`src.data.dataset`)
   - Carga archivos Parquet
   - Limpia y estandariza datos
   - Implementa pipeline completo desde carga hasta generación de features

2. **Ingeniería de Características** (`src.features.build_features`)
   - Crea características relevantes para la predicción de propinas
   - Maneja variables categóricas
   - Implementa selección de características

3. **Entrenamiento del Modelo** (`src.modeling.train`)
   - Entrena el Clasificador Random Forest
   - Guarda artefactos del modelo
   - Implementa validación cruzada

4. **Evaluación** (`src.modeling.predict`)
   - Realiza evaluaciones mensuales
   - Calcula métricas de rendimiento
   - Analiza la degradación del modelo

## Estado del Proyecto y Mejoras

Basado en el análisis del proyecto, aquí hay algunas mejoras sugeridas:

1. **Documentación**
   - Agregar docstrings a los módulos Python
   - Crear documentación API usando mkdocs
   - Agregar notebooks de ejemplo para casos específicos

2. **Pruebas**
   - Implementar pruebas unitarias para funciones principales
   - Agregar pruebas de integración para el pipeline
   - Configurar integración continua

3. **Configuración**
   - Crear archivo de configuración para parámetros del modelo
   - Agregar variables de entorno para rutas y configuraciones
   - Implementar logging

4. **Despliegue**
   - Crear API para servir el modelo
   - Containerizar la aplicación
   - Configurar monitoreo del rendimiento del modelo




