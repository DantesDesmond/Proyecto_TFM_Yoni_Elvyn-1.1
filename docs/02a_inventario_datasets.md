# 02a. Inventario de datasets

## 1. Objetivo del inventario
El presente documento tiene como finalidad registrar de forma operativa los datasets considerados para el proyecto, su estado de disponibilidad, su uso previsto dentro del experimento y su ubicación local dentro de la estructura de carpetas. Este inventario permite mantener trazabilidad sobre las fuentes de datos utilizadas y facilitar la reproducibilidad del desarrollo experimental.

## 2. Estado general
A continuación, se presenta el inventario inicial de datasets seleccionados para el proyecto.

| Dataset | Rol en el proyecto | Estado | Descargado | Ruta local | Formato esperado | Observaciones |
|---|---|---|---|---|---|---|
| multilingual_toxicity_dataset (es) | Entrenamiento principal | Seleccionado | No | `data/external/multilingual_toxicity_dataset_es/` | CSV o parquet | Será el primer dataset a preparar para baseline y transformer. |
| CLANDESTINO | Entrenamiento complementario | Seleccionado condicional | No | `data/external/clandestino/` | CSV, JSON o formato del repositorio | Se incorporará si la descarga y adaptación resultan viables sin afectar el cronograma. |
| DETOXIS | Validación externa | Seleccionado | No | `data/external/detoxis/` | CSV o archivos del repositorio | Dataset público y alineado con detección de toxicidad en comentarios en español. |
| Muestra propia de YouTube | Validación real del caso de uso | Seleccionado | No aplica todavía | `data/raw/youtube_comments/` y `data/processed/youtube_sample_labeled/` | CSV | Se construirá a partir de comentarios descargados mediante la API de YouTube y anotados manualmente por el equipo. |
| DETESTS-Dis | Validación temática opcional | Opcional | No | `data/external/detests_dis/` | CSV o formato académico | Solo se utilizará si el tiempo permite una validación adicional sobre estereotipos e implicitud. |

## 3. Criterios de seguimiento
Para cada dataset se deberá actualizar la siguiente información a medida que avance el proyecto:

- fecha de descarga o recepción;
- fuente exacta de obtención;
- versión del recurso, si aplica;
- cantidad de registros disponibles;
- etiquetas utilizadas;
- transformaciones aplicadas antes del entrenamiento o validación;
- incidencias encontradas durante la preparación.

## 4. Registro detallado por dataset

### 4.1 multilingual_toxicity_dataset (es)
- **Rol**: entrenamiento principal.
- **Prioridad**: alta.
- **Acción pendiente**: descargar, inspeccionar columnas, verificar etiquetas y preparar subconjunto oficial de trabajo.
- **Resultado esperado**: dataset base para entrenar el baseline clásico y el primer modelo transformer.

### 4.2 CLANDESTINO
- **Rol**: entrenamiento complementario.
- **Prioridad**: alta, condicionada a facilidad de integración.
- **Acción pendiente**: revisar estructura del repositorio, identificar archivos utilizables y verificar compatibilidad con la tarea binaria definida.
- **Resultado esperado**: ampliación de cobertura léxica y regional del español.

### 4.3 DETOXIS
- **Rol**: validación externa.
- **Prioridad**: alta.
- **Acción pendiente**: descargar el repositorio, inspeccionar la carpeta de datos y verificar compatibilidad de etiquetas con la tarea binaria definida.
- **Resultado esperado**: evaluación del modelo sobre un conjunto distinto al del entrenamiento principal, sin depender de solicitudes externas.

### 4.4 Muestra propia de YouTube
- **Rol**: validación real del proyecto.
- **Prioridad**: alta.
- **Acción pendiente**: descargar comentarios desde videos seleccionados, construir lote inicial y etiquetar muestra manual.
- **Resultado esperado**: evidencia empírica propia para pruebas visibles, capturas de la aplicación y análisis cualitativo de errores.

### 4.5 DETESTS-Dis
- **Rol**: validación temática opcional.
- **Prioridad**: media.
- **Acción pendiente**: evaluar disponibilidad y compatibilidad de etiquetas con el problema definido.
- **Resultado esperado**: prueba complementaria sobre mensajes implícitos, xenofobia o estereotipos.

## 5. Convención de estados
Se utilizarán los siguientes estados para actualizar este inventario:

- **Seleccionado**: dataset aprobado para uso dentro del diseño experimental.
- **Seleccionado condicional**: dataset aprobado, pero sujeto a validación técnica o de tiempo.
- **Opcional**: dataset no obligatorio, considerado como ampliación metodológica.
- **Descargado**: recurso obtenido y almacenado localmente.
- **Preparado**: recurso limpiado y adaptado para uso experimental.
- **Utilizado**: dataset efectivamente empleado en entrenamiento o validación.

## 6. Próximas acciones inmediatas
Las siguientes acciones deben ejecutarse en el orden indicado:

1. descargar y registrar `multilingual_toxicity_dataset (es)`;
2. revisar e intentar integrar `CLANDESTINO`;
3. confirmar acceso a `NewsCom-TOX`;
4. preparar la estructura para almacenar la muestra propia de YouTube;
5. actualizar este inventario con fechas, rutas definitivas y observaciones técnicas.

## 7. Criterio de cierre
Este inventario se considerará correctamente inicializado cuando todos los datasets seleccionados tengan una ruta local definida, un estado asignado y una observación clara sobre su función dentro del proyecto.