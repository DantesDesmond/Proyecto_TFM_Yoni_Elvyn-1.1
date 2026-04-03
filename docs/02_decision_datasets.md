# 02. Decisión oficial de datasets

## 1. Objetivo de esta decisión
El presente documento define los datasets que se utilizarán en el proyecto para las fases de entrenamiento, validación externa y validación en escenario real. La selección se ha realizado considerando criterios de disponibilidad, idioma, cercanía al problema planteado, facilidad de uso en un entorno local y utilidad metodológica para un Trabajo Fin de Máster de tipo desarrollo de software.

## 2. Criterios de selección
Para la elección de los datasets se han considerado los siguientes criterios:

- disponibilidad o acceso razonable para uso académico;
- presencia de datos en español;
- alineación con tareas de toxicidad o discurso de odio;
- utilidad para clasificación binaria en una primera fase;
- posibilidad de complementar el entrenamiento con validación externa;
- capacidad de apoyar la evaluación de una aplicación local con comentarios reales.

## 3. Decisión de datasets seleccionados

### 3.1 Dataset principal de entrenamiento
Se selecciona como dataset principal de entrenamiento el conjunto `multilingual_toxicity_dataset` en su partición correspondiente al español.

#### Justificación
Este dataset se elige como punto de partida por las siguientes razones:

- permite iniciar rápidamente el entrenamiento del baseline y de un primer modelo transformer;
- presenta una estructura ya preparada para clasificación binaria;
- contiene datos en español;
- facilita la experimentación inicial sin requerir procesos complejos de acceso o transformación;
- resulta adecuado para establecer un primer benchmark interno del proyecto.

#### Rol dentro del proyecto
Su función principal será servir como base inicial para:

- entrenamiento del baseline clásico;
- entrenamiento del primer modelo transformer;
- comparación temprana entre enfoques tradicionales y modelos basados en deep learning.

---

### 3.2 Dataset complementario de entrenamiento
Se selecciona `CLANDESTINO` como dataset complementario de entrenamiento, condicionado a que su descarga, revisión y adaptación sean viables en el tiempo disponible del proyecto.

#### Justificación
Este dataset se incorpora como complemento por las siguientes razones:

- amplía la cobertura de variantes del español;
- permite incorporar mayor diversidad léxica y regional;
- fortalece la robustez del modelo frente a distintas formas de expresión tóxica;
- puede mejorar la generalización del sistema más allá del conjunto inicial balanceado.

#### Rol dentro del proyecto
Su función será:

- complementar el entrenamiento principal;
- enriquecer el vocabulario y las expresiones del modelo;
- apoyar el análisis de variación lingüística y regional en los errores del sistema.

#### Nota metodológica
En caso de que la integración de este dataset genere un coste técnico excesivo o problemas de compatibilidad con las etiquetas definidas para el proyecto, se priorizará la estabilidad experimental utilizando únicamente el dataset principal de entrenamiento.

---

### 3.3 Dataset de validación externa
Se selecciona `DETOXIS` como dataset de validación externa.

#### Justificación
Este recurso se considera adecuado como conjunto de validación externa por las siguientes razones:

- se encuentra disponible en un repositorio público sin requerir permisos adicionales;
- está centrado en la detección de toxicidad en comentarios escritos en español;
- su dominio de aplicación se aproxima al problema del proyecto al trabajar con comentarios publicados en respuesta a noticias;
- permite evaluar tanto la detección binaria de toxicidad como, potencialmente, niveles de toxicidad.

#### Rol dentro del proyecto
Su función principal será:

- evaluar el comportamiento del modelo fuera del dataset principal de entrenamiento;
- medir la capacidad de generalización del sistema ante un dominio diferente;
- fortalecer la evaluación experimental sin depender de procesos de acceso restringido.

#### Nota metodológica
Aunque el dominio de DETOXIS no corresponde a YouTube, su naturaleza de comentarios en español lo convierte en una alternativa adecuada para validación externa dentro del alcance del presente proyecto.
#### Rol dentro del proyecto
Su función principal será:

- evaluar el comportamiento del modelo sobre un conjunto distinto al de entrenamiento;
- identificar caídas de rendimiento por cambio de dominio;
- reforzar la discusión de resultados y limitaciones.

#### Limitación
El acceso a este dataset puede requerir condiciones específicas para investigación. En caso de no disponer del recurso completo durante la fase experimental, se documentará esta limitación y se mantendrá como referencia metodológica dentro del diseño experimental.

---

### 3.4 Dataset propio de validación real
Se define como dataset propio del proyecto una muestra manualmente anotada de comentarios de YouTube obtenidos mediante la API oficial.

#### Justificación
Esta muestra constituye el componente más importante para la validación real del sistema, por las siguientes razones:

- conecta directamente la solución desarrollada con el caso de uso del TFM;
- permite evaluar el comportamiento del modelo en comentarios reales del entorno elegido;
- facilita la obtención de capturas, ejemplos y pruebas visibles para la memoria;
- aporta evidencia empírica propia, más allá de los datasets académicos externos.

#### Rol dentro del proyecto
Su función será:

- validar el comportamiento del modelo en un escenario real;
- alimentar la demostración funcional de la aplicación en Streamlit;
- apoyar el análisis cualitativo de errores;
- servir como evidencia principal en la descripción del piloto local.

#### Tamaño inicial recomendado
Se establece como objetivo inicial una muestra de entre 200 y 500 comentarios anotados manualmente. Si el tiempo disponible lo permite, esta muestra podrá ampliarse hasta 1,000 comentarios.

---

### 3.5 Dataset temático opcional
Se considera `DETESTS-Dis` como dataset opcional de validación temática.

#### Justificación
Este recurso puede aportar valor adicional en escenarios relacionados con estereotipos, xenofobia e implicitud, lo que permite explorar el comportamiento del modelo frente a contenidos más ambiguos o menos explícitos.

#### Rol dentro del proyecto
Su uso será opcional y estará condicionado al tiempo disponible. En caso de utilizarse, servirá para:

- analizar robustez temática;
- complementar la discusión sobre sesgo y límites del sistema;
- enriquecer la interpretación de resultados.

## 4. Estructura experimental derivada de la selección
La estructura experimental inicial quedará organizada de la siguiente forma:

- entrenamiento principal: `multilingual_toxicity_dataset (es)`
- entrenamiento complementario: `CLANDESTINO`
- validación externa: `NewsCom-TOX`
- validación real del caso de uso: muestra propia de YouTube
- validación temática opcional: `DETESTS-Dis`

## 5. Orden de uso recomendado
Se define el siguiente orden operativo para trabajar con los datasets:

1. preparar y limpiar el dataset principal de entrenamiento;
2. entrenar el baseline clásico;
3. entrenar el primer modelo transformer;
4. incorporar el dataset complementario si mejora la cobertura sin comprometer la consistencia;
5. evaluar sobre el dataset de validación externa;
6. evaluar finalmente sobre la muestra propia de YouTube.

## 6. Riesgos identificados
Los principales riesgos asociados a la selección de datasets son los siguientes:

- diferencias de dominio entre datasets académicos y comentarios reales de YouTube;
- desbalance o diferencias de definición entre etiquetas;
- dificultades de acceso a ciertos corpus;
- variación regional del español;
- presencia de sarcasmo, ironía o contexto implícito difícil de capturar.

## 7. Estrategia de mitigación
Para reducir estos riesgos se adoptarán las siguientes medidas:

- iniciar el proyecto con una tarea binaria estable;
- documentar claramente las reglas de anotación;
- mantener una validación externa separada del entrenamiento;
- construir una muestra propia de YouTube;
- analizar errores de forma cualitativa en comentarios ambiguos o conflictivos.

## 8. Conclusión de la decisión
La combinación de un dataset principal accesible, un dataset complementario multirregional, un corpus de validación externa y una muestra propia de YouTube ofrece un equilibrio adecuado entre rapidez de desarrollo, solidez experimental y aplicabilidad real. Esta selección es coherente con el alcance local del proyecto y proporciona una base suficiente para entrenar, evaluar y demostrar una aplicación funcional de detección de toxicidad en español.