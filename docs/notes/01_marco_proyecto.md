# 01. Marco del proyecto

## 1. Título provisional del TFM
Diseño e implementación de una aplicación local en Streamlit para la detección automática de toxicidad y discurso de odio en redes sociales mediante procesamiento de lenguaje natural: estudio de caso con comentarios de YouTube.

## 2. Tipo de trabajo
Tipo 2. Desarrollo de software.

## 3. Problema que se aborda
El crecimiento del contenido generado por usuarios en plataformas digitales ha incrementado la presencia de comentarios ofensivos, tóxicos y potencialmente asociados al discurso de odio. En entornos como YouTube, el volumen y la velocidad de publicación de comentarios dificultan la moderación manual, lo que limita la detección oportuna de contenido nocivo y reduce la capacidad de análisis sistemático de este fenómeno.

## 4. Justificación del proyecto
La detección automática de toxicidad y discurso de odio mediante técnicas de procesamiento de lenguaje natural constituye una alternativa viable para apoyar la moderación y el análisis de contenido en español. Sin embargo, persisten limitaciones importantes relacionadas con la disponibilidad de recursos lingüísticos, la variación del español, la dificultad de interpretar contexto, sarcasmo e ironía, y la escasez de soluciones locales demostrables centradas en comentarios de redes sociales. En este contexto, se plantea el desarrollo de una aplicación local que permita capturar, procesar, clasificar y visualizar comentarios, utilizando YouTube como caso de estudio inicial y dejando abierta la posibilidad de replicación en otras plataformas.

## 5. Alcance del proyecto
El presente proyecto contempla el diseño e implementación de una aplicación local desarrollada en Streamlit para la detección automática de toxicidad y discurso de odio en comentarios escritos en español. La solución incluirá un módulo de ingesta de comentarios desde la API de YouTube, un flujo de preprocesamiento textual, uno o más modelos de clasificación de texto y una interfaz para visualizar resultados y realizar inferencias sobre comentarios individuales o conjuntos de comentarios en formato tabular.

La solución se desarrollará como un piloto experimental local, sin pretensión de despliegue empresarial completo en producción. YouTube se utilizará como fuente principal de datos y como escenario de validación, aunque el enfoque metodológico se diseñará de forma que pueda ser reutilizado posteriormente en otras redes sociales.

## 6. Fuera de alcance
Quedan fuera del alcance del presente proyecto los siguientes elementos:

- despliegue productivo empresarial en la nube como componente obligatorio;
- moderación automática en tiempo real con acción directa sobre plataformas;
- integración con múltiples redes sociales dentro de la primera versión funcional;
- entrenamiento de modelos multimodales con audio o video;
- sistemas completos de autenticación corporativa, observabilidad avanzada o MLOps empresarial.

## 7. Unidad de análisis
La unidad de análisis será el comentario textual individual en español, obtenido desde YouTube o desde datasets externos utilizados para entrenamiento y validación.

## 8. Fuente de datos
Se emplearán dos tipos de fuentes de datos:

1. datasets externos en español para entrenamiento y validación experimental;
2. una muestra propia de comentarios de YouTube etiquetada por el equipo para validación en un escenario real de uso.

## 9. Tarea de aprendizaje seleccionada
Se trabajará inicialmente con una tarea de clasificación binaria:

- clase 0: no tóxico
- clase 1: tóxico

En una fase posterior, y solo si el rendimiento y el tiempo lo permiten, podrá explorarse una extensión a clasificación de tres clases:

- no tóxico
- tóxico
- discurso de odio

## 10. Métrica principal de evaluación
La métrica principal de evaluación será el Macro-F1, por su utilidad en escenarios con posible desbalance de clases y por su uso frecuente en la literatura reciente sobre detección de toxicidad y discurso de odio en español.

## 11. Métricas secundarias
Además del Macro-F1, se analizarán las siguientes métricas:

- precisión;
- recall;
- F1 por clase;
- matriz de confusión;
- accuracy como métrica complementaria no principal.

## 12. Meta mínima de éxito
Se considerará satisfactorio alcanzar un Macro-F1 igual o superior a 0.75 en la tarea binaria sobre el conjunto de evaluación definido para el proyecto.

## 13. Meta fuerte
Se considerará un resultado fuerte alcanzar un Macro-F1 igual o superior a 0.80 en la tarea binaria y superar el rendimiento del baseline clásico implementado bajo el mismo protocolo experimental.

## 14. Hipótesis de trabajo
La combinación de técnicas de procesamiento de lenguaje natural en español, modelos de clasificación de texto y una aplicación local desarrollada en Streamlit permitirá construir una solución funcional capaz de detectar automáticamente comentarios tóxicos con un rendimiento competitivo respecto a la literatura reciente y superior al baseline clásico definido para el proyecto.

## 15. Entregable principal del proyecto
El entregable principal será una aplicación local funcional en Streamlit que permita:

- cargar comentarios individuales o en lote;
- ejecutar inferencia automática;
- mostrar etiqueta y probabilidad estimada;
- visualizar resultados agregados;
- exportar resultados para análisis posterior.

## 16. Criterio de cierre del punto 1
El punto 1 se considerará completado cuando queden congelados el título, el alcance, la tarea de clasificación, las métricas y la meta mínima de éxito, y cuando estos elementos sean coherentes con el resto del desarrollo técnico y documental del TFM.