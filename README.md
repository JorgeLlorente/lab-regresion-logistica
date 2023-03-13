# lab_classification
## Antecedentes:

Estás trabajando como analista de riesgos en un banco. Además de los otros servicios bancarios y de préstamos, el banco también ofrece servicios de tarjetas de crédito, una fuente muy importante de ingresos para el banco. El banco quiere entender la demografía y otras características de sus clientes que aceptan una oferta de tarjeta de crédito y los que no la aceptan.

Por lo general, los datos observacionales para este tipo de problemas son algo limitados en el sentido de que a menudo la empresa solo ve a aquellos que responden a una oferta. Para superar esto, el banco diseña un estudio de marketing enfocado, con 18000 clientes actuales del banco. Este enfoque centrado permite al banco saber quién responde y quién no responde a la oferta, y utilizar los datos demográficos existentes que ya están disponibles sobre cada cliente.

## Objetivo:
El objetivo es construir un modelo que proporcione información sobre por qué algunos clientes bancarios aceptan ofertas de tarjetas de crédito. También hay otras posibles áreas de oportunidades que el banco quiere entender a partir de los datos.

Por otro lado se han generado estas otras preguntas por parte de la direccion que te ayudarán a comprender mejor a sus clientes.

Datos: El conjunto de datos consta de información sobre 18,000 clientes actuales del banco en el estudio. Estas son las definiciones de los puntos de datos proporcionados:

- Customer Number: Este es un número secuencial asignado a los clientes (esta columna está oculta y excluida; este identificador único no se usará directamente).

- Offer Accepted: ¿Aceptó el cliente (Sí) o rechazó (No) la oferta?

- Reward: el tipo de programa de recompensa ofrecido para la tarjeta.

- Mailer Type:carta o postal.

- Income Level: bajo, medio o alto.

- Bank Accounts Open: cuántas cuentas no de tarjeta de crédito tiene el cliente.

- Overdraft Protection: ¿Tiene el cliente protección contra sobregiros en sus cuentas de cheques (Sí o No)?

- Credit Rating: baja, media o alta.

- Credit Cards Held: el número de tarjetas de crédito poseídas en el banco.

- Homes Owned: el número de viviendas poseídas por el cliente.

- Household Size: número de individuos en la familia.

- Own Your Home: ¿El cliente es propietario de su hogar? (Sí o No).

- Average Balance: saldo promedio de la cuenta (en todas las cuentas a lo largo del tiempo). Q1, Q2, Q3 y Q4

- Balance: saldo promedio para cada trimestre en el último año.

## Exploración de datos
Se recomienda comprender a fondo los datos y tomar las medidas necesarias para preparar los datos para el modelado antes de construir modelos exploratorios o predictivos. Como se trata de un modelo de clasificación, se debe utilizar la regresión logística para la clasificación al construir un modelo.

Para explorar los datos, se pueden utilizar las técnicas que se han discutido en clase. Algunas de ellas incluyen el uso del método "describe", comprobar los valores nulos, utilizar matplotlib y _seaborn


Los datos tienen varias variables categóricas y numéricas. Es importante explorar la naturaleza de los datos para estas variables antes de comenzar con el proceso de limpieza y preprocesamiento de datos (escalado de variables numéricas y codificación de variables categóricas).

Para la variable objetivo (Oferta aceptada - Sí/No), es importante verificar el desbalanceo de datos, es decir, el número de personas que respondieron con un "sí" frente al número de personas que respondieron con un "no".

Se debe profundizar en los datos para los clientes que aceptaron la oferta y para aquellos que no lo hicieron, y verificar sus características. Por ejemplo, seleccionar el nivel "Sí" en la variable "Oferta aceptada" y examinar la distribución de las ofertas aceptadas en relación con las otras variables en el conjunto de datos, y de manera similar para las personas que no aceptaron la oferta.


# Modelo

Compara las métricas de error en los conjuntos de train y test para asegurarte de que tu modelo no sufre de sobreajuste /subajuste y encuentra la escalera que mejor se ajuste a tus datos (tienes la libertad de probar otras escaleras que no se cubren en el curso).