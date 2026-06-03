# PokemonGAN

En este repositorio se puede encontrar mi trabajo para la creación de un modelo GAN para la generación de sprites de Pokémon.

---

### Selección del dataset

Para poder crear nuestro modelo necesitaríamos un dataset con ciertas restricciones, por ejemplo, que todas las imágenes base de Pokémon sigan el mismo estilo de sprite 2D y no modelos 3D como en generaciones más actuales.

Afortunadamente, en Kaggle encontré el dataset [Pokemon sprite images](https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images), el cual contiene 10,437 imágenes en una resolución 96x96 de 898 Pokémon a lo largo de diferentes juegos.

---

### Preprocesado de los datos

Aunque pudiera parecer que encontramos oro en este dataset, tenemos que revisarlo a fondo y tomar decisiones en base a lo que encontremos. Para empezar, casi la mitad de nuestras imágenes son de Pokémon shiny, los cuales son Pokémon con su paleta de colores alterada y esto no nos conviene para entrenar a nuestro modelo. Tomemos de ejemplo a Charizard, un Pokémon tipo fuego muy icónico y con una paleta de colores roja que claramente indica que es tipo fuego, pero su versión shiny cambia a una paleta oscura, la cual podría confundir a nuestro modelo que apenas está logrando reconocer que rojo == fuego, por lo que decidí eliminar las imágenes de Pokémon shiny.

El dataset también contiene imágenes de los Pokémon de frente y de espaldas. Aunque en un principio decidí eliminar las imágenes de espaldas para ver si el modelo lograba aprender facciones faciales, al final opté por mantenerlas ya que duplican el tamaño efectivo del dataset y el modelo puede aprender a condicionarse en la orientación como una etiqueta más.

Para la normalización, todas las imágenes se transforman al rango [-1, 1] con media y desviación estándar de 0.5 en cada canal, que es el rango esperado por la función `Tanh` a la salida del generador. Adicionalmente, se aplica `RandomHorizontalFlip` como augmentación básica durante la carga del dataset.

---

### Implementación del modelo

Para la generación de imágenes decidí usar un modelo Generative Adversarial Network (GAN) multicondicional, el cual consiste en dos modelos, un generador que aprende a generar imágenes y un discriminador que aprende a distinguir entre los datos reales y los generados por el generador. En un entrenamiento ideal, el generador empieza produciendo imágenes claramente falsas y el discriminador fácilmente lo identifica, pero conforme sigue el entrenamiento el generador va mejorando a un punto donde el discriminador no logra discernir cuál imagen es real o del generador.

![GAN](GAN.png)

Para apoyarme en la creación y comprensión de mi modelo, usaré el artículo [Taming the Tail in Class-Conditional GANs: Knowledge Sharing via Unconditional Training at Lower Resolutions](https://openaccess.thecvf.com/content/CVPR2024/papers/Khorram_Taming_the_Tail_in_Class-Conditional_GANs_Knowledge_Sharing_via_Unconditional_CVPR_2024_paper.pdf) en el que se analiza a fondo cómo las GANs tienden a perder diversidad y calidad en las muestras cuando el dataset tiene clases muy específicas o distribuciones desbalanceadas (llamadas tail classes y por eso el nombre del artículo).

#### Arquitectura del Generador

El generador recibe dos entradas: un vector de ruido aleatorio `z` de dimensión 100, y un vector de etiquetas binarizado que codifica el tipo, la orientación y la forma del Pokémon. Ambas entradas se concatenan y se pasan por una serie de bloques de `ConvTranspose2d` que van aumentando progresivamente la resolución espacial hasta llegar a la imagen final de 96×96:

```
z + etiquetas (100 + n_labels, 1, 1)
        ↓ ConvTranspose2d
   Feature map (1024, 6, 6)
        ↓ Upsample block ×3
   Feature map (128, 48, 48)
        ↓ ConvTranspose2d + Tanh
   Imagen generada (3, 96, 96)
```

Cada bloque de upsampling consiste en un `ConvTranspose2d` para doblar la resolución, seguido de dos `Conv2d` adicionales para refinar los features, todo con `BatchNorm` y `LeakyReLU(0.2)`. Se usa `LeakyReLU` en lugar de `ReLU` para evitar el problema de *dying neurons* y `Tanh` a la salida para que los valores queden en el rango [-1, 1] consistente con la normalización del dataset.

#### Arquitectura del Discriminador

El discriminador hace el proceso inverso: recibe una imagen de 96×96 y la pasa por bloques de `Conv2d` que van reduciendo la resolución, terminando en un feature map de 512 canales. Desde ahí se bifurca en dos cabezas:

- **Cabeza adversarial**: clasifica si la imagen es real o falsa (salida escalar)
- **Cabeza auxiliar**: predice el tipo del Pokémon (salida de n_labels dimensiones)

Para estabilizar el entrenamiento, se aplica `SpectralNorm` en todas las capas convolucionales del discriminador. Esto controla la norma espectral de los pesos y evita que el discriminador se vuelva demasiado poderoso demasiado rápido, lo cual terminaría colapsando el gradiente que le llega al generador. Adicionalmente se usa `AdditiveGaussianNoise` al inicio de cada bloque del discriminador, con un std que decae linealmente a lo largo del entrenamiento, para regularizar y dificultar que el discriminador memorice las imágenes reales.

#### Funciones de pérdida

El modelo usa tres losses que se combinan con pesos:

**1. Pérdida adversarial** (`BCELoss`): la clásica de cualquier GAN, mide qué tan bien distingue el discriminador entre imágenes reales y falsas. El discriminador busca maximizarla y el generador minimizarla.

**2. Pérdida auxiliar** (`BCELoss` ponderada): mide qué tan bien predicen ambos modelos el tipo del Pokémon. Para compensar el desbalance natural del dataset (hay muchos más Pokémon de tipo Water que de tipo Flying, por ejemplo), cada clase tiene un peso inversamente proporcional a su frecuencia en el dataset.

**3. Pérdida contrastiva** (`Data2DataCrossEntropyLoss`): obliga al discriminador a aprender un espacio de embeddings donde imágenes del mismo tipo estén cerca entre sí y lejos de otros tipos, lo cual ayuda precisamente con el problema de tail classes del paper.

La pérdida total se calcula como:

```
Loss = Loss_adv + 0.25 * Loss_aux + 0.03 * Loss_clr
```

#### Augmentación durante el entrenamiento

Además del `RandomHorizontalFlip` aplicado al cargar el dataset, durante el entrenamiento se aplica **Differentiable Augmentation (DiffAugment)** a todas las imágenes antes de pasarlas al discriminador, tanto reales como falsas. Las políticas usadas son color, translation y cutout. Esto es especialmente importante dado el tamaño reducido del dataset (~5,000 imágenes después de filtrar shiny), ya que DiffAugment está diseñado precisamente para mejorar la calidad de GANs entrenadas con pocos datos.

#### Hiperparámetros

| Hiperparámetro | Valor |
|---|---|
| Dimensión del ruido `dim_z` | 100 |
| Batch size | 128 |
| Learning rate (G y D) | 0.0002 |
| Betas del optimizador Adam | (0.5, 0.999) |
| Épocas | 1000 |
| Peso de loss auxiliar `lambda_aux` | 0.25 |
| Peso de loss contrastiva `lambda_clr` | 0.03 |

El uso de `beta1=0.5` en lugar del default de 0.9 es una práctica estándar en GANs desde el paper original de DCGAN, ya que reduce el momentum del optimizador y hace el entrenamiento más estable.

---

### Evaluación inicial del modelo

Normalmente en un modelo de Deep Learning buscamos que la pérdida baje a cero, pero en una GAN si la pérdida de cualquiera de nuestros dos modelos llega a cero significa que ha fracasado. Al ser un entrenamiento "adversarial" entre el generador y discriminador, D_loss (la pérdida del discriminador) y G_loss (la pérdida del generador) representan qué tan balanceada está la competencia entre ambos modelos. Las métricas más directas para evaluar esto son `D(x)` y `D(G(z))`:

- **D(x)**: qué tan "real" clasifica el discriminador a las imágenes reales. En un entrenamiento balanceado debería acercarse a 0.7-0.8.
- **D(G(z))**: qué tan "real" clasifica el discriminador a las imágenes falsas. Si colapsa a 0, el generador dejó de recibir gradiente útil y el entrenamiento fracasó.

Al terminar de ejecutar nuestro primer modelo obtenemos las siguientes métricas:

```
D_Loss: 0.0128 | G_Loss: 3.0363
```

Estos valores nos dejan ver que el discriminador se volvió extremadamente preciso, lo que implica que `D(x) → 1` y `D(G(z)) → 0`. Cuando esto ocurre el gradiente que le llega al generador se vuelve prácticamente cero y deja de aprender, lo que se conoce como **mode collapse**. Esto es exactamente el escenario de tail classes descrito en el paper: el discriminador aprendió a memorizar las clases frecuentes y el generador no tuvo forma de competir en las clases con pocos ejemplos.

Y ahora veamos la métrica más importante, la calidad de las imágenes generadas:

![pokemones](pokemones.png)

Como era de esperarse, los "pokemones" generados no son más que manchas de colores. El generador supo aprender que los sprites contienen un contorno negro y los colores genéricos de cada tipo de Pokémon, pero no hay nada con forma de piernas, brazos, cara o alguna forma anatómica de Pokémon.

---

### Refinamiento del modelo

El colapso observado en el modelo inicial dejó claro que se necesitaban cambios fundamentales, no solo ajustes de hiperparámetros. Para el modelo refinado tomé como base el repositorio [PokeTypeGAN](https://github.com/ye-hongjiang/PokeTypeGAN) de ye-hongjiang, el cual implementa un AC-GAN con DiffAugment y Data2Data Cross-Entropy específicamente para la generación de sprites de Pokémon condicionada por tipo. A partir de esta base adapté el código para correr en Kaggle y ajusté los hiperparámetros.

#### Condicionamiento: de embeddings separados a vector binarizado

El modelo original condicionaba el generador y el discriminador pasando tres entradas separadas — tipo, color y si es legendario — cada una con su propio `Embedding` o capa `Dense`. El problema de este enfoque es que el discriminador recibía las etiquetas como una capa de activaciones proyectada espacialmente sobre la imagen, lo que lo hacía depender de que el generador y el discriminador acordaran una representación implícita de las etiquetas.

El modelo refinado abandona las etiquetas de color y legendario, y en su lugar usa un único vector binarizado que codifica directamente el tipo del Pokémon con `MultiLabelBinarizer`. Esto tiene dos ventajas: permite condicionamiento multi-etiqueta (un Pokémon puede ser tipo Fuego y tipo Volador simultáneamente), y el vector de etiquetas se concatena directamente con el ruido `z` en el espacio latente del generador, en lugar de inyectarse como canal adicional en la imagen.

#### Arquitectura del Generador: `Dense + UpSampling` vs `ConvTranspose2d`

El generador original partía de una capa `Dense` que proyectaba el vector latente a un feature map de 3×3, y luego usaba `UpSampling2D` (interpolación nearest-neighbor) seguido de `Conv2D` para ir subiendo la resolución. El problema de `UpSampling2D` es que la interpolación nearest-neighbor introduce artefactos de cuadrícula que el generador tiene que aprender a corregir, lo cual desperdicia capacidad.

El generador refinado usa `ConvTranspose2d` (también llamado deconvolución) para el upsampling, que aprende los pesos del upsample directamente en lugar de interpolar. Adicionalmente, cada bloque añade dos `Conv2d` extra para refinar los features antes del siguiente upsample, y reemplaza `ReLU` por `LeakyReLU(0.2)` en todas las capas intermedias para evitar el problema de *dying neurons*.

#### Arquitectura del Discriminador: una sola pérdida vs pérdidas múltiples

Este es el cambio más significativo. El discriminador original tenía una sola salida escalar (real/falso) y recibía las etiquetas como input adicional, pero no tenía ningún mecanismo para verificar si las imágenes generadas correspondían al tipo solicitado.

El discriminador refinado tiene dos cabezas de salida:

- **Cabeza adversarial**: clasifica real/falso, igual que antes
- **Cabeza auxiliar**: predice el tipo del Pokémon directamente desde la imagen

Esto convierte el modelo en un **AC-GAN** (Auxiliary Classifier GAN), donde el generador aprende no solo a engañar al discriminador sino también a generar imágenes que el discriminador clasifique correctamente por tipo. Adicionalmente se agrega `SpectralNorm` en todas las capas del discriminador y `AdditiveGaussianNoise` con std decreciente al inicio de cada bloque para regularizar.

#### Función de pérdida: una `BCELoss` vs tres losses combinadas

El modelo original usaba una única `BinaryCrossentropy` para todo. El modelo refinado combina tres:

- **Loss adversarial** (`BCELoss`): igual que antes, real vs falso
- **Loss auxiliar** (`BCELoss` ponderada por clase): mide qué tan bien predice el tipo el discriminador. Los pesos son inversamente proporcionales a la frecuencia de cada tipo en el dataset, compensando directamente el desbalance de clases que causó el colapso inicial
- **Loss contrastiva** (`Data2DataCrossEntropyLoss`): obliga al discriminador a aprender un espacio de embeddings donde imágenes del mismo tipo estén cercanas entre sí. Esta loss está motivada directamente por el paper de referencia y ataca el problema de tail classes desde el espacio de representación

```
Loss_total = Loss_adv + 0.25 * Loss_aux + 0.03 * Loss_clr
```

#### Augmentación: ninguna vs DiffAugment

El modelo original no aplicaba ninguna augmentación durante el entrenamiento, solo el `RandomHorizontalFlip` al cargar el dataset. El modelo refinado aplica **Differentiable Augmentation (DiffAugment)** en cada iteración a todas las imágenes que ve el discriminador, tanto reales como falsas, con políticas de color, traslación y cutout. Esto aumenta artificialmente la variedad de imágenes reales que ve el discriminador y evita que las memorice, que es una de las causas principales del colapso en datasets pequeños.