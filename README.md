# PokemonGAN

En este repositorio se puede encontrar mi trabajo para la creación de un modelo GAN para la generación de sprites de Pokémon.
---
### Selección del dataset
Para poder crear nuestro modelo necesitaríamos un dataset con ciertas restricciones, por ejemplo, que todas las imágenes base de Pokémon sigan el mismo estilo de sprite 2D y no modelos 3D como en generaciones más actuales.
Afortunadamente, en Kaggle encontré el dataset [Pokemon sprite images](https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images), el cual contiene 10,437 imágenes en una resolución 96x96 de 898 Pokémon a lo largo de diferentes juegos.

### Preprocesado de los datos
Aunque pudiera parecer que encontramos oro en este dataset, tenemos que revisarlo a fondo y tomar decisiones en base a lo que encontremos. Para empezar, casi la mitad de nuestras imágenes son de Pokémon shiny, los cuales son Pokémon con su paleta de colores alterada y esto no nos conviene para entrenar a nuestro modelo. Tomemos de ejemplo a Charizard, un Pokémon tipo fuego muy icónico y con una paleta de colores roja que claramente indica que es tipo fuego, pero su versión shiny cambia a una paleta oscura, la cual podría confundir a nuestro modelo que apenas está logrando reconocer que rojo == fuego, por lo que decidí eliminar las imágenes de Pokémon shiny.
El dataset también contiene imágenes de los Pokémon de frente y de espaldas, esta decisión es un poco más arbitraria, pero decidí eliminar también las imágenes de Pokémon de espalda ya que quiero ver si el modelo logra crear Pokémon con facciones faciales o similar.

### Implementación del modelo
Para la generación de imágenes decidí usar un modelo Generative Adversarial Network (GAN) multicondicional (tipo, color, y si es legendario el Pokémon), el cual consiste en dos modelos, un generador que aprende a generar imágenes y un discriminador que aprende a distinguir entre los datos reales y los generados por el generador.
En un entrenamiento ideal, el generador empieza produciendo imágenes claramente falsas y el discriminador fácilmente lo identifica, pero conforme sigue el entrenamiento el generador va mejorando a un punto donde el discriminador no logra discernir cuál imagen es real o del generador.
![GAN](GAN.png)

Para apoyarme en la creación y comprensión de mi modelo, usaré el artículo [Taming the Tail in Class-Conditional GANs: Knowledge Sharing via Unconditional Training at Lower Resolutions](https://openaccess.thecvf.com/content/CVPR2024/papers/Khorram_Taming_the_Tail_in_Class-Conditional_GANs_Knowledge_Sharing_via_Unconditional_CVPR_2024_paper.pdf) en el que se analiza a fondo cómo las GANs tienden a perder diversidad y calidad en las muestras cuando el dataset tiene clases muy específicas o distribuciones desbalanceadas (llamadas tail classes y por eso el nombre del artículo).

### Evaluación inicial del modelo
Normalmente en un modelo de Deep Learning buscamos que la pérdida baje a cero, pero en una GAN si la pérdida de cualquiera de nuestros dos modelos aumenta significa que ha fracasado.
Al ser un entrenamiento "adversarial" entre el generador y discriminador, D_loss (la pérdida del discriminador) y G_loss (la pérdida del generador) representan qué tan balanceada está la competencia entre ambos modelos.

Al terminar de ejecutar nuestro primer modelo obtenemos las siguientes métricas:
D_Loss: 0.0128 | G_Loss: 3.0363
Estos valores nos dejan ver que el discriminador se volvió muy preciso para distinguir las imágenes y por ende el entrenamiento del generador terminó colapsando.
Y ahora veamos la métrica más importante, la calidad de las imágenes generadas:
![pokemones](pokemones.png)
Como era de esperarse, los "pokemones" generados no son más que manchas de colores. El generador supo aprender que los sprites contienen un contorno negro y los colores genéricos de cada tipo de Pokémon, pero no hay nada con forma de piernas, brazos, cara o alguna forma anatómica de Pokémon.
### Refinamiento del modelo
