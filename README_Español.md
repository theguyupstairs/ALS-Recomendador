# Sistema recomendador, filtro colaborativo para preferencias implícitas 

## Introducción 

Este sistema recomendador está basado en el paper ["Collaborative Filtering for Implicit Feedback Datasets," de Yifan Hu, Yehuda Koren y Chris Volinsky.](http://yifanhu.net/PUB/cf.pdf) 
Para la implementación, también me apoyé de [ALS Implicit Collaborative Filtering, escrito por Victor.](https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe)

En resumen, este paper nos presenta una manera de recomendar productos usando información de preferencias implícitas. Esto quiere decir que se suponen las preferencias de nuestros usuarios
a través de su comportamiento, en vez de por expresión directa (preferencias explícitas). Un ejemplo para ilustrar es un tablero de ventas a usuarios. Suponemos que un usuario demuestra
preferencia por un producto si lo compra. 

## ¿Por qué?

Decidí aprender sobre sistemas recomendadores para un e-commerce arbitrario. Comencé con lo más simple: reglas de asociación. Me di cuenta que este algoritmo sirve exclusivamente a 
industrias con matrices de información poco escasas. Los supermercados son un clásico explotador de este algoritmo. Como cada 'usuario' de supermercado compra muchos productos, es fácil 
crear asociaciones fuertes. El problema de escasez también me llevó a descartar el algoritmo de K-Nearest Neighbors (KNN). 

Familiar con la técnica de Singular Value Decomposition (SVD), decidí optar por la factorización de matrices. La primera implementación que se me ocurrió fue Funk SVD, un método
creado por Simon Funk para factorizar matrices usando descenso de gradiente estándar. Si bien es conocido por ser el recomendador más preciso, su adaptación requiere tener mayoritariamente
información explícita. También falla en eficiencia cuando se presenta a escasez excesiva. Por lo tanto, decidí optar por un sistema de Alternating Least Squares (ALS). 

## Alternating Least Squares

Este modelo, basado en el paper citado en la introducción, se ha vuelto predominante para datos escasos e implícitos. Le atribuye un factor de confianza a las preferencias de un usuario. 
Si un usuario compra un producto, se le atribuye un 1, y si no, un 0. Este número es multiplicado por el coeficiente de confianza (factor de cantidad de compras del producto/usuario dado y una
variable electa).

He adjuntado una explicación de este modelo en el archivo [ALS_EXPLICACION_Español.md.] 

## El Código

### Preparación

Este código incluye una preparación de datos. Comienza con una tabla de ventas arbitraria (que contiene id de usuario) y termina con una matriz CSR, ideal para comprimir información en matrices
escasas. Para llegar a esto es necesario agrupar las ventas por usuario. Después se cuenta cuántas veces cada usuario compra cada producto. Finalmente, se codifica a una matriz CSR usando variables
de categoría. El código contiene una explicación detallada.

### Dependencias

Para usar este código, aparte de adaptar las variables a su tabla de venta, es necesario descargar bibliotecas. Las descargas son Pandas, Numpy, Sklearn, Scipy y Pyspark.

## Mejoras

Para mejorar este archivo, es necesario implementar una prueba. Esta debe mostrar el Root Mean Square Error (RMSE), la principal métrica de error en regresiones. Otra forma de mejorar sería
reducir los usuarios demasiado escasos, y así incrementar la eficiencia.
