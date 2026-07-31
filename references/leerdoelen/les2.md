Read chapter 10 (convolutions) and 11 (residual networks). Chapter 5 (losses) now belongs to
lesson 1, together with `1_pytorch_intro/05_losses.ipynb` and learning goal 1.9.

De student begrijpt:  
2.2 - Wat het dimensionality probleem is bij afbeeldingen waar Dense Neural Networks last van hebben  
2.3 - Hoe Convolutions dat dimensionality probleem oplossen  
2.4 - Wat maxpooling doet  
2.5 - Wat een stride en padding zijn.  
2.6 - Wat een Convolution van 1x1 filter doet  
2.7 - Kent ruwweg de innovaties die de verschillende architecturen doen: AlexNet, VGG, GoogleNet, ResNet, SENet   
2.8 - Begrijpt wat overfitting is en hoe regularisatie (batchnorm, splits, dropout, learning rate) helpt.  
2.9 - hoe een Convolution schaalt (O(n)) ten opzicht van een NN (O(n^2)).  

De student kan:  
2.10 - Een configureerbaar Convolution NN bouwen en handmatig hypertunen  
2.11 - MLFlow gebruiken om experimenten te tracken en te vergelijken  
2.12 - een Machine Learning probleem classificeren als classificatie / regressie / reinforcement / (semi)unsupervised.  
2.13 - begrijpt welke dimensionaliteit een convolution nodig heeft en wat de strategieen zijn om dit te laten aansluiten op andere tensors  
2.14 - kan uitleggen waarom een pretrained encoder het wint van een klein, from-scratch getraind CNN op een kleine dataset — in termen van wat die encoder al gezien heeft — en kan de condities noemen waaronder dat *niet* meer opgaat.  
2.15 - kan het kostenprofiel van beide aanpakken apart benoemen: eenmalige embedding-kosten versus training-kosten per experiment, en weet welke van de twee een hyperparameter-search vermenigvuldigt.  
2.16 - herkent dat een image embedding `(D,)` dezelfde weg naar beneden aflegt als elke andere vector — zelfde `Collate`, zelfde head — en kan uitleggen waarom de pipeline daarmee modaliteit-agnostisch is.  

Kijk https://www.youtube.com/watch?v=FmpDIaiMIeA  
Kijk https://www.youtube.com/watch?v=KuXjwB4LzSA  


|       topic | description                                                           |            page | notebooks                             |
| ----------- | --------------------------------------------------------------------- |   ------------- | -------------                         |
|         1.1 | understands how an image is represented as a 3D or 4D tensor          | lesson          | 01_convolutions                       |
|         2.2 | Which problem is solved by convolutions                               |             161 | 01_convolutions                       |
|         2.3 | How convolutions solve this problem                                   |    171, 161-174 | 01_convolutions                       |
|         2.4 | what is maxpooling                                                    | 171-172, 10.4.1 | 01_convolutions                       |
|         2.5 | Understand what kernel_size, stride and padding is and how to use it  |         164-165 | 01_convolutions.ipynb                 |
|         2.7 | Understand how AlexNet works                                          | lesson          | deep_learning_2.pdf                   |
|         2.7 | Understand what a residual layer is                                   |         186-191 | deep_learning_2.pdf                   |
|         2.7 | Understand how an inception layer works                               | lesson          | deep_learning_2.pdf                   |
|         2.7 | Understand how a VGG layer works                                      | lesson          | deep_learning_2.pdf                   |
|         2.7 | Understand how a SEnet works                                          | lesson          | deep_learning_2.pdf                   |
|         2.8 | Understands batchnorm                                                 |         192-194 | deep_learning_2.pdf                   |
|         2.8 | Understand why to use train-valid-test splits                         |         118-120 | deep_learning_2                       |
|   2.8, 2.10 | Know how to recognize overfitting from a dashboard and what to do     |  119, chapter 9 | 03_mlflow, exercises, deep_learning_2 |
|         2.8 | Understand what dropout does and how to implement it                  |         147-149 | deep_learning_2                       |
|         2.8 | Understand regularization                                             |             138 | deep_learning_2                       |
|         2.8 | Understand learning rate                                              |           85-90 | deep_learning_2                       |
|        2.11 | Track and compare runs in one MLflow experiment                       | lesson          | 03_mlflow                             |
|        2.14 | Why a frozen pretrained encoder beats a small CNN, and when it doesn't | lesson          | 02_pretrained                         |
|        2.15 | One-time encoding cost vs per-experiment training cost                | lesson          | 02_pretrained, 03_mlflow              |
|        2.16 | An image embedding takes the same path as any other vector            | lesson          | 02_pretrained                         |
