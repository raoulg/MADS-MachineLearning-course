Read chapter 5 (losses), chapter 10 (convolutions) and 11 (residual networks).

De student begrijpt:
2.1 - Wat de verschillende loss functies (MSE en RMSE, Negative Log Likelihood, Cross Entropy Loss, Binary Cross Entropy loss) zijn en in welke gevallen je ze moet gebruiken of vermijden.  
2.2 - Wat het dimensionality probleem is bij afbeeldingen waar Dense Neural Networks last van hebben  
2.3 - Hoe Convolutions dat dimensionality probleem oplossen  
2.4 - Wat maxpooling doet  
2.5 - Wat een stride en padding zijn  
2.6 - Wat een Convolution van 1x1 filter doet  
2.7 - Kent ruwweg de innovaties die de verschillende architecturen doen: AlexNet, VGG, GoogleNet, ResNet, SENet  
2.8 - Begrijpt wat overfitting is en hoe regularisatie (batchnorm, splits, dropout, learning rate) helpt.  
2.9 - hoe een Convolution schaalt (O(n)) ten opzicht van een NN (O(n^2))  

De student kan:  
2.10 - Een configureerbaar Convolution NN bouwen en handmatig hypertunen  
2.11 - MLFlow gebruiken naast gin-config  
2.12 - een Machine Learning probleem classificeren als classificatie / regressie / reinforcement / (semi)unsupervised  
2.13 - begrijpt welke dimensionaliteit een convolution nodig heeft en wat de strategieen zijn om dit te laten aansluiten op andere tensors  

Kijk https://www.youtube.com/watch?v=FmpDIaiMIeA  
Kijk https://www.youtube.com/watch?v=KuXjwB4LzSA  


|       topic | description                                                           |            page | notebooks                             |
| ----------- | --------------------------------------------------------------------- |   ------------- | -------------                         |
|         1.1 | understands how an image is represented as a 3D or 4D tensor          | lesson          | 02_convolutions                       |
|         2.1 | when to use which loss function                                       |           56-72 | 01_losses.ipynb                       |
|         2.2 | Which problem is solved by convolutions                               |             161 | 02_convolutions                       |
|         2.3 | How convolutions solve this problem                                   |    171, 161-174 | 02_convolutions                       |
|         2.4 | what is maxpooling                                                    | 171-172, 10.4.1 | 02_convolutions                       |
|         2.5 | Understand what kernel_size, stride and padding is and how to use it  |         164-165 | 02_convolutions.ipynb                 |
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
