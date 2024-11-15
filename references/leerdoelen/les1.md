Read pages 17-55 (chapter 2-4) from 'Understanding Deep Learning'. I have specified for you which pages answer specific learning goals to help you filter the more important parts.  
Some learning goals are better gained from a notebook, some from the text, some from the lesson. Don't worry if you dont get all of them; that's why we need a lesson :)  
There is a youtube video that also explains some stuff.  

De student begrijpt:  
1.1 Wat tensors zijn en voorbeelden voor de dimensies tm 5D  
1.2 hoe neurale netwerken zijn opgebouwd (linear + activation), en hoe ze een extensie zijn van lineaire modellen (wanneer kun je ze stapelen)  
1.3 Hoe neurale netwerken een universal function zijn.  
1.4 Wat gradient descent doet en wat een gradient is  
1.5 Wat de voordelen van een datastreamer zijn oa voor grote datasets  
1.6 Wat de invloed is van het aantal units in een dense layer  
1.7 Hoe een gin file werkt en wat de voordelen daarvan zijn.  
1.8 Waarom we train-test-valid splits maken  
1.9 Wat een Activation function is, en kent er een aantal (ReLU, Leaky ReLU, ELU, GELU, Sigmoid)  
1.10 wat een loss functie is, en enkele voorbeelden  
1.11 Hoe deep learning past binnen de geschiedenis van AI, en wat deep learning kenmerkt ten opzichte van de rest.  
1.12 Wat de stappen zijn van het trainen van een NN: datapreparatie, trainbare gewichten, predict, lossfunctie, optimizers  

De student is in staat om:  
1.13 Een configureerbaar dense neural network te maken met pytorch  
1.14 tensorboard gebruiken om zijn model te monitoren  
1.15 kan gin files gebruiken  
1.16 een dataclass maken  

watch: https://youtu.be/aircAruvnKk?si=2BVezkYqWVb6-eR2  

From the other chapters, sample the pages summed up below. You can skip the full chap 5 and 6 for now.

|                topic | description                                                |              page | notebooks                                 |
|--------------------- | -----------------------------------------------------------|     ------------- | -----------------------                   |
|                  1.2 | how are neural networks built up                           |             25-28 | 04_neuralnets.ipynb                       |
|                  1.3 | how are neural networks a universal function approximator  |    29-30, 38 , 50 |                                           |
|                  1.1 | scalars, vectors, matrix, batched 3D / 4D / 5D.            |     30-34, lesson | 01_tensors.ipynb                          |
|                  1.4 | what is a gradient?                                        |             77-80 | 07_extra_autograd.ipynb                   |
|                  1.6 | impact of hidden layer depth / width                       |             46-50 |                                           |
|                 1.10 | what is a loss function?                                   |     23, (chap 5)  | 2_convolutions/01_losses.ipynb            |
|                 1.11 | What is the difference between DL and other AI?            |               3-5 | deep_learning_01.pdf                      |
|                 1.11 | How did AI evolve over time?                               |                37 | deep_learning_01.pdf                      |
|                 1.12 | data, trainable weights, predict, loss, optimize           |             17-22 | 04_neuralnets.ipynb, deep_learning_01.pdf |
|                 1.12 | what is optimization?                                      | 22-23, 91 (chap 6)| 07_extra_autograd.ipynb                   |
|       1.5, 1.16, 1.8 | exercise I & II: be able to implement a data class         |     lesson        | 03_dataloader.ipynb                       |
|        1.2, 1.3, 1.6 | How can we stack multiple layers?                          |             41-46 | 04_neuralnets.ipynb, 06_exercises         |
|                  1.9 | What are activation functions?                             |            25, 37 | deep_learning_01.pdf                      |
|1.7, 1.13, 1.14, 1.15 | Learn to use gin & tensorboard to configure your model     |                   | 06_exercises.ipynb                        |
