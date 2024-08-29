# Les 4
De student begrijpt:
4.1 - wat een wordembedding is en wat de motivatie is tov one-hot-encoding
4.2 - wat de curse of dimensionality is (bv met betrekking tot de searchspace)
4.3 - wat de voordelen van Ray zijn
4.4 - Hoe bayesian search en hyperband werken
4.5 - wat een learning rate scheduler is, en hoe je kunt herkennen dat je er een nodig hebt.
4.6 - Kent verschillende soorten schedulers (cosine warm up, reduce on plateau) en weet wanneer ze te gebruiken
4.7 - Begrijpt in welke situaties transfer-learning zinvol is

De student kan:
4.8 - de parameters in een pretrained model fixeren zodat het uitgebreid en gefinetuned kan worden
4.9 - Een pretrained model uitbreiden met een extra neuraal netwerk
4.10 - Een python script bouwen dat een configureerbaar model via Ray hypertuned.
4.11 - De student kan redeneren over de grootte van de hyperparameter ruimte, daar afwegingen in maken (curse of dimensionality) en prioriteiten stellen in de tuning van hyperparameters zoals lossfuncties, learning rate, units, aantal lagen, aantal filters, combinatie van Dense / Convolution.
4.12 - een afweging maken tussen de verschillende manieren om een model te monitoren (gin, tensorboard, mlflow, ray)

|            topic | description                 |                              notebooks |
|   -------------- | --------------------------- |  ------------------------------------- |
|              4.1 | wordembeddings              |                    01_embeddings.ipynb |
|              4.2 | curse of dimensionality     |                                lecture |
|              4.3 | ray                         |                  03_ray.ipynb, lecture |
|              4.4 | search algorithms           |           02_hypertuner.ipynb, lecture |
|         4.5, 4.6 | schedulers                  |                                lecture |
|    4.7, 4.8, 4.9 | transfer learning           | 04_transfer_learning_with_resnet.ipynb |
| 4.10, 4.11, 4.12 | hypertuning                 |            hypertune.py, 05_exercises.


Lees [[../Automated Machine Learning.pdf | Automated Machine Learning]], hoofdstuk 1.