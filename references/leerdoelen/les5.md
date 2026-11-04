# Les 5 — Self-supervised: waar de representatie vandaan komt

Lees uit *Understanding Deep Learning* hoofdstuk 14, en de blog over I-JEPA:
https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/

Les 1 gaf je een representatie in handen en liet zien dat die elke andere keuze
domineert. De voor de hand liggende vervolgvraag — waar kwam die vandaan, en zou ik er
zelf een kunnen maken voor een domein dat er geen heeft — is deze les.

## De student begrijpt

5.1 Hoe de architectuur van een autoencoder in elkaar zit, en dat het niet meer is dan
een encoder en een decoder achter elkaar.

5.2 Wat een latent space is, en hoe je die kunt visualiseren.

5.3 Wat het contrast is tussen een autoencoder en supervised learning, op de latent space
en op de lossfunctie.

5.4 **Waarom een pretext task een bruikbare ruimte oplevert**, en dat de keuze van de
pretext task de keuze is van wat de ruimte *vergeet*: reconstructie ordent op gelijkenis
in de input, contrastief leren ordent op invariantie onder de augmentatie die jij koos.

5.5 Wat een lineaire probe is en waarom die het meetinstrument is van deze les.

5.6 Wat de elementen van een siamese network zijn, en wanneer je er een zou willen
gebruiken.

5.7 Wat representation collapse is, hoe je het detecteert, en waarom een dalende loss
niets bewijst.

5.8 Wat de motivatie voor JEPA is, en hoe JEPA zich verhoudt tot autoencoders en tot
generatieve modellen: voorspellen in de *latente* ruimte in plaats van in pixels.

5.9 Waarom een patch-as behouden blijven soms beter is dan poolen, en wat pooling weggooit.

5.10 Wat praktische toepassingen zijn van een geleerde representatie, en hoe de latent
space als halffabrikaat voor andere modellen gebruikt kan worden.

5.11 Hoe anomaly detection en unsupervised classification met een geleerde representatie
werken.

5.12 Dat de kwaliteit van een representatie zich niet betrouwbaar laat aflezen aan de
meetkunde van de vectorruimte zelf, en dat een lineaire probe daarvoor de eerlijke — en
goedkope — meting is.

## De student is in staat om

5.13 Een autoencoder bouwen uit vectormesh-componenten (`Serial` van twee `NeuralNet`s)
en trainen op een cache.

5.14 Drie ruimtes op dezelfde data bouwen — autoencoder, contrastief, supervised — en ze
met dezelfde lineaire probe tegen elkaar afzetten.

5.15 Een encoder uit een getraind model hergebruiken voor een andere taak.

5.16 Een SSL-model hypertunen, met een collapse-detector aan de meting gekoppeld.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 5.1-5.5, 5.13, 5.14 | waarom pretext tasks werken; AE vs contrastief vs supervised | `01_why_pretext_tasks_work` |
| 5.6, 5.7, 5.16 | siamese, collapse, hypertunen | `02_selfsupervised` |
| 5.9, 5.10, 5.11 | patch tokens, wat pooling weggooit | `03_patch_tokens` |
| 5.8 | I-JEPA | `04_ijepa` |
| 5.12, 5.15 | de ruimte zelf meten, encoder hergebruiken | les |

## Resources

- https://youtu.be/qiUEgSCyY5o
- https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/
