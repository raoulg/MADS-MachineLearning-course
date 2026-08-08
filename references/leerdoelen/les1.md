# Les 1 — Representatie, compositie en search

Lees uit *Understanding Deep Learning* hoofdstuk 2-4 (p. 17-55) en hoofdstuk 10
(convoluties, p. 161-174). Convoluties en dense lagen zitten in deze les bij elkaar,
omdat ze dezelfde bouwsteen delen: `σ(Wx + b)` en `σ(W * x)` verschillen in hoe de
gewichten gedeeld worden, niet in wat de niet-lineariteit doet.

Sommige leerdoelen haal je beter uit een notebook, sommige uit de tekst, sommige uit de
les. Maak je geen zorgen als je ze niet allemaal meteen hebt; daar is de les voor.

## De student begrijpt

1.1 Wat tensors zijn en voorbeelden voor de dimensies tm 5D.

1.2 Hoe neurale netwerken zijn opgebouwd (linear + activation), en hoe ze een extensie
zijn van lineaire modellen (wanneer kun je ze stapelen).

1.3 Hoe neurale netwerken een universal function approximator zijn.

1.4 Wat gradient descent doet en wat een gradient is.

1.5 Wat een activation function is, en kent er een aantal (ReLU, Leaky ReLU, ELU, GELU,
Sigmoid).

1.6 Wat een lossfunctie is, en enkele voorbeelden — en waarom loss en metric niet
hetzelfde zijn, ook niet als ze op elkaar lijken.

1.7 Wat de invloed is van het aantal units in een dense layer.

1.8 Waarom we train-test-valid splits maken, en wat overfitting is en hoe regularisatie
(batchnorm, dropout, learning rate, ruis) daarbij helpt.

1.9 Wat de stappen zijn van het trainen van een NN: datapreparatie, trainbare gewichten,
predict, lossfunctie, optimizer.

1.10 Hoe deep learning past binnen de geschiedenis van AI, en wat deep learning kenmerkt
ten opzichte van de rest.

1.11 Het verschil tussen het *formele* niveau van een tensor (rank en shape; een 2D
tensor is een 2D tensor) en het *semantische* niveau (wat de assen betekenen). Kan de
treden van de ladder benoemen — (D) één item, (C,D) één gechunkt item, (B,D) een batch,
(B,C,D) een batch gechunkte items — en welke operatie tussen twee treden beweegt. Kan
uitleggen waarom "kan een CNN op een timeseries?" een formele vraag is met ja als
antwoord.

1.12 Het verschil tussen een *ontworpen* feature vector (transparant, met de hand
geschreven, bv regex counts) en een *geleerde* embedding (ondoorzichtig, pretrained), en
kan voor beide een situatie noemen waarin die wint.

### Convoluties

1.13 Wat het dimensionality-probleem is bij afbeeldingen waar dense netwerken last van
hebben, en hoe convoluties dat oplossen.

1.14 Wat kernel_size, stride, padding en maxpooling doen, en wat een 1x1 convolutie doet.

1.15 Hoe een convolutie schaalt (O(n)) ten opzichte van een dense laag (O(n²)).

1.16 Kent ruwweg de innovaties van de verschillende architecturen: AlexNet, VGG,
GoogleNet, ResNet, SENet.

1.17 Welke dimensionaliteit een convolutie nodig heeft, en wat de strategieën zijn om
dit te laten aansluiten op andere tensors.

### De cache: embed once, train many heads

1.18 Waarom een bevroren pretrained encoder wint van een klein, from-scratch getraind
CNN op een kleine dataset — in termen van wat die encoder al gezien heeft — en kan de
condities noemen waaronder dat *niet* meer opgaat.

1.19 Kan het kostenprofiel van beide aanpakken apart benoemen: eenmalige embedding-kosten
versus trainingskosten per experiment, en weet welke van de twee een hyperparameter-search
vermenigvuldigt.

1.20 Herkent dat een image embedding (D,) dezelfde weg naar beneden aflegt als elke
andere vector — zelfde Collate, zelfde head — en kan uitleggen waarom de pipeline daarmee
modaliteit-agnostisch is.

1.21 Kan uitleggen waarom de keuze van de encoder de accuracy tientallen punten beweegt
terwijl de architectuur van de head dat zelden meer dan één punt doet, en wat dat betekent
voor waar je je tijd aan besteedt.

1.22 Begrijpt dat een resultaat waarvan de marge binnen de eigen seed-spreiding valt geen
resultaat is, en dat een negatief resultaat ("het simpelere model won") een volwaardige
uitkomst is.

## De student is in staat om

1.23 Een configureerbaar dense neural network te bouwen met pytorch, en componenten
(`Serial`, `Parallel`, aggregators, heads) tot een pipeline samen te stellen.

1.24 De `metadata.json` van een cache lezen en, vóórdat er code draait, de shape van een
batch voorspellen plus welke componenten er tussen de DataLoader en de loss nodig zijn.

1.25 Een cache van de hub laden, subsamplen (met shuffle vóór het slicen) en er een
baseline op trainen die als vergelijkingspunt dient voor al het volgende.

1.26 MLflow gebruiken om experimenten te tracken en te vergelijken.

1.27 Een ray.tune search opzetten over een architectuurruimte, en het resultaat lezen —
inclusief het geval waarin de search niets vindt.

1.28 Een machine learning probleem classificeren als classificatie / regressie /
reinforcement / (semi)unsupervised.

## Materiaal

| leerdoel | onderwerp | boek | notebook |
|---|---|---|---|
| 1.1, 1.11 | tensors, shape vs betekenis, de ladder | 30-34 | `01b_tensors` |
| 1.2, 1.3, 1.7 | opbouw en stapelen van lagen | 25-30, 38-50 | `02_nonlinearity` |
| 1.4, 1.9 | gradient, optimalisatie | 22-23, 77-80, 91 | `03_training` |
| 1.5 | activation functions | 25, 37 | `02_nonlinearity` |
| 1.6 | loss vs metric | 23, hfst 5 | `03_training` |
| 1.8 | splits, overfitting, regularisatie | 118-120, 138, 147-149, 192-194 | `03_training`, `05_exercises` |
| 1.10 | DL binnen de AI-geschiedenis | 3-5, 37 | `deep_learning_1.pdf` |
| 1.12 | ontworpen features vs geleerde embeddings | les | `01_the_cache` |
| 1.13-1.17 | convoluties | 161-174, 186-191 | `02_nonlinearity`, `deep_learning_2.pdf` |
| 1.18-1.21 | de cache, encoderkeuze, kosten | les | `01_the_cache`, `04_search` |
| 1.22 | ruis, seeds, negatieve resultaten | les | `04_search`, `05_exercises` |
| 1.23, 1.24 | pipeline bouwen, shapes voorspellen | les | `02_nonlinearity`, `03_training` |
| 1.25 | cache laden en subsamplen | les | `01_the_cache`, `05_exercises` |
| 1.26, 1.27 | mlflow, ray.tune | les | `04_search`, `05_exercises` |
| 1.28 | soorten ML-problemen | les | les |

Optioneel: `01c_build_your_own_cache` laat zien hoe een cache gemaakt wordt. Je hebt het
niet nodig om de rest van de cursus te volgen — de caches staan klaar op de hub — maar
als je zelf een dataset wilt toevoegen begint het daar.

## Resources

- https://www.youtube.com/watch?v=FmpDIaiMIeA
- https://www.youtube.com/watch?v=KuXjwB4LzSA
