# Les 1 — Representatie, compositie en search

Lees uit *Understanding Deep Learning* hoofdstuk 2-4 (p. 17-55) en hoofdstuk 10
(convoluties, p. 161-174). Convoluties en dense lagen zitten in deze les bij elkaar,
omdat ze dezelfde bouwsteen delen: `σ(Wx + b)` en `σ(W * x)` verschillen in hoe de
gewichten gedeeld worden, niet in wat de niet-lineariteit doet.

De les begint bij de cache: een bevroren, voorgetraind model heeft de vectorruimte al
geordend, en jij traint alleen het goedkope stuk erbovenop. Alles daarna — tensors,
lagen, trainen, zoeken — gebeurt op die vectoren.

Sommige leerdoelen haal je beter uit een notebook, sommige uit de tekst, sommige uit de
les. Maak je geen zorgen als je ze niet allemaal meteen hebt; daar is de les voor.

## De student begrijpt

1.1 Wat tensors zijn, met voorbeelden voor de dimensies tm 5D: een zin of een afbeelding
als één vector `(D)`, een batch vectoren `(B, D)`, een afbeelding `(C, H, W)`, een batch
afbeeldingen `(B, C, H, W)`, een batch video's `(B, T, C, H, W)`.

1.2 Hoe neurale netwerken zijn opgebouwd (linear + activation), en hoe ze een extensie
zijn van lineaire modellen. Kan uitleggen waarom een stapel lineaire lagen zonder
activatie samenvalt tot één lineaire afbeelding — dense én convolutioneel — en waarom
diepte zonder niet-lineariteit dus niets oplevert (UDL p. 35).

1.3 Wat de universal approximation theorem zegt: met genoeg hidden units bestaat er een
ondiep netwerk dat elke continue functie op een begrensd domein willekeurig goed benadert,
omdat elke extra ReLU-unit een extra lineair stuk toevoegt (UDL §3.2, p. 29-30,
figuur 3.5). En wat de stelling niet zegt: hoeveel units je nodig hebt, of training die
gewichten vindt, en of het resultaat generaliseert.

1.4 Wat gradient descent doet en wat een gradient is, en wat de optimizer daarmee doet:
de gradient zegt per gewicht welke kant op, de optimizer bepaalt hoe ver (Adam past die
stap per parameter aan).

1.5 Wat een activation function is, en kent er een aantal (ReLU, Leaky ReLU, GELU,
sigmoid, tanh). Kan saturatie (vanishing gradients bij sigmoid en tanh) en dode
ReLU-units uitleggen aan de hand van de afgeleide.

1.6 Wat een lossfunctie is, en enkele voorbeelden — en waarom loss en metric niet
hetzelfde zijn, ook niet als ze op elkaar lijken: de loss is er voor de optimizer en moet
differentieerbaar zijn, de metric is er voor jou en moet interpreteerbaar zijn. Weet
wanneer klassen met elkaar concurreren (`CrossEntropyLoss` + accuracy) en wanneer niet
(`BCEWithLogitsLoss` + F1), en dat beide logits verwachten.

1.7 Wat de invloed is van het aantal units in een dense layer: meer units is meer
capaciteit (meer lineaire stukken), en op een bevroren encoder koopt die capaciteit
zelden iets.

1.8 Waarom we train-test-valid splits maken, en wat overfitting is en hoe regularisatie
(batchnorm, dropout, learning rate, ruis op de inputvectoren) daarbij helpt — en waarom
regularisatie niet helpt als het plafond in de representatie zit en niet in de head.

1.9 Wat de stappen zijn van het trainen van een NN: datapreparatie, batchen (rijen
stapelen van `(D)` naar `(B, D)`, labels naar one-hot), forward, loss, backward,
optimizer-stap — en waarom `zero_grad()`, `eval()` en `no_grad()` op hun plek staan.

1.10 Hoe deep learning past binnen de geschiedenis van AI — de reeks narcistische
krenkingen waarin de mens telkens minder uniek bleek — en wat deep learning kenmerkt ten
opzichte van de rest.

1.11 Het verschil tussen het *formele* niveau van een tensor (rank en shape; een 2D
tensor is een 2D tensor) en het *semantische* niveau (wat de assen betekenen). Kan
uitleggen dat een batch (losse items op elkaar stapelen, `(B, D)`) en een chunk (één item
in stukken knippen, `(C, D)`) dezelfde shape geven met een andere betekenis, en dat
torch dat verschil niet ziet. Kan de treden van de ladder benoemen — (D) één item,
(C,D) één gechunkt item, (B,D) een batch, (B,C,D) een batch gechunkte items — en welke
operatie tussen twee treden beweegt. Kan uitleggen waarom "kan een CNN op een
timeseries?" een formele vraag is met ja als antwoord.

1.12 Het verschil tussen een *ontworpen* feature vector (transparant, met de hand
geschreven, bv regex counts) en een *geleerde* embedding (ondoorzichtig, pretrained), en
kan voor beide een situatie noemen waarin die wint.

### Convoluties

1.13 Wat het dimensionality-probleem is bij afbeeldingen waar dense netwerken last van
hebben, en hoe convoluties dat oplossen: een convolutie is een lineaire operator met
weight sharing en lokaliteit.

1.14 Wat kernel_size, stride, padding en maxpooling doen, en wat een 1x1 convolutie doet.
Kan de output-shape van een convolutie voorspellen.

1.15 Hoe een convolutie schaalt (O(n)) ten opzichte van een dense laag (O(n²)).

1.16 Kent ruwweg de innovaties van de verschillende architecturen: AlexNet, VGG,
GoogleNet, ResNet, SENet.

1.17 Welke dimensionaliteit een convolutie nodig heeft (`(B, C, L)` of `(B, C, H, W)`,
channels op de tweede plek), en wat de strategieën zijn om dit te laten aansluiten op
andere tensors.

### De cache: embed once, train many heads

1.18 Dat een voorgetraind foundation model de hoogdimensionale vectorruimte al ordent
zonder dat het ooit op jouw taak getraind is — vergelijkbare items liggen dicht bij
elkaar, zodat 1-nearest-neighbour zonder enige training al classificeert — en dat je die
ordening kunt gebruiken voor je eigen doel (klassen voorspellen), ook al was dat niet het
trainingsdoel van de encoder. Kan de condities noemen waaronder dat *niet* meer opgaat:
een domein ver van de pretraining-data, of een modaliteit zonder goede encoder.

1.19 De economie van beide aanpakken: embedden is duur en gebeurt één keer, een head
trainen kost seconden en gebeurt honderden keren. Zolang je niet finetunet, is de encoder
bij elk experiment opnieuw draaien honderd keer dezelfde vectoren uitrekenen. Een
hyperparameter-search vermenigvuldigt de kosten van de head, niet die van de embedding.

1.20 Herkent dat een image embedding (D,) dezelfde weg naar beneden aflegt als elke
andere vector — zelfde Collate, zelfde head — en kan uitleggen waarom de pipeline daarmee
modaliteit-agnostisch is.

1.21 Dat verschillende encoders verschillende resultaten geven, en dat dat bepaald wordt
door waarop ze getraind zijn en welke taak je ze geeft: dezelfde drie encoders liggen op
EuroSAT binnen 3 punten van elkaar en op DTD 14 punten uit elkaar, en meer dimensies is
geen betere representatie. Kan uitleggen waarom de keuze van de encoder de accuracy meer
dan tien punten beweegt terwijl de architectuur van de head dat zelden meer dan één punt
doet, en wat dat betekent voor waar je je tijd aan besteedt.

1.22 Begrijpt dat een resultaat waarvan de marge binnen de eigen seed-spreiding valt geen
resultaat is, dat het maximum van een search geen eerlijke vergelijking is met het
gemiddelde van een baseline, en dat een negatief resultaat ("het simpelere model won")
een volwaardige uitkomst is.

## De student is in staat om

1.23 Een configureerbaar dense neural network te bouwen met pytorch, en componenten
(`Serial`, `Parallel`, `Concatenate2D`, aggregators, heads) tot een pipeline samen te
stellen.

1.24 De `metadata.json` van een cache lezen en, vóórdat er code draait, de shape van een
batch voorspellen plus welke componenten er tussen de DataLoader en de loss nodig zijn.

1.25 Een cache van de hub laden, subsamplen (met shuffle vóór het slicen) en er een
baseline op trainen — met seed-spreiding, gelogd in MLflow, vóór er iets gezocht wordt —
die als vergelijkingspunt dient voor al het volgende.

1.26 MLflow gebruiken om experimenten te tracken en te vergelijken, vanaf de eerste run.

1.27 Een ray.tune search opzetten: de zoekruimte definiëren (welke assen erin, welke
eruit, welke ranges, en waarom), het resultaat per as lezen — inclusief het geval waarin
de search niets vindt — en de winnaar over meerdere seeds opnieuw draaien voordat je hem
een winst noemt.

1.28 Een machine learning probleem classificeren als classificatie / regressie /
reinforcement / (semi)unsupervised, en uitleggen hoe reinforcement learning werkt als een
lus van actie, beloning en bijsturing.

1.29 Een architectuur lezen en tekenen als een keten van tensor-shapes — bij elke pijl de
shape, van `(B, D)` tot `(B, K)` — inclusief een parallelle (fusie-)architectuur waarin
twee streams, bv `(B, 384)` en `(B, 512)`, elk een eigen tak krijgen en via concatenatie
in één head samenkomen. Kan aanwijzen waar in zo'n tekening de parameters zitten.

1.30 De eigen runs volgens het inleverprotocol indienen (`docs/submit-runs.md`), en in
het MLflow-overzicht van de hele klas, waarmee de volgende les opent, de eigen
architectuur en score vergelijken met die van de anderen, tegen dezelfde baseline.

## Materiaal

| leerdoel | onderwerp | boek | notebook |
|---|---|---|---|
| 1.1, 1.11 | tensors, shape vs betekenis, batch vs chunk, de ladder | 30-34 | `01b_tensors`, `01_the_cache` |
| 1.2 | opbouw en stapelen van lagen, lineaire collapse | 25-30, 35, 41-50 | `02_nonlinearity` |
| 1.3 | universal approximation theorem | 29-30 (fig. 3.5) | `02_nonlinearity` |
| 1.7 | aantal units, capaciteit | 29-30, 34 | `02_nonlinearity`, `04_search` |
| 1.4, 1.9 | gradient, optimizer, de trainingsloop, batchen | 22-23, 77-80, 91 | `03_training` |
| 1.5 | activation functions | 25, 37 | `02_nonlinearity` |
| 1.6 | loss vs metric | 23, hfst 5 | `03_training` |
| 1.8 | splits, overfitting, regularisatie | 118-120, 138, 147-149, 192-194 | `03_training`, `05_exercises` |
| 1.10 | DL binnen de AI-geschiedenis | 3-5, 37 | presentatie les 1 |
| 1.12 | ontworpen features vs geleerde embeddings | les | les |
| 1.13-1.17 | convoluties | 161-174, 186-191 | `02_nonlinearity`, `deep_learning_2.pdf` |
| 1.18-1.21 | de cache, encoderkeuze, kosten | les | `01_the_cache`, `04_search` |
| 1.22 | ruis, seeds, negatieve resultaten | les | `04_search`, `05_exercises` |
| 1.23, 1.24 | pipeline bouwen, shapes voorspellen | les | `02_nonlinearity`, `03_training` |
| 1.25 | cache laden, subsamplen, baseline | les | `01_the_cache`, `03_training`, `05_exercises` |
| 1.26, 1.27 | mlflow, ray.tune | les | `04_search`, `05_exercises` |
| 1.28 | soorten ML-problemen, reinforcement learning | les | presentatie les 1 |
| 1.29 | architecturen lezen met shapes, fusie | les | `04_search` |
| 1.30 | runs inleveren en vergelijken | les | `05_exercises`, `docs/submit-runs.md` |

Optioneel: `01c_build_your_own_cache` laat zien hoe een cache gemaakt wordt. Je hebt het
niet nodig om de rest van de cursus te volgen — de caches staan klaar op de hub — maar
als je zelf een dataset wilt toevoegen begint het daar.

## Resources

- https://www.youtube.com/watch?v=FmpDIaiMIeA
- https://www.youtube.com/watch?v=KuXjwB4LzSA
