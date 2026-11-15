# Les 1 — Representatie en head

Lees uit *Understanding Deep Learning* hoofdstuk 2-4 (p. 17-55).

De les begint bij de cache: een bevroren, voorgetraind model heeft de vectorruimte al
geordend, en jij traint alleen het goedkope stuk erbovenop. Alles daarna — tensors,
lagen, trainen, zoeken — gebeurt op die vectoren. Kernels, state en gates zijn het
onderwerp van les 2.

Sommige leerdoelen haal je beter uit een notebook, sommige uit de tekst, sommige uit de
les. Maak je geen zorgen als je ze niet allemaal meteen hebt; daar is de les voor.

## De student begrijpt

1.1 Wat een tensorshape is en hoe je die gebruikt om data te beschrijven: 1D is een
vector (een rij), 2D een matrix (een vlak), 3D een tensor (een kubus), 4D een batched
tensor (een batch kubussen). Een as van lengte 1 kun je altijd invoegen of weghalen
zonder dat de data verandert: een zwart-witplaatje kan zowel `(H, W)` als `(1, H, W)`
zijn.

1.2 Een tensor heeft twee aspecten: de vorm (rank en shape) en de semantiek (wat de
assen betekenen). Tijdens het trainen voeg je altijd een batchdimensie toe. Dezelfde
matrix `(3, 4)` kan dan drie vectoritems in een batch zijn, `(3, 1, 4)`, of één 2D-item
in een batch van één, `(1, 3, 4)`. De shape vertelt je niet of de data tekst of beeld
is.

1.3 Een pretrained model ordent jouw data in een vectorruimte zonder ooit op jouw data
getraind te zijn. Vergelijkbare items liggen (hopelijk) dicht bij elkaar, zodat
1-nearest-neighbour al resultaat kan geven. Dat werkt niet altijd: bijvoorbeeld bij een
domein ver van de pretrainingdata, of een modaliteit zonder goede encoder.

1.4 Een encoder berekent embeddings één keer en slaat ze op in een cache. Dat vraagt
veel rekenkracht (FLOPs). Daarna kun je een downstream model (een head) met veel minder
FLOPs trainen op de vectoren uit de cache. Een hyperparametersearch herhaalt zo alleen
het trainen van de head, en dat scheelt veel FLOPs.

1.5 De keuze van de encoder kan meer verschil maken voor de score dan de architectuur
van de head. Welke encoder het best werkt, hangt onder andere af van het verschil
tussen zijn pretrainingdataset en jouw taak. Dit verschil analyseer je tijdens de
hyperparameteroptimalisatie (zie 1.14).

1.6 Zonder nonlineariteit σ is de compositie van lineaire functies f₁ ∘ f₂ ∘ f₃ gelijk
aan één lineaire functie f₄. De universal approximation theorem zegt dat een netwerk f₁
∘ σ ∘ f₂ met voldoende capaciteit en een geschikte nonlineariteit elke continue functie
op een compact domein willekeurig nauwkeurig kan benaderen. De stelling zegt niet
hoeveel capaciteit nodig is, of training die oplossing vindt, of de functie
generaliseert.

1.7 In één trainingsstap batch je de data, maak je predictions en meet je de error met
een loss. De gradient geeft de richting van een verbetering aan; de optimizer bepaalt
hoe groot de stap is.

1.8 De lossfunctie moet bruikbaar zijn voor gradient descent en stuurt het leren. Een
metric beoordeelt de voorspellingen in een maat die je als mens kunt interpreteren,
zoals accuracy of MAPE. De data en het doel bepalen mede welke loss je kiest; dezelfde
maat, zoals MSE, kan soms beide rollen vervullen.

1.9 De trainingssplit leert de gewichten, de validatiesplit helpt bij
hyperparameteroptimalisatie, en de testsplit beoordeelt het gekozen model. Een te groot
verschil tussen train en valid wijst op overfitting; regularisatie kan daarbij helpen.
Wat de encoder nooit in zijn embedding stopte, kan geen enkele head er ooit weer uit
halen.

1.10 Vergelijk een gevonden verbetering binnen je budget (tijd/FLOPs) met de spreiding
tussen herhaalde runs. Een verschil binnen die spreiding is nog geen overtuigend
resultaat. Vergelijk de winnaar van een hyperparametersearch met de baseline; als het
simpelere model wint, is dat ook een resultaat.

1.11 Begrijp hoe classificatie, regressie, reinforcement learning en
(semi-)unsupervised learning van elkaar verschillen.

## De student kan

1.12 Lees een architectuur als een keten van blokken met de tensorshape bij elke pijl.
Gebruik de `metadata.json` van een cache om vooraf te begrijpen welke shapes en
componenten nodig zijn.

1.13 Meet eerst een baseline: laad een cache, bepaal bijvoorbeeld de score van
1-nearest-neighbour en train een simpele lineaire head. Herhaal het trainen van de head
binnen je FLOPs-budget om de spreiding tussen runs te zien en log die resultaten
voordat je een complexere architectuur onderzoekt.

1.14 Teken en verdedig je zoekruimte vóór de search: wat is je budget in termen van
tijd, FLOPs en hardware? Welke assen en parameterranges kies je, wat laat je weg, en
waarom? Lees daarna het resultaat van een hyperparametersearch per as, ook wanneer die
geen verbetering vindt. Herhaal de winnende configuratie binnen budget om een verschil
overtuigend winst te noemen.

1.15 Publiceer je runs volgens `docs/submit-runs.md`. Bekijk daarna de resultaten van
klasgenoten in het klasoverzicht en bouw met bronvermelding voort op wat hun
experimenten laten zien.

## Materiaal

| leerdoel | onderwerp | boek | notebook |
|---|---|---|---|
| 1.1, 1.2 | tensor shapes, shape en betekenis, batchdimensie | 30-34 | `01b_tensors`, `01_the_cache` |
| 1.3, 1.4, 1.5 | de cache, de kosten, de encoderkeuze | les | `01_the_cache`, `04_search` |
| 1.6 | lineaire collapse, nonlineariteit, capaciteit | 25-30, 35, 41-50 | `02_nonlinearity` |
| 1.7, 1.8 | de trainingsstap, loss en metric | 22-23, 77-80, 91, hfst 5 | `03_training` |
| 1.9 | splits, overfitting, regularisatie | 118-120, 138, 147-149 | `03_training`, `05_exercises` |
| 1.10 | spreiding tussen runs, eerlijke vergelijking, negatieve resultaten | les | `04_search`, `05_exercises` |
| 1.11 | soorten leerproblemen | les | presentatie les 1 |
| 1.12 | compositieschema's en tensor shapes | les | `04_search`, `05_exercises` |
| 1.13 | de baseline | les | `03_training`, `04_search`, `05_exercises` |
| 1.14 | de zoekruimte en de search | les | `04_search`, `05_exercises` |
| 1.15 | publiceren en voortbouwen | les | `05_exercises`, `docs/submit-runs.md` |

Optioneel: `01c_build_your_own_cache` laat zien hoe een cache gemaakt wordt. Je hebt het
niet nodig om de rest van de cursus te volgen — de caches staan klaar op de hub — maar
als je zelf een dataset wilt toevoegen begint het daar.

## Resources

- Intelligentie: https://youtu.be/JxbeMbEp-Ak
- Semantic vector spaces I: https://www.youtube.com/watch?v=G5wlnlCWfVk
- Semantic vector spaces II: https://www.youtube.com/watch?v=9kBW9Nhd0Zw
- What is a neural network: https://www.youtube.com/watch?v=aircAruvnKk
