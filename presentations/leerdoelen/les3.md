# Les 3 — De impact van scaling laws

Lees *Automated Machine Learning* hoofdstuk 1, en uit *Understanding Deep Learning*
hoofdstuk 6, 8 en 9. Die hoofdstukken gaan over het trainingsproces in het algemeen en
geven de theorie onder wat je in les 1 en 2 al deed.

Een model wordt bijna altijd beter als je het meer gewichten, meer data of meer compute
geeft, en dat verband is gemeten: voor LLMs, image models en sequence models daalt de test
loss als een rechte lijn op log-log-assen. Dat zijn de scaling laws. Ze maken van elke
architectuurclaim een vraag: test je een beter idee, of gewoon 10× zoveel gewichten? Deze
les leert je die twee uit elkaar halen, met de gate uit les 2 als voorbeeld: in een
highway en in een mixture of experts kiest een gate tussen parallelle routes.

## De student begrijpt

3.1 Waarom je twee architecturen test bij een gelijk aantal gewichten: een baseline met
minder gewichten verliest misschien alleen omdat hij kleiner is. Kan bij een
architectuurclaim de control aanwijzen: hetzelfde model zonder het nieuwe onderdeel, met
evenveel gewichten.

3.2 Wat de scaling laws zeggen: de test loss daalt als functie van compute (PF-days),
dataset size (tokens) en parameters (non-embedding), zolang de andere twee niet de
bottleneck zijn. Kan zo'n figuur lezen: een rechte lijn op log-log-assen, waarvan de
helling zegt hoeveel 10× meer oplevert, alleen binnen het gemeten bereik. Weet dat dit
verband gemeten is in LLMs, image models en sequence models (speech, video), en dat de
helling per domein verschilt en dus op je eigen probleem gemeten moet worden.

3.3 Dat gewichten en data samen moeten groeien: een groter model op dezelfde data gaat
overfitten. Kan uit een gemeten verhouding aflezen hoeveel data erbij moet; bij Kaplan
vraagt 8× zoveel gewichten maar ongeveer 5× zoveel data. Weet dat die verhouding per
studie verschilt.

3.4 Dat een gate een onderdeel is van een parallelle architectuur: een highway weegt een
blok met g en de skip eromheen met 1 − g, en een mixture of experts is een highway met
meer routes, waarbij een router voor elke expert een gewicht geeft. Kan in het schema
aanwijzen welke routes worden samengevoegd en welke shapes daarvoor moeten passen.

3.5 Dat een sparse mixture of experts per input maar een deel van zijn gewichten
gebruikt: hij traint alle experts en rekent alleen de gekozen expert uit. Kan het totale
en het gebruikte aantal gewichten uit elkaar houden, weet dat minder gebruikte gewichten
niet vanzelf sneller is, en herkent router collapse: alle inputs naar één expert. Weet dat
load balancing de router dwingt de experts te spreiden.

3.6 Hoe grid search, random search, TPE en een populatie hun volgende trial kiezen: grid
weet niets, random trekt in plaats van op te sommen, TPE leert van eerdere trials, een
populatie houdt goede kandidaten vast. Kan bij elk zeggen welke tekortkoming van het
vorige het oplost.

3.7 Hoe een afvalrace je budget beter benut: successive halving en ASHA starten veel trials
maar geven alleen de besten meer epochs. Weet dat een trial die laat op gang komt daardoor
ten onrechte kan afvallen, en dat een zoekruimte exponentieel groeit met het aantal assen,
terwijl je budget gelijk blijft.

3.8 Wat een learning rate scheduler doet: binnen één training de learning rate verlagen,
eerst grote stappen, dan kleine. Kan een schedule over epochs lezen, en het verschil
zeggen met een afvalrace, die hele trials stopt.

3.9 Wat een Pareto front is: de modellen waarvan geen ander model op alle gekozen maten
minstens even goed is en op één beter. Kan een Pareto front lezen over score, aantal
gewichten en snelheid, en uitleggen waarom "de beste trial" zonder tweede maat een
onvolledige vraag is.

## De student kan

3.10 De relatie meten tussen de loss en minstens twee van gewichten, data en compute, met
herhaalde runs, en er een conclusie uit trekken die groter is dan de spreiding. Kan een
conclusie trekken die verder gaat dan de hoogste score, bijvoorbeeld dat een kleiner model
de voorkeur heeft omdat het bijna even goed scoort en sneller is of minder energie
verbruikt.

3.11 Scaling verwerken in een zoekruimte: een as voor het aantal gewichten opnemen of een
control met evenveel gewichten ernaast zetten, zodat de search het effect van een
architectuur kan scheiden van het effect van meer gewichten. Kiest de ranges van gewichten,
data en epochs op basis van een gemeten curve.

3.12 Een parallelle architectuur met routing ontwerpen, trainen en tunen op de
hartslagen, met een control met evenveel gewichten ernaast: vooraf een verwachting
uitspreken, achteraf het gebruik van de experts lezen. Als klas uitzoeken wat routing
opleverde: de runs publiceren, het werk van anderen bekijken en daarop voortbouwen met
bronvermelding.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 3.1, 3.4, 3.5 | gate, highway, mixture of experts, de control | `01_gates_highways_moe` |
| 3.2, 3.3, 3.10 | scaling laws, gewichten en data samen, eigen curves | `02_scaling_curves` |
| 3.6 | zoekalgoritmes | `03_what_does_the_algorithm_know` |
| 3.7, 3.8, 3.9 | afvalrace, learning rate scheduler, Pareto front | `04_search` |
| 3.11, 3.12 | de klasopdracht | `05_exercise`, `exercises/` |

Train, validatie en test (1.9) en de spreiding tussen runs (1.10) zijn voorkennis uit les
1: elke vergelijking in deze les kiest op validatie en rapporteert op test.

Optioneel: `8_extras/swarms` laat zoeken met een zwerm zien, voor problemen zonder
bruikbare gradient.
