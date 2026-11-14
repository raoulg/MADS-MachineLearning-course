# Les 1 — Representatie en head

Lees uit *Understanding Deep Learning* hoofdstuk 2-4 (p. 17-55).

De les begint bij de cache: een bevroren, voorgetraind model heeft de vectorruimte al
geordend, en jij traint alleen het goedkope stuk erbovenop. Alles daarna — tensors,
lagen, trainen, zoeken — gebeurt op die vectoren. Kernels, state en gates zijn het
onderwerp van les 2.

Sommige leerdoelen haal je beter uit een notebook, sommige uit de tekst, sommige uit de
les. Maak je geen zorgen als je ze niet allemaal meteen hebt; daar is de les voor.

## De student begrijpt

1.1 Dat een tensor twee niveaus heeft: het *formele* (rank en shape) en het *semantische*
(wat de assen betekenen). Kent voorbeelden tot en met 4D — één item als `(D)`, een batch
`(B, D)`, een afbeelding `(C, H, W)`, een batch afbeeldingen `(B, C, H, W)` — en kan
uitleggen dat een batch `(B, D)` en een gechunkt item `(C, D)` dezelfde shape hebben met
een andere betekenis, en dat torch dat verschil niet ziet.

1.2 Dat een voorgetraind model de vectorruimte al ordent zonder ooit op jouw taak getraind
te zijn: vergelijkbare items liggen dicht bij elkaar, zodat 1-nearest-neighbour zonder
training al classificeert. Kan noemen wanneer dat niet opgaat: een domein ver van de
pretraining-data, of een modaliteit zonder goede encoder.

1.3 De economie van de cache: embedden is duur en gebeurt één keer, een head trainen kost
seconden en gebeurt honderden keren. Een search vermenigvuldigt de kosten van de head,
niet die van de embedding.

1.4 Dat de keuze van de encoder de score meer beweegt dan de architectuur van de head, en
dat welke encoder wint afhangt van waarop hij getraind is en van jouw taak. Kan uitleggen
wat dat betekent voor waar je je tijd aan besteedt, en waarom dezelfde head op een image
embedding en op een text embedding werkt.

1.5 Waarom een stapel lineaire lagen zonder activatie samenvalt tot één lineaire
afbeelding, en wat de nonlineariteit daaraan verandert: elke extra unit voegt een lineair
stuk toe, dus meer units is meer capaciteit. Weet wat de universal approximation theorem
zegt en wat niet (hoeveel units, of training ze vindt, of het generaliseert).

1.6 Wat één trainingsstap doet: items stapelen tot een batch, voorspellen, de fout meten
met een loss, en de gewichten bijstellen. De gradient zegt per gewicht welke kant op, de
optimizer bepaalt hoe ver.

1.7 Waarom loss en metric niet hetzelfde zijn: de loss is er voor de optimizer en moet
differentieerbaar zijn, de metric is er voor jou en moet interpreteerbaar zijn. Weet
wanneer klassen met elkaar concurreren en wanneer niet, en dat dat de loss bepaalt.

1.8 Waarom we train-, valid- en testsplits maken, wat overfitting is en hoe regularisatie
daarbij helpt — en waarom regularisatie niets oplevert als het plafond in de representatie
zit en niet in de head.

1.9 Wanneer een verschil een resultaat is: een marge binnen de eigen seed-spreiding is
geen resultaat, het maximum van een search is geen eerlijke vergelijking met het
gemiddelde van een baseline, en "het simpelere model won" is een volwaardige uitkomst.

1.10 Hoe deep learning past in de geschiedenis van AI, en welke soorten leerproblemen er
zijn (classificatie, regressie, reinforcement, (semi-)unsupervised).

## De student kan

1.11 Een architectuur lezen en tekenen als een keten van tensor-shapes, met bij elke pijl
de shape, van `(B, D)` tot `(B, K)` — ook als twee streams elk een eigen tak krijgen en
via concatenatie in één head samenkomen. Kan uit de `metadata.json` van een cache
voorspellen welke shape een batch heeft en welke componenten er tussen de DataLoader en
de loss nodig zijn, vóórdat er code draait.

1.12 Een baseline neerzetten vóór er iets gezocht wordt: een cache van de hub laden, de
1-nearest-neighbour-vloer en een lineaire head met seed-spreiding meten, en die runs in
MLflow loggen als vergelijkingspunt voor alles wat volgt.

1.13 Een zoekruimte ontwerpen en verdedigen: welke assen erin, welke eruit, welke ranges,
en wat je verwacht en waarom. Kan het resultaat van een ray.tune search per as lezen —
ook als de search niets vindt — en draait de winnaar over meerdere seeds opnieuw voordat
het een winst heet.

1.14 Het eigen werk publiceren naar de runs-repository van de klas
(`docs/submit-runs.md`), het werk van anderen bekijken in het MLflow-overzicht waarmee de
volgende les opent, en daarop voortbouwen met bronvermelding.

## Materiaal

| leerdoel | onderwerp | boek | notebook |
|---|---|---|---|
| 1.1 | tensors, shape en betekenis, batch en chunk | 30-34 | `01b_tensors`, `01_the_cache` |
| 1.2, 1.3, 1.4 | de cache, de kosten, de encoderkeuze | les | `01_the_cache`, `04_search` |
| 1.5 | lineaire collapse, nonlineariteit, capaciteit | 25-30, 35, 41-50 | `02_nonlinearity` |
| 1.6, 1.7 | de trainingsstap, loss en metric | 22-23, 77-80, 91, hfst 5 | `03_training` |
| 1.8 | splits, overfitting, regularisatie | 118-120, 138, 147-149 | `03_training`, `05_exercises` |
| 1.9 | seeds, eerlijke vergelijking, negatieve resultaten | les | `04_search`, `05_exercises` |
| 1.10 | geschiedenis van AI, soorten leerproblemen | 3-5 | presentatie les 1 |
| 1.11 | architecturen lezen met shapes, fusie | les | `04_search`, `05_exercises` |
| 1.12 | de baseline | les | `03_training`, `04_search`, `05_exercises` |
| 1.13 | de zoekruimte en de search | les | `04_search`, `05_exercises` |
| 1.14 | publiceren en voortbouwen | les | `05_exercises`, `docs/submit-runs.md` |

Optioneel: `01c_build_your_own_cache` laat zien hoe een cache gemaakt wordt. Je hebt het
niet nodig om de rest van de cursus te volgen — de caches staan klaar op de hub — maar
als je zelf een dataset wilt toevoegen begint het daar.

## Resources

- Intelligentie: https://youtu.be/JxbeMbEp-Ak
- History of AI: https://www.youtube.com/watch?v=0aMM3Q5t7p0
- Semantic vector spaces I: https://www.youtube.com/watch?v=G5wlnlCWfVk
- Semantic vector spaces II: https://www.youtube.com/watch?v=9kBW9Nhd0Zw
- What is a neural network: https://www.youtube.com/watch?v=aircAruvnKk
