# Les 7 — Generalisten en hun geheugen

In les 1 trainde je een kleine head op een bevroren encoder. Een generalist is een groot
voorgetraind model, in deze les een LLM, dat jouw taak uitvoert zonder dat jij iets
traint. Je bepaalt wat je hem meegeeft en voor welke items je hem aanroept.

Alles in deze les draait op een gepubliceerde cache van modeluitvoer: op een CPU,
herhaalbaar en gratis. Wie de hardware heeft kan die cache zelf opnieuw maken met een
lokaal model; nodig is dat niet.

## De student begrijpt

7.1 Wat een generalist is: een groot voorgetraind model, hier een LLM, dat zero-shot op
jouw taak werkt, dus zonder dat jij iets traint. In deze les blijven zijn parameters
bevroren; wat je verandert is de instructie en de voorbeelden die je meegeeft.

7.2 Waarom je een generalist en een getrainde head op score en op kosten vergelijkt: de
generalist kost per item veel rekentijd of geld, de head bijna niets. Kan beide op
dezelfde testset naast elkaar zetten.

7.3 Hoe je de uitvoer van een generatief model beoordeelt. Is er een label, vergelijk
daarmee. Een antwoord dat je niet kunt uitlezen tel je als fout; je laat het niet weg.
Gebruik je een LLM als beoordelaar, controleer die dan eerst op voorbeelden die je zelf
gelabeld hebt.

7.4 Welke gevallen de head aankan: de recall per klasse laat zien op welke klassen de
goedkope head goed genoeg is en op welke niet. Voor die laatste is de generalist bedoeld.

7.5 Wat een cascade is: een router voorspelt per document of het in een klasse valt die
de head aankan. Zo ja, dan antwoordt de head; zo nee, dan gaat het document naar de
generalist. Dit is de routing van les 3 met twee experts, en het doorverwijzen van les 6.
Met de drempel van de router kies je hoeveel documenten naar de generalist gaan, en dus
wat het kost.

7.6 Dat de cache uit les 1 als geheugen kan dienen: bij een nieuw document haal je de
dichtstbijzijnde gelabelde voorbeelden op en geef je die mee aan de generalist. Dat heet
retrieval. Of de juiste voorbeelden worden opgehaald meet je met recall@k, zonder de
generalist aan te roepen.

## De student kan

7.7 Een cascade bouwen van head, router en generalist, en de drempel kiezen uit de
kosten. Vergelijk de router met willekeurig doorsturen bij dezelfde kosten: het verschil
komt van welke documenten hij doorstuurt.

7.8 De generalist een geheugen geven: de dichtstbijzijnde gelabelde uitspraken uit de
cache ophalen, meegeven, en meten wat dat oplevert bij een verschillend aantal
voorbeelden.

7.9 Als klas head, generalist en retrieval combineren op de SCOTUS-uitspraken.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 7.1, 7.2 | de generalist tegenover de head | `03_who_can_the_head_serve`, `04_the_cascade` |
| 7.3 | generatieve uitvoer beoordelen | `02_evaluating_generative_output` |
| 7.4 | welke gevallen de head aankan | `03_who_can_the_head_serve` |
| 7.5, 7.7 | de cascade | `04_the_cascade` |
| 7.6, 7.8 | retrieval als geheugen | les |
| 7.9 | de klasopdracht | les |

Publiceren naar de klas en voortbouwen op het werk van anderen (1.15, 2.12) geldt in elke
les en staat hier niet opnieuw.
