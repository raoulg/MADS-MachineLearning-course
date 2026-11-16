# Les 7 — Generalisten en hun geheugen

De hele cursus ging over "bevroren representatie plus een goedkope head". Een generalist
is hetzelfde patroon, één niveau hoger. Je traint hem niet, dus zit alle hefboom in wat je
hem voert en in wanneer je hem inzet.

Alles in deze les draait op een gepubliceerde cache van modeluitvoer: op een CPU,
herhaalbaar en gratis. Wie de hardware heeft kan die cache zelf opnieuw maken met een
lokaal model; nodig is dat niet.

## De student begrijpt

7.1 Wat een generalist is: een model dat zero-shot op jouw taak werkt zonder dat jij iets
getraind hebt. Weet dat je hem niet verbetert door te trainen, wel door wat je hem
meegeeft.

7.2 Waarom je een generalist met een getrainde head vergelijkt op score *en* kosten: de
generalist is duur per item, de head bijna gratis.

7.3 Hoe je uitvoer van een generatief model beoordeelt: waar een goed antwoord bestaat
gebruik je dat, uitvoer die je niet kunt lezen is ook een resultaat, en een model dat als
beoordelaar dient, moet zelf beoordeeld worden.

7.4 Voor wie de head volstaat: de recall per klasse zegt welke gevallen de goedkope head
aankan en welke niet, en dat is de beslissing over de route.

7.5 Wat een cascade is: een router stuurt de makkelijke gevallen naar de head en de
moeilijke naar de generalist. Weet dat de drempel van de router een knop is die je tegen
echte kosten afregelt, en dat een router zijn plek moet verdienen tegen een control.

7.6 Dat de cache uit les 1 een geheugen is: de dichtstbijzijnde gelabelde voorbeelden
ophalen en aan de generalist meegeven is retrieval, en dat is hergebruik van wat je al
had. Weet dat je de retrieval kunt meten zonder de generalist aan te roepen.

## De student kan

7.7 Een zero-shot generalist en een zelf getrainde head op dezelfde taak vergelijken, op
score en op kosten.

7.8 Een cascade bouwen van head, router en generalist, de drempel kiezen uit de kosten,
en met een control laten zien dat de router iets oplevert.

7.9 De generalist een geheugen geven: de dichtstbijzijnde gelabelde uitspraken uit de
cache ophalen, meegeven, en meten wat dat oplevert.

7.10 Als klas head, generalist en retrieval combineren op de SCOTUS-uitspraken: de runs
publiceren, het werk van anderen bekijken en daarop voortbouwen met bronvermelding.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 7.1, 7.2, 7.7 | de generalist tegenover de head | `03_who_can_the_head_serve`, `04_the_cascade` |
| 7.3 | generatieve uitvoer beoordelen | `02_evaluating_generative_output` |
| 7.4 | voor wie de head volstaat | `03_who_can_the_head_serve` |
| 7.5, 7.8 | de cascade | `04_the_cascade` |
| 7.6, 7.9 | retrieval als geheugen | les |
| 7.10 | de klasopdracht | les |
