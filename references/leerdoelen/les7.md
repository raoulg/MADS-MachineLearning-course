# Les 7 — Generalisten die je niet traint, en het geheugen dat je ze geeft

> **Deze les wordt als laatste gebouwd en deze leerdoelen zijn voorlopig.** Elk ander
> hoofdstuk van deze cursus rust op getallen die op deze machine gemeten zijn; les 7 nog
> niet. Doelen waar geen meting onder komt te liggen, sneuvelen.

De hele cursus ging tot nu toe over "bevroren representatie plus een goedkope head". Een
generalistisch model is hetzelfde patroon, één niveau hoger. Je kunt hem niet trainen —
dus zit alle hefboom in *wat je hem voert*.

Alles wat gecijferd wordt draait op een gepubliceerde cache van modeluitvoer: CPU,
deterministisch, gratis. Wie de hardware heeft kan die cache optioneel zelf opnieuw
genereren met een lokaal ~3B-model via ollama, maar dat is nooit verplicht.

## De student begrijpt

7.1 Wat een foundation model is, en wat het betekent dat het zero-shot op jouw taak
werkt zonder dat jij iets getraind hebt.

7.2 Waarom een zero-shot generalist vergeleken moet worden met een getrainde head op
*zowel* accuracy als kosten, en niet alleen op accuracy.

7.3 Wat contaminatie is bij foundation models — dat jouw serie of corpus in de
pretraining-mix kan hebben gezeten — en hoe je dat controleert in plaats van aanneemt.

7.4 De vraag "comprimeert deze encoder, of expandeert hij?", en waarom een geleerde
embedding van een window slechter kan zijn dan het ruwe window.

7.5 Dat een `VectorCache` een retrieval-index *is*, en dat RAG dus hergebruik is van de
infrastructuur uit les 1 en niet nieuwe machinerie.

7.6 Hoe je retrieval meet zonder ook maar één keer een LLM aan te roepen: recall@k, MRR,
nDCG over een index.

7.7 Waarom platte chunk-gebaseerde RAG faalt: geen relaties, geen tijd, geen herkomst —
en welke van die faalwijzen op een kleine corpus aantoonbaar zijn.

7.8 De definitie van een graph, wat een adjacency- en degree-matrix zijn, en een
intuïtief begrip van degree, betweenness en clustering coefficient.

7.9 Dat je met de eigenvectoren van een Laplaciaan een graph kunt clusteren (spectral
clustering), en waarvoor je dat bij een kennisgraaf zou gebruiken.

7.10 Hoe entiteiten en relaties met een LLM uit tekst gehaald worden, **en hoe fout dat
gaat** — en wat die foutmarge doet met de retrieval-kwaliteit erachter.

7.11 Wat hybride retrieval is (vector + graph, vector + BM25, vector + symbolisch filter)
en wanneer het niet helpt.

7.12 Wat geheugen over tijd vereist: wat bewaar je, wat vat je samen, wat vergeet je —
en dat vergeten een ontwerpkeuze met meetbare kosten is, geen bug.

7.13 Wat RAG, LoRA en MCP zijn en waar ze zich tot elkaar verhouden.

7.14 Wat distillatie is: de uitvoer van de generalist als labels, een goedkope head
erop, en hoe groot het gat is dat je gratis kreeg.

7.15 Kan redeneren over AI alignment-uitdagingen.

## De student is in staat om

7.16 Een zero-shot generalist en een zelfgetrainde head op dezelfde taak vergelijken, op
accuracy én kosten, in minstens twee modaliteiten.

7.17 Een retrieval-index bouwen op een bestaande cache en de kwaliteit ervan meten.

7.18 Een kennisgraaf uit een gepubliceerde extractie-cache opbouwen, clusteren, en de
retrieval ervan afzetten tegen platte RAG — met een getal, in beide richtingen eerlijk.

7.19 Een goedkope head distilleren op de uitvoer van een generalist en het prestatiegat
rapporteren.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 7.1-7.4, 7.16 | de generalist, gemeten | `01_foundation_models` |
| 7.5-7.7, 7.17 | retrieval over je eigen cache | nog te bouwen |
| 7.8-7.11, 7.18 | structuur, graaf, hybride retrieval | nog te bouwen |
| 7.12 | geheugen, samenvatten, vergeten | nog te bouwen |
| 7.13-7.15, 7.19 | RAG/LoRA/MCP, distillatie, alignment | nog te bouwen, les |

## Resources

Nog te bepalen bij de bouw van deze les.
