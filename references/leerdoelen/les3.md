# Les 3 — Wat heeft die capaciteit je opgeleverd?

Lees *Automated Machine Learning* hoofdstuk 1, en uit *Understanding Deep Learning*
hoofdstuk 6, 8 en 9. Die hoofdstukken gaan over het trainingsproces in het algemeen en
geven de theoretische achtergrond onder wat je in les 1 en 2 al aan het doen was.

Je hebt ray.tune nu twee lessen gebruikt. Deze les gaat niet over het gereedschap maar
over de vraag die elke meting in deze cursus dezelfde kant op beantwoordt: **je hebt
parameters toegevoegd — laat zien dat je er iets voor terugkreeg.** Omdat een trainingsrun
seconden kost, kun je de curve zelf tekenen in plaats van erover te lezen.

## De student begrijpt

3.1 Wat een capacity-matched control is, en waarom een architectuurclaim zonder zo'n
control betekenisloos is: "de gate hielp" zegt niets als de vergelijking een model met
minder parameters was.

3.2 Wat scaling laws zijn: hoe accuracy zich verhoudt tot parameters, tot hoeveelheid
data, en tot compute per run — en dat die curves empirisch zijn, niet gepostuleerd.

3.3 Dat het aantal *getrainde* parameters en het aantal *gebruikte* parameters per input
verschillende grootheden zijn, en waarom sparse routing die twee loskoppelt.

3.4 Hoe een gate, een highway en een mixture-of-experts werken, dat het drie resoluties
van hetzelfde idee zijn, en waar ze vandaan komen (GRU, Highway Networks, Jacobs & Jordan,
Switch Transformer).

3.5 Wat router collapse is en waarom een sparse MoE zonder load-balancing loss erin kan
lopen.

3.6 Wat de curse of dimensionality is met betrekking tot de searchspace.

3.7 Wat een search-algoritme weet dat random search niet weet: grid → random → TPE →
populatiemethodes (PSO, differential evolution, PBT) → multi-fidelity schedulers, elk
gemotiveerd door een tekortkoming van de vorige.

3.8 Hoe bayesian search en hyperband/ASHA werken, en welke vraag ASHA beantwoordt: bij
een vast budget, liever 50 trials van 10 epochs of 10 trials van 50?

3.9 Wat een learning rate scheduler is, hoe je herkent dat je er een nodig hebt, en kent
er een aantal (cosine warm-up, reduce on plateau).

3.10 Dat accuracy niet de enige as is: kan een Pareto-front over accuracy, parameters en
latency lezen en uitleggen waarom "de beste trial" zonder tweede as een onvolledige vraag
is.

3.11 De voor- en nadelen van swarm-algoritmes voor optimalisatie, en herkent situaties
waarin gradient-based methodes falen.

3.12 Dat een vlak search-resultaat informatie is en geen mislukking.

## De student is in staat om

3.13 Redeneren over de grootte van de hyperparameterruimte, daar afwegingen in maken en
prioriteiten stellen — lossfuncties, learning rate, units, aantal lagen, mechanisme —
en vooraf opschrijven welke as hij verwacht dat ertoe doet.

3.14 Een python-script bouwen dat een configureerbaar vectormesh-model via Ray hypertuned.

3.15 Een capaciteitscurve meten en plotten over minstens twee assen (parameters, data,
of compute) en daaruit een conclusie trekken die de seed-ruis overleeft.

3.16 Een afweging maken tussen de verschillende manieren om een model te monitoren
(tensorboard, mlflow, ray).

3.17 Een resultaat rapporteren met een `+/-` uit minstens 3 seeds, en herkennen wanneer
het aantal seeds te klein was om de conclusie te dragen.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 3.1, 3.4, 3.5 | gate / highway / MoE, capacity-matched control | `05_gates_highways_moe` |
| 3.2, 3.3, 3.15 | scaling curves, params trained vs used | les, `06_search` |
| 3.6-3.9 | searchruimte, algoritmes, schedulers | `01_hypertuner`, `02_ray`, les |
| 3.10 | multi-objective, Pareto | `06_search` |
| 3.11 | swarm optimizers | `8_extras/swarms`, les |
| 3.12, 3.17 | vlakke resultaten, seeds | `06_search` |
| 3.13, 3.14, 3.16 | search opzetten en rapporteren | `hypertune.py`, exercises |

De search draait op `solar_10_minutes` — de serie uit les 2 waar N-BEATS wél won. Als je
in les 2 een eigen serie hebt gekozen, is dat de interessantere plek om deze search op
los te laten.

## Resources

Nog te bepalen bij de herbouw van deze les.
