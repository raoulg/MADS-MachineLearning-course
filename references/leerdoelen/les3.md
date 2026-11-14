# Les 3 — Wat heeft die capaciteit je opgeleverd?

Lees *Automated Machine Learning* hoofdstuk 1, en uit *Understanding Deep Learning*
hoofdstuk 6, 8 en 9. Die hoofdstukken gaan over het trainingsproces in het algemeen en
geven de theorie onder wat je in les 1 en 2 al deed.

Je hebt nu twee lessen gezocht. Deze les stelt de vraag die bij elke meting hoort: je
hebt parameters, data of rekentijd toegevoegd — laat zien wat je ervoor terugkreeg. Omdat
een run seconden kost, teken je de curve zelf in plaats van erover te lezen. De gate uit
les 2 groeit hier uit tot routing: een netwerk dat per input een pad kiest.

## De student begrijpt

3.1 Wat een capacity-matched control is, en waarom een architectuurclaim zonder zo'n
control niets zegt: "de gate hielp" is geen resultaat als het model ernaast minder
parameters had.

3.2 Wat een scaling curve is: hoe de score beweegt met het aantal parameters, met de
hoeveelheid data en met de rekentijd per run. Weet dat die curves gemeten zijn en niet
afgeleid, dat ze afvlakken, en dat de as waarop je het meest wint per probleem verschilt.

3.3 Dat een gate, een highway en een mixture of experts drie resoluties van hetzelfde idee
zijn: een gate weegt een signaal, een highway mengt een blok met de skip eromheen, en een
mixture of experts kiest per input een pad. Kan zeggen wat elk met de shape doet: niets.

3.4 Dat het aantal *getrainde* en het aantal *gebruikte* parameters per input twee
grootheden zijn, en dat sparse routing ze loskoppelt: specialisten en een generalist in
één netwerk. Weet wat router collapse is en waarom een router zonder load balancing erin
loopt.

3.5 Wat een search-algoritme weet: grid weet niets, random trekt in plaats van op te
sommen, TPE leert van eerdere trials, een populatie houdt goede kandidaten vast. Kan bij
elk zeggen welke tekortkoming van het vorige het oplost.

3.6 De budgetvraag: bij een vast budget liever veel korte trials of weinig lange. Weet
dat een scheduler als ASHA slechte trials vroeg stopt, en dat een zoekruimte met elke as
erbij sneller groeit dan je budget.

3.7 Wat een learning-rate scheduler doet: eerst grof, dan fijn — de kettingzaag voor de
vorm, de guts voor het detail. Herkent hetzelfde idee in ASHA: besteed de eerste moeite
breed en goedkoop, de laatste smal en precies.

3.8 Dat de score niet de enige as is: kan een Pareto-front over score, parameters en
latency lezen, en uitleggen waarom "de beste trial" zonder tweede as een onvolledige vraag
is.

3.9 Welk getal je mag rapporteren: dat van één run op data waar de search nooit naar
gekeken heeft. Weet dat een vlak search-resultaat informatie is en geen mislukking.

## De student kan

3.10 Een capaciteitscurve meten en tekenen over minstens twee assen (parameters, data of
rekentijd), en daaruit een conclusie trekken die groter is dan de spreiding tussen runs.

3.11 Een search opzetten met een budget: welke assen voorrang krijgen, welk algoritme en
welke scheduler, en hoeveel trials van hoeveel epochs.

3.12 Routing toevoegen aan een eigen ontwerp op de hartslagen, met een capacity-matched
control ernaast, en als klas uitzoeken wat die capaciteit opleverde: de runs publiceren,
het werk van anderen bekijken en daarop voortbouwen met bronvermelding.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 3.1, 3.3, 3.4 | gate, highway, mixture of experts, de control | `01_gates_highways_moe` |
| 3.2, 3.10 | scaling curves over parameters, data en rekentijd | `02_scaling_curves` |
| 3.5, 3.9 | wat een algoritme weet, het getal dat je rapporteert | `03_what_does_the_algorithm_know` |
| 3.6, 3.8, 3.11 | budget, scheduler, Pareto, de search | `04_search` |
| 3.7 | learning-rate scheduler | presentatie les 3 |
| 3.12 | de klasopdracht | `05_exercise` |

Optioneel: `8_extras/swarms` laat zoeken met een zwerm zien, voor problemen zonder
bruikbare gradient.
