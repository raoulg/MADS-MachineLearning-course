# Les 6 — Onzekerheid: van verdeling naar beslissing

Een model dat één getal voorspelt zegt niet hoe zeker het is. Deze les gaat over een
verdeling voorspellen, meten of die klopt, en er een beslissing op nemen. Ze begint bij
forecasting, waar je de breedte van een voorspelling goed kunt zien, en eindigt bij een
beslissing op de hartslagen.

## De student begrijpt

6.1 Hoe een reeks een forecasting-probleem wordt: eerst splitsen in de tijd, dan
windowen, en normaliseren met het gemiddelde en de standaarddeviatie van de trainset. De
baseline is een naïef model, zoals de waarde van een dag eerder herhalen.

6.2 Het verschil tussen een puntvoorspelling en een voorspelde verdeling. Gaat de vraag
over een grens (komt het verbruik boven de capaciteit?), dan heb je de kans nodig dat de
waarde erboven komt, en die geeft alleen een verdeling.

6.3 Het verschil tussen aleatorische onzekerheid (de wereld is variabel) en epistemische
onzekerheid (het model weet het niet), en dat meer data alleen de tweede verkleint.

6.4 Dat een verdeling een eigen metric nodig heeft. De mean absolute error (MAE) kijkt
alleen naar het midden van de voorspelling. De continuous ranked probability score (CRPS)
scoort de hele verdeling, en is lager als die smal is rond de waarheid.

6.5 Hoe een Gaussian head werkt: hij voorspelt een gemiddelde μ en een breedte σ, en de
loss is de negative log-likelihood (NLL). Daarmee neem je aan dat de fouten symmetrisch en
klokvormig verdeeld zijn. Een quantile head voorspelt een paar quantiles direct en doet
die aanname niet.

6.6 Wat kalibratie is: een interval van 90% hoort in 90% van de gevallen de waarheid te
bevatten. Voor een classifier is het hetzelfde: van alle gevallen waar hij `p = 0.8` geeft,
hoort 80% positief te zijn. Een model kan smalle intervallen geven en toch slecht
gekalibreerd zijn.

6.7 Wat conformal prediction doet: met een apart gehouden kalibratieset maak je per item
een set van klassen die in 90% van de gevallen de juiste klasse bevat, zonder aanname
over de verdeling. Een grote set betekent dat het model twijfelt. De garantie geldt
gemiddeld, niet per item, en alleen zolang nieuwe data op de kalibratieset lijkt.

6.8 Dat je de drempel waarboven een classifier "positief" zegt zelf kiest. Kost een gemist
geval meer dan een vals alarm, dan hoort de drempel lager dan 0.5 te liggen: meer recall,
minder precision. Waar precies volgt uit de verhouding van die kosten, mits de kansen
gekalibreerd zijn.

6.9 Dat een model ook kan doorverwijzen: met twee drempels zegt het bij een lage kans nee,
bij een hoge kans ja, en stuurt het de gevallen ertussen naar een mens of een duurder
model.

## De student kan

6.10 Een Gaussian en een quantile head bouwen op een reeks en beide met CRPS vergelijken
met de naïeve baseline.

6.11 Een kalibratieplot maken en aflezen of de gevraagde dekking gehaald wordt. Conformal
sets bouwen op een validatieset en de dekking op de testset controleren.

6.12 Op de hartslagen twee drempels op `p(ziek)` kiezen: eronder niets doen, erboven
alarm, ertussen doorverwijzen. Kies ze zo dat de totale kosten het laagst zijn, gegeven
wat een gemiste patiënt, een vals alarm en een doorverwijzing kosten. Als klas vergelijk
je wat verschillende modellen en kosten opleveren.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 6.1 | windowen, splitsen, de naïeve baseline | `00_forecasting_baselines` |
| 6.2, 6.3, 6.4, 6.5, 6.6, 6.10 | verdeling, CRPS, Gaussian en quantile head, kalibratie | `01_probabilistic` |
| 6.7, 6.11 | conformal prediction | `02_conformal` |
| 6.8, 6.9 | kosten, drempel, doorverwijzen | `03_decisions` |
| 6.12 | de klasopdracht | `04_exercise` |

Publiceren naar de klas en voortbouwen op het werk van anderen (1.15, 2.12) geldt in elke
les en staat hier niet opnieuw.
