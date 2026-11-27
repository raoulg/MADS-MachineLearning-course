# Les 6 — Onzekerheid: van verdeling naar beslissing

Een puntvoorspelling is een antwoord op een vraag die niemand gesteld heeft. Deze les
gaat over kunnen zeggen hoe zeker je bent, die claim scoren, en ernaar handelen. Ze begint
bij forecasting, waar de breedte van een voorspelling het duidelijkst het eigenlijke
antwoord is, en eindigt bij een beslissing op de hartslagen.

## De student begrijpt

6.1 Hoe een reeks een forecasting-probleem wordt: eerst splitsen in de tijd, dan
windowen, en normaliseren op de statistiek van de trainset alleen. Weet welk naïef model
de baseline is die verslagen moet worden.

6.2 Het verschil tussen een puntvoorspelling en een voorspelde verdeling, en wanneer de
breedte van die verdeling het eigenlijke antwoord is.

6.3 Het verschil tussen aleatorische onzekerheid (de wereld is variabel) en epistemische
onzekerheid (het model weet het niet), en dat meer data alleen de tweede verkleint.

6.4 Dat een verdeling een eigen metric nodig heeft: CRPS scoort de hele verdeling, MAE
alleen het midden.

6.5 Hoe een Gaussische head met NLL getraind wordt en welke aanname je daarmee doet, en
dat een quantile head die aanname niet nodig heeft.

6.6 Wat kalibratie is: een interval van 90% hoort in 90% van de gevallen de waarheid te
bevatten. Weet dat een model scherp én slecht gekalibreerd kan zijn.

6.7 Wat conformal prediction doet: een dekkingsgarantie zonder aanname over de verdeling.
Weet wat je ervoor inlevert, dat de grootte van de set het signaal is, en onder welke
aanname de garantie vervalt.

6.8 Dat een drempel een *beslissing* is en geen modelparameter: bij ongelijke kosten ligt
de beste drempel niet op 0.5, en waar hij wel ligt volgt uit de kosten. Kan de afweging
tussen precision en recall daarmee uitleggen.

6.9 Dat "niet antwoorden" een volwaardige uitkomst is: een model dat weet wanneer het iets
niet weet, kan doorverwijzen.

## De student kan

6.10 Een Gaussische en een quantile head bouwen op een reeks en beide met CRPS afzetten
tegen de naïeve baseline.

6.11 Een kalibratieplot maken en aflezen of de nominale dekking gehaald wordt, conformal
intervallen bouwen op een validatieset en de dekking op de testset controleren.

6.12 Op de hartslagen een drempel kiezen uit een kostenmatrix — een gemiste patiënt tegen
een vals alarm — met doorverwijzen als derde optie, en als klas vergelijken wat dat
oplevert: de runs publiceren, het werk van anderen bekijken en daarop voortbouwen met
bronvermelding.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 6.1 | windowen, splitsen, de naïeve baseline | `00_forecasting_baselines` |
| 6.2, 6.3, 6.4, 6.5, 6.6, 6.10 | verdeling, CRPS, Gaussische en quantile head, kalibratie | `01_probabilistic` |
| 6.7, 6.11 | conformal prediction | `02_conformal` |
| 6.8, 6.9 | kosten, drempel, weigeren | `03_decisions` |
| 6.12 | de klasopdracht | `04_exercise` |
