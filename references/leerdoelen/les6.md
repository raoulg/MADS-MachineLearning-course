# Les 6 — Onzekerheid: modelleren, scoren, en ermee beslissen

Een puntvoorspelling is een antwoord op een vraag die niemand gesteld heeft. Deze les
gaat over kunnen zeggen hóe zeker je bent, die claim scoren, en er vervolgens naar
handelen. Hij loopt over beide helften van de cursus heen, omdat dezelfde instrumenten
werken op een voorspelde verdeling en op een stochastische generator.

## De student begrijpt

6.1 Het verschil tussen een puntvoorspelling en een voorspelde verdeling, en wanneer de
breedte van die verdeling het eigenlijke antwoord is.

6.2 Het verschil tussen aleatorische onzekerheid (de wereld is variabel) en epistemische
onzekerheid (het model weet het niet), en dat meer data alleen de tweede verkleint.

6.3 Wat CRPS is en waarom je een verdeling niet met MAE kunt scoren.

6.4 Hoe een Gaussische head met NLL getraind wordt, en welke aanname je daarmee doet.

6.5 Hoe een quantile head werkt, en waarom die de Gaussische aanname niet nodig heeft.

6.6 Wat kalibratie is, hoe je het meet (coverage tegen nominaal niveau), en dat een model
scherp én slecht gekalibreerd kan zijn.

6.7 Wat conformal prediction doet: een dekkingsgarantie zonder aanname over de verdeling,
en wat je ervoor inlevert.

6.8 De precision-recall trade-off, de confusion matrix, en de ethische problemen die aan
die trade-off hangen.

6.9 Dat een drempel een *beslissing* is en geen modelparameter: bij asymmetrische kosten
ligt de optimale drempel niet op 0.5, en waar hij wel ligt volgt uit de kosten.

6.10 Hoe je van een verdeling naar een beslissing komt (newsvendor-achtige problemen), en
dat "niet antwoorden" een legitieme uitkomst is.

6.11 Dat het trekken van *k* samples uit een stochastische generator een Monte
Carlo-schatting is, dat self-consistency daarmee een ensemble is, en dat "is dit model
goed gekalibreerd?" dezelfde vraag is als bij de Gaussische head — met dezelfde
instrumenten te beantwoorden.

6.12 Het onderscheid tussen willekeur *door constructie* (sampling temperature) en
onzekerheid *over de wereld* (de spreiding van een voorspelling). Zelfde gereedschap,
ander object.

## De student is in staat om

6.13 Een Gaussische en een quantile head bouwen op een timeseries en beide tegen een
gratis baseline scoren met CRPS.

6.14 Een kalibratieplot maken en aflezen of de nominale dekking gehaald wordt.

6.15 Conformal intervallen construeren op een validatieset en de dekking op de testset
verifiëren.

6.16 Een drempel kiezen op basis van een kostenmatrix in plaats van op basis van accuracy,
en het verschil in verwachte kosten rapporteren.

6.17 Uit een gepubliceerde cache van *k* gesamplede generaties de spreiding, de
self-consistency-uitkomst en de kalibratie van de zelfgerapporteerde zekerheid berekenen.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 6.1-6.6, 6.13, 6.14 | CRPS, Gaussische en quantile heads, kalibratie | `01_probabilistic` |
| 6.7, 6.15 | conformal prediction | nog te bouwen |
| 6.8-6.10, 6.16 | metrics, drempels, kosten, beslissen | nog te bouwen |
| 6.11, 6.12, 6.17 | sampling als verdeling | nog te bouwen |

De serie die je in les 2 koos is hier de natuurlijke plek om verder te werken: daar
beoordeelde je of een puntvoorspelling volstond, hier bouw je de machinerie voor het
geval dat hij dat niet deed.

## Resources

Nog te bepalen bij de bouw van deze les.
