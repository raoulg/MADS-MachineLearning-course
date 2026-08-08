# Les 2 — Timeseries

Het boek behandelt geen RNNs, dus het materiaal komt uit de lessen, de notebooks en de
videos hieronder.

Deze les is bewust de tegenhanger van les 1. Daar was een bevroren encoder het hele
verhaal; hier is er geen encoder om te bevriezen die de moeite waard is, en train je
weer end-to-end. Begrijpen *wanneer* de aanpak uit les 1 niet opgaat is net zo belangrijk
als hem kunnen toepassen.

## De student begrijpt

2.1 Wat een window en een horizon zijn, en hoe windowing bij timeseries werkt en waarom
dat relevant is.

2.2 Wat belangrijk is bij datapreparatie om data leakage te voorkomen — inclusief het
geval waarin de splits van een dataset *prefixes van elkaar* zijn en dus niet disjunct,
zoals bij de GluonTS/Monash-conventie.

2.3 Hoe een naive model werkt en wat de motivatie erachter is (MASE), en waarom je die
baseline vaststelt vóór je iets anders traint.

2.4 Wat de motivatie is om RNNs te gebruiken.

2.5 Hoe een simpele RNN werkt (hidden state).

2.6 Waarin een GRU en LSTM verschillen van een RNN (gates), wat de functies van een gate
zijn (remember, forget, iets ertussenin) en hoe een gate werkt (met een hadamard-product).

2.7 Wat de voor- en nadelen van LSTM vs GRU vs RNN zijn, en waarom een lineair model op
veel series alsnog wint.

2.8 Hoe 1D convoluties werken bij timeseries, en wat het receptieve veld van een stack
convoluties is.

2.9 Hoe je het spectrum van een serie leest (FFT, autocorrelatie) en waarom je de periode
*meet* in plaats van hem aan te nemen.

2.10 Wat trend/seasonality-decompositie is, en wat het verschil is tussen een feature die
je *aan het model geeft* en een basis waarin het model *mag antwoorden*.

2.11 Hoe N-BEATS werkt: stacks, een backcast/forecast-splitsing, en een polynomiale of
Fourier-basis als constraint op wat het model mag zeggen.

2.12 Dat een inductive bias een *weddenschap over de structuur van de data* is en geen
gratis accuracy — en kan benoemen onder welke voorwaarde de weddenschap uitbetaalt.

2.13 Waarom vectorisatie (les 1) hier niet opgaat: er is geen algemene bevroren
timeseries-encoder die de moeite waard is, en een geleerde embedding van een window kan
slechter zijn dan het ruwe window.

## De student is in staat om

2.14 Een dataset windowen, splitsen en schalen zonder leakage.

2.15 Een configureerbare RNN/GRU/LSTM bouwen en vergelijken met een lineaire baseline.

2.16 Een N-BEATS-model bouwen uit basis-componenten en de decompositie ervan uitlezen.

2.17 Een serie kiezen uit het Monash-archief, de hele toolkit erop draaien, en daarover
rapporteren — inclusief minstens één negatief resultaat dat de student zelf geproduceerd
heeft en verdedigt.

2.18 Beoordelen of een puntvoorspelling voor een gegeven serie überhaupt het juiste
antwoord is, met een buur-plot als bewijs (de machinerie daarvoor komt in les 6).

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 2.1, 2.2, 2.14 | windowing, splits, leakage | `01_windowing` |
| 2.3 | naive baseline, MASE | `02_basemodels` |
| 2.4-2.8, 2.15 | RNN/GRU/LSTM, 1D convoluties, lineaire baseline | `02_basemodels` |
| 2.9-2.12, 2.16 | spectrum, decompositie, N-BEATS | `03_decomposition_and_nbeats` |
| 2.13 | de grens van de les-1-aanpak | les, `03_decomposition_and_nbeats` |
| 2.17, 2.18 | eigen serie, eigen negatief resultaat | `04_exercise` |

Datasets in deze les: `sunspot` (dagelijks, 1848-2020) en `solar_10_minutes`. De eerste
ken je al uit MADS-DAV, waar de decompositie ervan vastliep op de 11-jarige cyclus; deze
les is de overdracht van die openstaande vraag.

## Resources

Bekijk de eerste twee videos:

- https://youtu.be/LHXXI4-IEns
- https://youtu.be/8HyCNIVRbSU

De derde is van StatQuest, als je die stijl prettig vindt. De vierde legt het concept op
een andere manier uit — als die je in de war brengt, houd het bij de eerste drie.

- https://youtu.be/YCzL96nL7j0
- https://youtu.be/WCUNPb-5EYI
