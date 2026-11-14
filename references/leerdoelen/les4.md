# Les 4 — Taal en attention

Lees uit *Understanding Deep Learning* hoofdstuk 12, Transformers.

Een kernel ziet buren, een state loopt een as af. Attention is de derde manier om met een
as om te gaan: elke positie kijkt naar alle andere tegelijk. Het is de kern van de meeste
architecturen van nu, en daarom staat het hier — niet omdat taal het vereist.

Taal brengt wel een eigen probleem mee. In les 1 gaf een afbeelding één vector per item.
Een lang document past niet in een context window en wordt een matrix `(chunks, dim)`, en
daar worden padding, masking en aggregatie echte ontwerpkeuzes.

## De student begrijpt

4.1 Wat attention met een as doet: elke positie krijgt een nieuwe vector, als gewogen som
van alle posities, met gewichten naar relevantie in plaats van naar afstand. De shape
`(B, L, D)` blijft gelijk.

4.2 Hoe attention zich verhoudt tot een kernel en een state: een kernel ziet alleen buren,
een state draagt context stap voor stap mee, attention ziet alles in één keer. Weet dat
attention daardoor geen volgorde kent, en dat positional encoding die volgorde teruggeeft.

4.3 Waarom attention in bijna elke architectuur zit: lange afstand in één stap, en alle
posities parallel. Weet wat het kost: het aantal gewichten groeit met het kwadraat van de
lengte.

4.4 Wat een mask doet: posities die er niet toe mogen doen, zoals padding, krijgen gewicht
nul. Weet dat multi-head attention meerdere van zulke wegingen naast elkaar is.

4.5 Dat tekst via tokens een reeks vectoren wordt, dat subword-tokenization een
middenweg is tussen woorden en letters, en dat tokens een *budget* zijn: het context
window is eindig, en hoeveel tekst erin past hangt af van de tokenizer.

4.6 Wat er gebeurt als een document niet in het context window past: chunken geeft
`(chunks, dim)` per document, en de keuze van chunkgrootte en overlap kost rekentijd en
context.

4.7 Waarom een gewoon gemiddelde over gepadde chunks korte documenten naar de oorsprong
trekt, en wat een masked aggregator daaraan doet.

4.8 Dat twee stromen van representaties — een geleerde embedding en een zelf ontworpen
feature — alleen iets toevoegen als ze *complementaire* informatie dragen. "Meer
parameters" is geen reden.

## De student kan

4.9 Een lang document chunken, embedden via een cache en met padding en masked aggregatie
terugbrengen tot één vector per document, en vooraf de shape op elke stap voorspellen.

4.10 Twee stromen fuseren en de ablatie draaien die laat zien of de fusie iets oplevert:
één variabele tegelijk.

4.11 Een search draaien over de assen chunk, padding en aggregatie, en het resultaat
rapporteren op een split waar de search nooit naar gekeken heeft.

4.12 Als klas een ontwerp maken voor de SCOTUS-uitspraken — lange documenten, weinig
voorbeelden per klasse — en daarin attention inzetten waar het iets oplevert: de runs
publiceren, het werk van anderen bekijken en daarop voortbouwen met bronvermelding.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 4.1, 4.2, 4.3, 4.4 | attention, positional encoding, masking | `01_attention` |
| 4.5 | tokens als budget | `02_tokenization` |
| 4.6, 4.7, 4.9, 4.11 | lange documenten, padding, masked aggregatie, de search | `03_long_documents` |
| 4.8, 4.10 | fusie en de ablatie | `04_composition` |
| 4.12 | de klasopdracht | les |

Generatieve modellen zitten niet in deze les. Les 4 gaat over de representatie van taal;
een generalist gebruiken in plaats van zelf trainen is les 7.

## Resources

- Transformers & attention 1: https://www.youtube.com/watch?v=yGTUuEx3GkA
- Transformers & attention 2: https://www.youtube.com/watch?v=tIvKXrEDMhk
- Optioneel — Attention is all you need: https://arxiv.org/pdf/1706.03762.pdf
- Optioneel — Transformer explainer: https://poloclub.github.io/transformer-explainer/
