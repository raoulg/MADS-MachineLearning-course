# Les 4 — Taal en attention

Lees uit *Understanding Deep Learning* hoofdstuk 12, Transformers.

Een kernel ziet buren, een state loopt een as af. Attention is de derde manier om met een
as om te gaan: elke positie kijkt naar alle andere tegelijk. Attention zit in de meeste
architecturen van nu, ook buiten taal, en daarom staat het hier.

Taal brengt een eigen probleem mee. In les 1 gaf een afbeelding één vector per item.
Een lang document past niet in een context window en wordt een matrix `(chunks, dim)`.
Hoe je chunkt, hoe je aanvult met padding en hoe je de chunks weer samenvoegt zijn dan
ontwerpkeuzes.

## De student begrijpt

4.1 Wat attention met een as doet: elke positie krijgt een nieuwe vector, een gewogen som
van alle posities. Het gewicht hangt af van hoe relevant de andere positie is, niet van
hoe ver weg ze staat. De shape `(B, L, D)` blijft gelijk.

4.2 Hoe attention zich verhoudt tot een kernel en een state: een kernel ziet alleen buren,
een state neemt context stap voor stap mee, attention ziet alle posities in één keer.
Attention kent daardoor geen volgorde; positional encoding voegt de positie toe aan elke
vector.

4.3 Waarom attention in bijna elke architectuur zit: elke positie bereikt elke andere in
één stap, en alle posities worden parallel berekend. Wat het kost: elke positie krijgt een
gewicht naar elke andere, dus rekenwerk en geheugen groeien met het kwadraat van de
lengte. Het aantal parameters hangt niet van de lengte af.

4.4 Wat een mask doet: posities die niet mogen meetellen krijgen gewicht nul. Een padding
mask zet de padding op nul. Een causal mask laat elke positie alleen de posities ervoor
zien. Dat is het verschil tussen een encoder, die de hele tekst in één keer leest, en een
decoder, die steeds het volgende token voorspelt.

4.5 Wat multi-head attention is: meerdere gewogen sommen naast elkaar, elk met eigen
parameters, zodat elke head op iets anders kan letten.

4.6 Dat tekst via tokens een reeks vectoren wordt, en dat subword-tokenization tussen
woorden en letters in zit. Het context window is een maximum aantal tokens; hoeveel tekst
daarin past hangt af van de tokenizer.

4.7 Dat je twee teksten op twee manieren kunt vergelijken. Je kunt ze apart embedden en
de cosine distance nemen: elke tekst gaat één keer door de encoder, en daarna vergelijk je
er zoveel als je wilt. Of je leest ze samen in één context window, zodat attention over
beide teksten loopt (een reranker). Dat is nauwkeuriger, maar elk paar kost een eigen pass
door het model.

4.8 Wat je doet als een document niet in het context window past: chunken en elke chunk
embedden. Dat geeft per document een matrix `(chunks, dim)`. Kleinere chunks of meer
overlap geven meer chunks, en dus meer rekentijd. Ook een groter context window maakt
chunken niet overbodig, want attention kost kwadratisch in de lengte (4.3).

4.9 Waarom padding de aggregatie verstoort: documenten hebben een verschillend aantal
chunks, dus in een batch vul je korte documenten aan met nulvectoren. Een gewoon
gemiddelde telt die nullen mee en trekt korte documenten naar de oorsprong; een masked
mean middelt alleen over de echte chunks.

## De student kan

4.10 Een lang document chunken, embedden via een cache en met padding en een masked
aggregatie terugbrengen tot één vector per document, met chunkgrootte, overlap en
aggregatie als assen van een search.

4.11 Twee stromen fuseren, zoals een zelf ontworpen feature naast een geleerde embedding,
en hetzelfde model ook zonder elk van de stromen draaien. Een tweede stroom voegt alleen
iets toe als hij informatie draagt die de eerste stroom niet heeft.

4.12 Als klas een ontwerp maken voor de SCOTUS-uitspraken: lange documenten met weinig
voorbeelden per klasse, met attention op de plek waar het iets oplevert.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 4.1, 4.2, 4.3, 4.4, 4.5 | attention, positional encoding, padding en causal mask, multi-head | `01_attention` |
| 4.6, 4.7 | tokens, het context window, twee teksten apart of samen | `02_tokenization` |
| 4.8, 4.9, 4.10 | lange documenten, padding, masked aggregatie, de search | `03_long_documents` |
| 4.11 | fusie, en het model zonder één stroom | `04_composition` |
| 4.12 | de klasopdracht | les |

Publiceren naar de klas en voortbouwen op het werk van anderen (1.15, 2.12) geldt in elke
les en staat hier niet opnieuw.

Generatieve modellen zitten niet in deze les. Les 4 gaat over de representatie van taal;
een generalist gebruiken in plaats van zelf trainen is les 7.

## Resources

- Transformers & attention 1: https://www.youtube.com/watch?v=yGTUuEx3GkA
- Transformers & attention 2: https://www.youtube.com/watch?v=tIvKXrEDMhk
- Optioneel — Attention is all you need: https://arxiv.org/pdf/1706.03762.pdf
- Optioneel — Transformer explainer: https://poloclub.github.io/transformer-explainer/
