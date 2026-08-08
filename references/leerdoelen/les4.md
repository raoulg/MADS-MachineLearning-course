# Les 4 — Taal

Lees uit *Understanding Deep Learning* hoofdstuk 12, Transformers.

In les 1 gaf een afbeelding één vector per item, dus bleef de tensorladder kort:
(D) → (B, D) → (B, K). Tekst doet dat niet. Een lang document past niet in een
context window van 512 tokens en wordt een (chunks, dim)-matrix — en dáár worden padding,
masking en aggregatie echte ontwerpkeuzes in plaats van vocabulaire.

## De student begrijpt

4.1 Wat een word embedding is en wat de motivatie is ten opzichte van one-hot-encoding.

4.2 Wat tokenization is, welke soorten er zijn (word, subword, char) en hoe BPE daarin
past.

4.3 Dat tokens een *budget* zijn: het context window is eindig, en hoeveel tekst erin
past hangt af van de tokenizer, niet alleen van het aantal woorden.

4.4 Wat een semantische vectorruimte is.

4.5 De motivatie achter attention (ten opzichte van convoluties en recurrentie).

4.6 Hoe het reweighing van word-embeddings werkt met behulp van attention.

4.7 Waarom scaling en masking nodig zijn.

4.8 Wat multihead attention is.

4.9 Wat positional encoding is, en waarom dat behulpzaam kan zijn.

4.10 Hoe attention behulpzaam is bij timeseries in het algemeen en NLP in het bijzonder.

4.11 Kent het verschil in dimensionaliteit van tensors (2D, 3D, 4D) voor de diverse lagen
(Dense, 1D en 2D Convolution, MaxPool, RNN/GRU/LSTM, Attention, Activation, Flatten) en
hoe deze met elkaar te combineren zijn.

4.12 Wat er gebeurt als een document niet in het context window past: chunken, en wat de
keuze van chunkgrootte en overlap kost.

4.13 Het verschil tussen fixed en dynamic padding, en waarom een gewone mean over gepadde
chunks korte documenten naar de oorsprong trekt — en wat een masked aggregator daaraan doet.

4.14 Dat twee representatiestromen (bijvoorbeeld een geleerde embedding en een ontworpen
regex-feature) alleen iets toevoegen als ze *complementaire* informatie dragen, en dat
"meer parameters" geen reden is.

## De student is in staat om

4.15 Een attention-laag zelf implementeren en op een chunked document toepassen.

4.16 Een preprocessor voor NLP maken (punctuation, lowercase, spaces, xml strippen) en
een tokenizer trainen.

4.17 Een lang document chunken, embedden via een cache, en met padding + masked
aggregatie tot één vector per document terugbrengen.

4.18 Twee representatiestromen fuseren met `Parallel` + `Concatenate`, en de ablatie
draaien die aantoont of de fusie iets oplevert — één variabele tegelijk.

4.19 Een ray.tune search draaien over de chunk-, padding- en aggregatie-assen, en het
resultaat rapporteren op een split waar de search nooit naar gekeken heeft.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 4.5-4.10, 4.15 | attention, positional encoding | `01_reweighing` |
| 4.1-4.3, 4.16 | tokenization, BPE, tokens als budget | `02_tokenization` |
| 4.4, 4.11 | semantische vectorruimte, tensordimensies | les, alle lessen tot nu toe |
| 4.12, 4.13, 4.17, 4.19 | lange documenten, padding, masking, search | `03_long_documents` |
| 4.14, 4.18 | compositie en de eerlijke ablatie | `04_composition` |

Generatieve modellen zitten *niet* in deze les. Les 4 gaat over de representatie van taal;
het alternatief waarbij je een generalist gebruikt in plaats van zelf te trainen is les 7.

## Resources

Bekijk:

- https://www.youtube.com/watch?v=yGTUuEx3GkA
- https://www.youtube.com/watch?v=tIvKXrEDMhk

Optioneel:

- Attention is all you need: https://arxiv.org/pdf/1706.03762.pdf
- Transformers visually explained: https://poloclub.github.io/transformer-explainer/
