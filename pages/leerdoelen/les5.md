# Les 5 — Self-supervised: trainen zonder labels

Lees uit *Understanding Deep Learning* hoofdstuk 14, en de blog over I-JEPA:
https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/

In les 1 kreeg je een bevroren encoder en trainde je er een kleine head op. Deze les gaat
over de vraag waar zo'n encoder vandaan komt, en hoe je er zelf een traint voor een domein
dat er geen heeft.

De notatie is die van les 1: `x` is een item, `φ` een augmentatie, `g` de encoder en
`g(x)` de vector in `ℝᵈ`.

## De student begrijpt

5.1 Wat een pretext task is: een taak waarbij het doel uit de data zelf komt, zodat een
encoder zonder labels kan trainen.

5.2 Dat een autoencoder een encoder en een decoder achter elkaar is, met een smalle latent
space ertussen, en dat de loss de uitvoer vergelijkt met de invoer. De latent bewaart wat
nodig is om de pixels terug te tekenen, ook wat voor jouw taak niet uitmaakt, zoals
achtergrond en belichting.

5.3 Hoe contrastief leren werkt: van één item `x` maak je met twee augmentaties twee
views, `φ₁(x)` en `φ₂(x)`. Beide gaan door een encoder, en de loss vraagt dat hun vectoren
gelijk zijn: de vector van de ene view is het doel voor de andere.

5.4 Dat de encoder daardoor ongevoelig wordt voor wat `φ` verandert: `g(φ(x)) ≈ g(x)`.
Verandert `φ` de kleur, dan staat kleur niet meer in de vector. Kies `φ` dus zo dat hij
niets verandert wat jouw taak nodig heeft.

5.5 Wat representation collapse is: de encoder geeft elk item dezelfde vector. Dat is een
triviaal antwoord op "geef twee views dezelfde vector", dus de loss is laag terwijl er
niets geleerd is. Je ziet het aan de spreiding van de vectoren, niet aan de loss.
Methodes verschillen in hoe ze collapse voorkomen: de vectoren van andere items wegduwen,
of een tweede encoder gebruiken die geen gradient krijgt.

5.6 Wat JEPA anders doet: de views ontstaan door maskeren in plaats van augmenteren, en
een predictor voorspelt de vector van het verborgen deel, niet de pixels. Een tweede
encoder, de target encoder, levert die vector.

5.7 Wat pooling weggooit: de plaats. Met de patch-as `(B, P, D)` weet je ook waar iets op
de afbeelding staat; na pooling naar `(B, D)` alleen nog wat. Kan daarmee kiezen tussen
een gepoolde embedding en een embedding met patch-as, voor een taak waarin de plaats
ertoe doet.

5.8 Dat de encoder achter de cache van les 1 zo getraind is: self-supervised, zonder
labels, op heel veel afbeeldingen. Daarom kon je er in les 1 met een kleine head op
verder.

## De student kan

5.9 Drie encoders op dezelfde data trainen (autoencoder, contrastief en supervised) en ze
vergelijken met een linear probe. Dat is de lineaire head op een bevroren encoder uit
les 1: daar hield je de encoder vast en vergeleek je heads, hier houd je de head vast en
vergelijk je encoders.

5.10 Tijdens het trainen van een self-supervised model de spreiding van de vectoren
bijhouden om collapse te zien, en in een search de accuracy van de linear probe als
objective nemen in plaats van de loss.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 5.1, 5.2, 5.9 | pretext task, autoencoder, encoders vergelijken met een linear probe | `01_why_pretext_tasks_work` |
| 5.3, 5.4, 5.5, 5.10 | twee views, augmentatie, collapse, zoeken op de probe | `02_selfsupervised` |
| 5.7 | de patch-as behouden | `03_patch_tokens` |
| 5.6 | I-JEPA | `04_ijepa` |
| 5.8 | waar de encoder van les 1 vandaan komt | les |

## Resources

- Self-supervised learning: https://youtu.be/qiUEgSCyY5o
- I-JEPA: https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/
