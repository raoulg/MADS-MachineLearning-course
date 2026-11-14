# Les 5 — Self-supervised: de data als eigen leraar

Lees uit *Understanding Deep Learning* hoofdstuk 14, en de blog over I-JEPA:
https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/

Les 1 gaf je een representatie in handen en liet zien dat die elke andere keuze domineert.
Deze les beantwoordt de vraag die daarop volgt: waar kwam die representatie vandaan, en
zou je er zelf een kunnen maken voor een domein dat er geen heeft?

## De student begrijpt

5.1 Wat een pretext task is: een taak waarvan de data zelf het antwoord levert, zodat een
encoder zonder labels een bruikbare ruimte leert.

5.2 Dat de keuze van de pretext task de keuze is van wat de ruimte *vergeet*:
reconstructie ordent op gelijkenis in de input, contrastief leren ordent op wat gelijk
blijft onder de augmentatie die jij koos.

5.3 Dat een autoencoder een encoder en een decoder achter elkaar is, met een smalle latent
space ertussen, en dat de loss de input zelf als doel heeft.

5.4 Hoe twee views van hetzelfde item door één encoder met gedeelde gewichten gaan, en
dat de augmentatie de enige creatieve keuze is: een augmentatie die de taak kapotmaakt,
levert een ruimte op die de taak vergeet.

5.5 Wat representation collapse is — alle items op dezelfde vector — hoe je het ziet, en
waarom een dalende loss niets bewijst.

5.6 Dat een lineaire probe het meetinstrument van deze les is: de kwaliteit van een ruimte
lees je niet af aan haar meetkunde, wel aan wat een lineaire head eruit haalt.

5.7 Wat JEPA anders doet dan een autoencoder: voorspellen in de *latente* ruimte in plaats
van in pixels, met views die door maskeren ontstaan in plaats van door augmenteren.

5.8 Wat pooling weggooit: de plaats. Weet dat het behouden van de patch-as het verschil
is tussen weten *wat* er op een afbeelding staat en weten *waar*.

5.9 Dat een geleerde representatie een halffabrikaat is: de bevroren encoder en de cache
van les 1 zijn het resultaat van precies zo'n training.

## De student kan

5.10 Drie ruimtes op dezelfde data bouwen — autoencoder, contrastief en supervised — en ze
met dezelfde lineaire probe tegen elkaar afzetten.

5.11 Een self-supervised model trainen met een collapse-detector aan de meting gekoppeld,
en zoeken op die meting in plaats van op de loss.

5.12 Kiezen tussen een gepoolde embedding en een embedding met patch-as, voor een taak
waarin de plaats ertoe doet.

## Materiaal

| leerdoel | onderwerp | notebook |
|---|---|---|
| 5.1, 5.2, 5.3, 5.6, 5.10 | waarom een pretext task werkt, de lineaire probe | `01_why_pretext_tasks_work` |
| 5.4, 5.5, 5.11 | twee views, collapse, zoeken op de meting | `02_selfsupervised` |
| 5.8, 5.12 | de patch-as behouden | `03_patch_tokens` |
| 5.7 | I-JEPA | `04_ijepa` |
| 5.9 | de representatie als halffabrikaat | les |

## Resources

- Self-supervised learning: https://youtu.be/qiUEgSCyY5o
- I-JEPA: https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/
