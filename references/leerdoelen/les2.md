# Les 2 — Kernels, state en gates

Lees uit *Understanding Deep Learning* hoofdstuk 10 (convoluties, p. 161-174) en het
begin van hoofdstuk 11 (residual connections, p. 186-191). Het boek behandelt geen
recurrente netwerken; state en gates komen uit de les, de notebooks en de videos
hieronder.

In les 1 deed een bevroren encoder het zware werk. Deze les werkt op ruwe hartslagen,
waar geen encoder voor klaarligt, en geeft je de blokken om zelf een flow te ontwerpen:
een kernel die scant, een state die onthoudt, een gate die doorlaat, en een skip die
eromheen loopt. Hoe die blokken er vanbinnen uitzien weet je agent; welke je kiest en
waarom is aan jou.

Vanaf deze les werk je als klas: je publiceert je runs, bekijkt die van anderen en bouwt
daarop voort.

## De student begrijpt

2.1 Wat een skip is: een route om een blok heen die bij de uitkomst wordt opgeteld,
zonder de shape te veranderen. Weet dat de residual connection voor image-modellen is
uitgevonden en daarna in vrijwel elke architectuur is beland, en kan uit de runs van de
klas aflezen wat een skip op DTD opleverde en bij welke diepte.

2.2 Wat een kernel doet: lokaal scannen langs een geordende as, met op elke positie
dezelfde gewichten. Kan uitleggen waarom dat op een reeks `(B, L, C)` en op een afbeelding
`(B, C, H, W)` dezelfde operatie is, en waarom een kernel veel minder gewichten nodig
heeft dan een dense laag op dezelfde input.

2.3 Dat downsampling minder posities op een as is: een kernel met stride leert hoe, pooling
doet het met een vaste regel, en global pooling is een aggregatie die de as weghaalt.

2.4 Wat het receptive field is, dat het groeit met diepte en met downsampling, en dat
kernels van verschillende grootte naast elkaar verschillende schalen zien.

2.5 Wat een state is: context die langs een geordende as wordt meegedragen. Kan zeggen
wat een state kan dat een kernel niet kan (verband over willekeurige afstand, volgorde)
en wat dat kost (stap voor stap rekenen, vergeten).

2.6 Wat een gate is: een geleerde vermenigvuldiger tussen 0 en 1 die bepaalt wat
doorgaat. Kan uitleggen hoe een gate een state laat onthouden of vergeten, en dat een
skip de route is waar geen gate op staat.

2.7 Dat een inductive bias een weddenschap over de data is: een kernel wedt op
lokaliteit, een state op volgorde, een basis van trend en seizoen op periodiciteit. Kan
benoemen wanneer zo'n weddenschap uitbetaalt en wanneer niet, en herkent het bredere
idee: benut de geometrie van je data.

2.8 Dat dezelfde reeks als 1D-reeks of als 2D-raster aanbieden een ontwerpkeuze is: de
shape verandert, de data niet, en welke buren een kernel ziet verandert mee.

2.9 Dat je zonder voorgetrainde encoder het hele model zelf traint, en dat dat vaak geen
probleem is: veel timeseries zijn met een simpel model of een GRU al goed te leren. Kan
uitleggen waarom een foundation model voor images veel meer uitmaakt dan voor zo'n reeks.

## De student kan

2.10 Blokken combineren in een design script — één getypeerde stap per bewerking — en het
schema daarvan in atlas lezen: de shape op elke pijl, waar takken splitsen en waar ze
samenkomen.

2.11 Samen met een agent een ontwerp maken en het verdedigen: vooraf zeggen wat je
verwacht en waarom, een metric kiezen die bij het probleem past (bij hartslagen weegt een
gemiste patiënt zwaarder dan een vals alarm), en elke keuze in het ontwerp kunnen
uitleggen zonder de agent erbij.

2.12 Werken als klas: de eigen runs en het design script publiceren naar de
runs-repository, het werk van anderen bekijken in MLflow, en daarop voortbouwen met
bronvermelding.

## Materiaal

| leerdoel | onderwerp | boek | notebook |
|---|---|---|---|
| 2.1 | skip, residual connection | 186-191 | `02_state_and_gates`, presentatie les 2 |
| 2.2, 2.3, 2.4 | kernel, downsampling, receptive field | 161-174 | `01_kernels` |
| 2.5, 2.6 | state en gate | les | `02_state_and_gates` |
| 2.7 | inductive bias als weddenschap | les | presentatie les 2, `01_kernels` |
| 2.8, 2.10 | 1D of 2D, design script, atlas | les | `03_design` |
| 2.9 | zelf trainen zonder encoder | les | `01_kernels` |
| 2.11, 2.12 | ontwerpen, verdedigen, werken als klas | les | `04_exercise`, `docs/submit-runs.md` |

Optioneel: `03_decomposition_and_nbeats` bouwt de decompositie uit MADS-DAV in een
architectuur in — een route die trend en seizoen uitrekent, en een netwerk dat de rest
fit.

## Resources

- How convolutional neural networks work: https://www.youtube.com/watch?v=FmpDIaiMIeA
- But what is a convolution?: https://www.youtube.com/watch?v=KuXjwB4LzSA
- Illustrated guide to recurrent neural networks: https://youtu.be/LHXXI4-IEns
- Illustrated guide to LSTM's and GRU's: https://youtu.be/8HyCNIVRbSU
- Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
