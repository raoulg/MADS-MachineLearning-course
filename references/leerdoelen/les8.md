# Extra's — niet getoetst

Materiaal dat goed is maar geen lessdeel meer vult. Niets hieronder wordt getoetst; het
staat er omdat het de moeite waard is en omdat een deel ervan in eerdere jaren wél in de
cursus zat.

## Graph neural networks (`8_extras/graphs`)

Lees hoofdstuk 13 uit *Understanding Deep Learning* als je hier verder in wilt.

De student die dit doorneemt begrijpt:

- Wat het verschil is tussen euclidische en niet-euclidische geometrie, en wat een metric
  is.
- Wanneer en waarvoor hyperbolische embeddings bruikbaar zijn.
- Hoe een graph convolution werkt, en hoe dat zich verhoudt tot een 2D convolutie.
- Hoe je een Laplaciaan van een graaf maakt.

De grafenkennis die je in deze cursus wél nodig hebt — graafdefinitie, adjacency,
degree, betweenness, clustering coefficient, spectral clustering — zit in les 7, waar een
graaf retrieval-infrastructuur is in plaats van een architectuur. `torch-geometric` zit
daarom in een optionele dependency group.

Videos:

- https://youtu.be/n7GYYerlQWs (niet-euclidische ruimtes)
- https://www.youtube.com/watch?v=fOctJB4kVlM
- https://www.youtube.com/watch?v=ABCGCf8cJOE
- Overzicht hyperbolische representaties:
  https://github.com/marlin-codes/Awesome-Hyperbolic-Representation-and-Deep-Learning

## Swarms en cellulaire automaten (`8_extras/swarms`)

De populatie-optimizers uit dit materiaal — differential evolution, PSO, Grey Wolf —
horen bij les 3, waar ze een search-algoritme zijn dat iets weet wat TPE niet weet. Wat
hier blijft staan is het emergentiedeel:

- Hoe simpele regels complex gedrag kunnen genereren, tot en met universele functies.
- De vier klassen van gedrag in cellulaire automaten.
- Het El Farol-probleem als voorbeeld van begrensde rationaliteit.

## Reinforcement learning

Niet langer onderdeel van deze cursus. Voor wie het wil nakijken:

- Wat een markov-proces is, en de definitie van een MDP (⟨S, A, T, R, γ⟩).
- Wat een policy π*(s)=a is, en wat een POMDP is en hoe die van een MDP verschilt.
- Continue versus discrete state-searchspaces.
- Markov-procesdiagrammen lezen, zoals in het crying-baby-probleem.
- De kenmerken van NP-complete problemen, en situaties waarin gradient-based methodes
  falen.

## Deployment en Julia

`8_extras/deployment` en `8_extras/julia_dl` staan er ongewijzigd bij.
