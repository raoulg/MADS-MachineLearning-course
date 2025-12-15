# Les 5
Lees van Understanding Deep Learning hoofdstuk 12, Transformers

Begrip:
5.0.1 - wat een wordembedding is en wat de motivatie is tov one-hot-encoding
5.0.2 - wat tokenization is, en welke soorten er zijn (word, subword, char) en hoe BPE daarin past
5.1 - Precision vs Recall trade off, Confusion matrix, ethische problemen van de trade off
5.2 - de motivatie achter attention (vs convolutions)
5.3 - wat een semantische vectorruimte is
5.4 - Hoe het "reweighing" van word-embeddings werkt met behulp van attention
5.5 - waarom scaling en masking nodig zijn
5.6 - wat multihead attention is
5.7 - wat positional encoding is, en waarom dat behulpzaam kan zijn
5.8 - kan uitleggen hoe attention behulpzaam is bij timeseries in het algemeen, en NLP in het bijzonder.
5.9 - kent het verschil in dimensionaliteit van tensors (2D tensors, 3D tensors, 4D tensors) voor de diverse lagen  (Dense, 1D en 2D Convulutionl, MaxPool, RNN/GRU/LSTM, Attention, Activation functions, Flatten) en hoe deze met elkaar te combineren zijn.
5.13 - kan uitleggen wat scaling laws zijn
5.14 - begrijpt RAG, LoRA en MCP
5.15 - Kan redeneren over AI alignment uitdagingen

Vaardigheden:

5.10 - een attention layer toevoegen aan een RNN model
5.11 - kan een preprocessor voor NLP maken (bv punctuation, lowercase, spaces en xml strippen)
5.12 - een datapreprocessor aanpassen voor een dataset

|                         topic | description                        |                            material |
|------------------------------ | ---------------------------------- |------------------------------------ |
|                          5.0  | tokenization / wordembeddings      |               04_tokenization.ipynb |
|                           5.1 | confusion matix, precision-recall  |                    05_metrics.ipynb |
| 5.2, 5.4, 5.5, 5.6, 5.7,  5.8 | attention                          | 01_reweighing.ipynb, lecture, paper |
|                           5.3 | semnatische vectorruimte           |                             lecture |
|                           5.9 | tensors                            |              alle lessen tot nu toe |
|                          5.10 | attention implementeren            |  04_imdb_attention.ipynb, exercises |
|                          5.11 | preprocessing                      |         03_imdb_preprocessing.ipynb |
|                          5.12 | Bring your own dataloader          |            exercises, mads_datasets |

# resources
Kijk de volgende filmpjes:
- https://www.youtube.com/watch?v=yGTUuEx3GkA
- https://www.youtube.com/watch?v=tIvKXrEDMhk

Optioneel: lees de attention paper [attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
Optioneel: [transformers visually explained](https://poloclub.github.io/transformer-explainer/)
Optioneel: [chatgpt made me delusional](https://www.youtube.com/watch?v=VRjgNgJms3Q)
