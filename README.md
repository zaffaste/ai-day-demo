# Embedding-test
## _A simple solution to test Azure OpenAI Embeddings_
This sample contains a jupyter notebook that demonstrates an approach for consuming Azure OpenAI Embeddings service with a simple Python script. It uses Azure OpenAI Service to access the OpenAI embeddings model (text-embedding-ada-002).

The repo doesn't include sample data for pre-trained word vector, all these data could be found on [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) site and they are made available under the [Public Domain Dedication and License v1.0](http://opendatacommons.org/licenses/pddl/). This sample was based on "glove.6B.50d.txt" file.

## Features

* Read data from file and load it in a dictionary
* Visualize data contained in arrays into a heatmap chart to illustrate the vector relationship between different words

## Getting Started

> **IMPORTANT:** In order to deploy and run this example, you'll need an **Azure subscription with access enabled for the Azure OpenAI service**. You can request access [here](https://aka.ms/oaiapply). You can also visit [here](https://azure.microsoft.com/free/cognitive-search/) to get some free Azure credits to get you started.

> **AZURE RESOURCE COSTS** this sample make use of Azure OpenAI resources that have a monthly cost. The creation of this resource is not included in this sample

### Prerequisites
* Azure subscription with access enabled for the Azure OpenAI service
* text-embedding-ada-002 version 2 deplyment model 
