

<div class="row">
    <div class="col-md-4">
        <img src="resources/favicon_large.png" alt="logo" width="120" height="120" />
    </div>
    <div class="col-md-8">
        <h1>SPAR: Semantic Projection with Active Retrieval</h1>
    </div>
</div>

- [Overview](#overview)
- [Installation and Quick Start](#installation-and-quick-start)
- [Interface and Usage](#interface-and-usage)

## Overview
SPAR is a Python NLP package that enables interactive measurement of text. With SPAR, you can quantify short documents (e.g., social media posts) using latent, continuous scales such as *`creativity`*, *`collaboration`*, *`danger`*, by measuring their semantic similarity with a set of example (seed) documents, for example:  _`'encourage new ways of thinking'`_, _`'working together to weather the storm'`_, _`'we are facing a deadly virus.'`_ 

Main features:

* conducts domain-adaptive and few-shots measurements, without requiring any model training or fine-tuning. It is combines the idea of semantic projection with active semantic search ([Grand et al. 2022](https://www.nature.com/articles/s41562-022-01316-8), Blinded Authors 2023), which allows users to find the most relevant, context-specific documents to define the scales. 
* supports multiple state-of-the-arts text embedding methods, such as [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html) or [OpenAI Text Embeddings API](https://platform.openai.com/docs/guides/embeddings). 
* comes with a user-friendly web interface that makes defining scales and conducting measurements intuitive and accessible. 

SPAR is built on open source packages such as [HuggingFace Transformers](https://huggingface.co/transformers/), [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/), and [Gradio](https://gradio.app/). 

If you find SPAR useful in your work, please cite the following paper:

* Blinded Authors (2023), A Computational Framework for Understanding Firm Communication During Disasters, Under Review at *Information Systems Research*.
  
**Please note that the project is currently in a research preview (pre-alpha) stage**. To view the planned features for the project, please see the [Road Map](ROADMAP.MD).

## Installation and Quick Start
To quickly launch SPAR in Google Colab, click the following button and run the notebook code:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maifeng/SPAR_measure/blob/master/resources/example_colab.ipynb)

You can also install SPAR on your own machine. It is recommended to use a virtual environment and upgrade pip first with `pip install -U pip`. SPAR can be installed via pip: 

    pip install -U spar-measure

To launch SPAR on your own machine, use the following command in the terminal:

    python -m spar_measure.gui

And open the interactive app in your browser at `http://localhost:7860/`.

If a CUDA GPU is available, SPAR will use it to speed up embedding. If you choose not to use a GPU, you can set the CUDA_VISIBLE_DEVICES environment variable to an empty string:

    CUDA_VISIBLE_DEVICES="" python -m spar_measure.gui

See full documentation for other usage options [here](resources/Manual.MD). 

## Interface and Usage

SPAR is based on the following 4 simple steps: 

1. Upload a CSV file with the text content to be measured and a document ID column. Select embedding method and embed the documents.

<img src="resources/imgs/sc1.png" alt="sc1" style="width: 90%;"/>

2. Define a set of dimensions and **generic** seed queries. For example:
   * `Creative`: *"We should adapt and innovate."*
   * `Positive emotion`: *"We are happy."*
   * `Danger`: *"It is dangerous."*
  
    Then, search for sentences in a corpus that are similar to the generic seed queries, and use the results to define dimensions **in the context** of the corpus. 
    For example:

   * `Creative`: 
     * *"Digital technology will play a huge role going forward."*
     * *"How do you adapt to these uncharted waters? "*
   * `Positive emotion`: 
     * *"The smiling faces say it all."*
     * *"A round of applause to all of our recent WaFd Foundation grant recipients!"*
   * `Danger`: *"How do you prevent the spread of a deadly virus?"*
    
    Enter the above new context-specific sentences to the query box and click the "Embed Queries and Save Dimensions" button.
<img src="resources/imgs/sc2.png" alt="sc2" style="width: 90%;"/>

3. Define scales, which consists of one or more demensions. For example:  
     * `Sentiment = Positive emotion - Negative emotion`
     * `Creativity = Creative` 

<img src="resources/imgs/sc3.png" alt="sc3" style="width: 90%;"/>

4. Project the document embeddings onto the scale embeddings. A CSV file with the results can be downloaded.

<img src="resources/imgs/sc4.png" alt="sc4" style="width: 90%;"/>

