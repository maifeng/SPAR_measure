## Overview
SPAR is a Python NLP package that enables interactive measurement of text. With SPAR, you can quantify short documents (e.g., social media posts) using latent, continuous scales such as *`creativity`*, *`collaboration`*, *`danger`*, by measuring their semantic similarity with a set of example (seed) documents, for example:  _`'encourage new ways of thinking'`_, _`'working together to weather the storm'`_, _`'we are facing a deadly virus.'`_ 

Main features:

* conducts domain-adaptive and few-shots measurements, without requiring any model training or fine-tuning. It is combines the idea of semantic projection with active semantic search ([Grand et al. 2022](https://www.nature.com/articles/s41562-022-01316-8), Blinded Authors 2023), which allows users to find the most relevant, context-specific documents to define the scales. 
* supports multiple state-of-the-arts text embedding methods, such as [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html) or [OpenAI Text Embeddings API](https://platform.openai.com/docs/guides/embeddings). 
* comes with a user-friendly web interface that makes defining scales and conducting measurements intuitive and accessible. 

If you find SPAR useful in your work, please cite the following paper:

* Blinded Authors (2023), A Computational Framework for Understanding Firm Communication During Disasters, Under Review at *Information Systems Research*.


## Installation and Quick Start
Simply click the following button and run the code in the notebook to launch SPAR in Google Colab for quick testing:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maifeng/SPAR_measure/resources/example_colab.ipynb)

You can also install SPAR on your own machine. It is recommended to use a virtual environment and upgrade pip first with `pip install -U pip`. SPAR can be installed via pip: 

    pip install -U spar-measure

To launch SPAR on your own machine, use the following command in the terminal:

    python -m spar_measure.gui

And open the interactive app in your browser at `http://localhost:7860/`.

If a CUDA GPU is available, SPAR will use it to speed up embedding. If you choose not to use a GPU, you can set the CUDA_VISIBLE_DEVICES environment variable to an empty string:

    CUDA_VISIBLE_DEVICES="" python -m spar_measure.gui

## Additional Details
For additional details and information, please refer to the project's [GitHub Repository](https://github.com/maifeng/SPAR_measure).