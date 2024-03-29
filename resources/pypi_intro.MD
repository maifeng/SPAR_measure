
## Overview
SPAR is a Python NLP package that facilitates interactive measurement of text using LLMs. With SPAR, you can quantify short documents (e.g., social media posts) based on theoretical concepts such as *`creativity`* and *`collaboration`*, by measuring their semantic similarity with a set of example (seed) documents. 

**How it works:**
1. Start with   
   i. A corpus of documents that you want to measure;  
   ii. generic seed sentences that define theoretical concepts, e.g., `Creativity: we should innovate` and `Collaboration: we should collaborate`.   
2. Embed them into a semantic space using a pre-trained LLM.
3. Use semantic search to find domain-specific exemplary documents in the corpus that reflect the theoretical concepts in context. For example:  _`'We encourage new ways of thinking'`_, _`'We should working together to weather the storm'`_.
4. Compute the dot product between docuements and exemplary documents. 

**Main features:**

* Enables domain-adaptive and few-shot measurements of theoretical concepts without requiring model training or fine-tuning. 
* Combines the idea of semantic projection with active semantic search, which allows users to find the most relevant, context-specific documents to define the theoretical scales. 
* Supports multiple state-of-the-arts text embedding models, such as [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html) and [OpenAI Text Embeddings API](https://platform.openai.com/docs/guides/embeddings). 
* Comes with a user-friendly web interface that makes defining theoretical scales and conducting measurements intuitive and accessible. 
* Reference:
  * Bei Yan, Feng Mai, Chaojiang Wu, Rui Chen, Xiaolin Li (2023). A Computational Framework for Understanding Firm Communication During Disasters. Information Systems Research.
https://doi.org/10.1287/isre.2022.0128 

SPAR is built on open source packages such as [HuggingFace Transformers](https://huggingface.co/transformers/), [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/), and [Gradio](https://gradio.app/). 


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


## Limitations
* SPAR may not be suitable for longer or more complex documents since it represents a document using a single vector.
* Sentence embeddings may not be suitable for theoretical constructs that rely primarily on syntactic features.
* Pretrained LLMs may not have up-to-date world knowledge or new vocabularies.
* Semantic projection is a linear operation, so it may not capture non-linear patterns in the data as well as fine-tuning approaches.


## Additional Details
For additional details and source code, please refer to the project's [GitHub Repository](https://github.com/maifeng/SPAR_measure).