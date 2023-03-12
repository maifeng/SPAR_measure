"""
Util funcitons.
Several functions are copied from https://github.com/UKPLab/sentence-transformers under Apache 2.0 license, available at https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, device


def print_time_now():
    """print current time"""
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def time_now():
    """print current time"""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def line_counter(a_file):
    """Count the number of lines in a text file

    Arguments:
        a_file {str or Path} -- input text file

    Returns:
        int -- number of lines in the file
    """
    n_lines = 0
    with open(a_file, "rb") as f:
        n_lines = sum(1 for _ in f)
    return n_lines


def file_to_list(a_file):
    """Read a text file to a list, each line is an element

    Arguments:
        a_file {str or path} -- path to the file

    Returns:
        [str] -- list of lines in the input file, can be empty
    """
    file_content = []
    with open(a_file, "rb") as f:
        for l in f:
            file_content.append(l.decode(encoding="utf-8").strip())
    return file_content


def list_to_file(list, a_file, validate=True):
    """Write a list to a file, each element in a line
    The strings needs to have no line break "\n" or they will be removed

    Keyword Arguments:
        validate {bool} -- check if number of lines in the file
            equals to the length of the list (default: {True})
    """
    with open(a_file, "w", 8192000, encoding="utf-8") as f:
        for e in list:
            e = str(e).replace("\n", " ")
            f.write("{}\n".format(e))
    if validate:
        assert line_counter(a_file) == len(list)


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])

    Copied from https://github.com/UKPLab/sentence-transformers under Apache 2.0 license
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])

    Copied from https://github.com/UKPLab/sentence-transformers under Apache 2.0 license
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def semantic_search(
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.

    Copied from https://github.com/UKPLab/sentence-transformers under Apache 2.0 license
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(
                query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                corpus_embeddings[
                    corpus_start_idx : corpus_start_idx + corpus_chunk_size
                ],
            )

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k, len(cos_scores[0])),
                dim=1,
                largest=True,
                sorted=False,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id].append(
                        {"corpus_id": corpus_id, "score": score}
                    )

    # Sort and strip to top_k results
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(
            queries_result_list[idx], key=lambda x: x["score"], reverse=True
        )
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

    return queries_result_list


def normalize_embeddings(embeddings: Tensor):
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length

    Copied from https://github.com/UKPLab/sentence-transformers under Apache 2.0 license
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def pytorch_cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    Copied from https://github.com/UKPLab/sentence-transformers under Apache 2.0 license

    """
    return cos_sim(a, b)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
