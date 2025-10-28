from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


# def _load_data(
#     path: str,
#     splits: str,
#     cache_dir: str | None = None,
#     revision: str | None = None,
# ):
#     corpus = {}
#     queries = {}
#     relevant_docs = {}

#     for split in splits:
#         query_ds = load_dataset(
#             path,
#             "queries",
#             split=split,
#             cache_dir=cache_dir,
#             revision=revision,
#         )
#         query_ds = query_ds.map(
#             lambda x: {
#                 "id": f"query-{split}-{x['query-id']}",
#                 "text": x["query"],
#                 "image": None,
#                 "modality": "text",
#             },
#             remove_columns=["query-id", "query"],
#         )
#         queries[split] = query_ds

#         corpus_ds = load_dataset(
#             path,
#             "corpus",
#             split=split,
#             cache_dir=cache_dir,
#             revision=revision,
#         )
#         corpus_ds = corpus_ds.map(
#             lambda x: {
#                 "id": f"corpus-{split}-{x['corpus-id']}",
#                 "text": None,
#                 "modality": "image",
#             },
#             remove_columns=["corpus-id"],
#         )
#         corpus[split] = corpus_ds

#         qrels_ds = load_dataset(
#             path,
#             "qrels",
#             split=split,
#             cache_dir=cache_dir,
#             revision=revision,
#         )
#         relevant_docs[split] = {}
#         for row in qrels_ds:
#             qid = f"query-{split}-{row['query-id']}"
#             did = f"corpus-{split}-{row['corpus-id']}"
#             if qid not in relevant_docs[split]:
#                 relevant_docs[split][qid] = {}
#             relevant_docs[split][qid][did] = int(row["score"])

#     return corpus, queries, relevant_docs


class SeaDocRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="SeaDocRetrieval",
        description="Retrieve associated pages in low-resource languages according to English questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "/mnt/workspace/workgroup/chx/embedding_data/SeaDoc",
            "revision": "97a97e91cd4a817898b80bc19984a7b98c6e1777",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{xiao2025scaling,
  title={Scaling Language-Centric Omnimodal Representation Learning},
  author={Xiao, Chenghao},
  journal={arXiv preprint arXiv:2507.00000},
  year={2025}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 99.328,
                    "num_documents": 500,
                    "num_queries": 500,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    # def load_data(self, **kwargs):
    #     self.corpus, self.queries, self.relevant_docs = _load_data(
    #         path=self.metadata_dict["dataset"]["path"],
    #         splits=self.metadata_dict["eval_splits"],
    #         cache_dir=kwargs.get("cache_dir", None),
    #         revision=self.metadata_dict["dataset"]["revision"],
    #     )

    #     self.data_loaded = True

