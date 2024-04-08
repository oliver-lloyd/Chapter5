# Chapter 5: Out-of-sample learning in LibKGE

Results from chapter 4 showed that while some inductive success can be had by simply aggregating embeddings learned by a transductive model for known drugs, there is a definite drop-off in performance compared to the transductive-only case. A useful real-world polypharmacy side effect prediction model should be able to handle new drugs without undergoing the costly process of re-embedding the whole graph OR sacrificing prediction capability to the extent we observed.

[Other research](https://aclanthology.org/2020.findings-emnlp.241.pdf) has suggested that this kind of aggregation doesn't perform well because the vectors aren't <i>learned to be aggregable</i>. The authors suggest a modification to the embedding process to addresses this issue: with probability psi, look up the vectors of a node's neighbours and aggregate those to create a single vector to use in place of said node's actual embedding.

In this work, chapter 5 of my thesis, I aim to implement this procedure within the LibKGE framework. I will then embed the Decagon graph as done in chapter 3, before testing it in the inductive case as performed in chapter 4. In both cases, performance will be compared directly to the corresponding performance of the transductive model to give an understanding of the inductive capabilities of Albooyeh et al's method.

Contact: oliver.lloyd@bristol.ac.uk