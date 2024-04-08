# Target selection algorithm

The steps for the target selection algorithm are as follows:

1. Extract the associated diseases for each small molecule in the DrugBank dataset with an LLM model.
2. Gather the target proteins for each disease based on the drug-target interactions in the DrugBank dataset.
3. Compute the the target protein co-occurrence matrix for each disease.
4. Iteratively select the best combination of target proteins for each disease through a greedy algorithm.

By extracting the diseases associated with each drug, we can also compute the drugs SMILES representation for the defined diseases and save them as holdout set for the de Novo Design experiment (stored in `guacamol/data/holdout_set_gcm_multitarget.smiles`).

To run the LLM model locally, we used the [ollama](https://github.com/ollama/ollama) tool, loading a 4-bit quantized Mixtral model to perform the structured information retrieval task.