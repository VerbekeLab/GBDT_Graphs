# Inductive Inference of Gradient-Boosted Decision Trees on Graphs for Insurance Fraud Detection </br><sub><sub>Félix Arthur Vandervorst, Wouter Verbeke, Tim Verdonck [[nd]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4887265)</sub></sub>
This paper combines the performance of gradient boosting on tabular data with heterogeneous network learning in insurance fraud detection. It illustrates how metapaths can be used to aggregate the network data, and how the principle of gradient boosting can be applied on this data. The performance is tested on synthetic, open-source and proprietary network data, building a strong case for the method's usefulness. 

The main contributions of this paper are:
1) We present a novel method, graph-gradient boosted machine (G-GBM), based on probability-weighted metapaths
adapted to gradient-boosted trees.
2) We compare its performance with that of GraphSage on simulated random graphs for independent
cross-validation and that of HinSage on a real insurance heterogeneous graph.
3) We present an adapted interpretability adaptation of the popular SHAP-based explanation in classic supervised learning problems.

A preprint is available on [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4887265). 

## Repository structure
This repository is organised as follows:
```bash
|- config/
    |- data/
        |- config.yaml
    |- methods/
        |- config.yaml
|- data/
    |- insurance/
|- res/
|- scripts/
    |- experiment.py
|- src/
    |- data/
        |- graph_data.py
    |- methods/
        |- G_GBM.py
        |- network.py
        |- utils/
            |- classifier.py
            |- paths.py
    |- utils/
        |- evaluation.py
        |- param_dict.py
        |- setup.py
```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Citing
Please cite our paper and/or code as follows:
*Use the BibTeX citation*

```tex

@article{vandervorst4887265inductive,
  title={Inductive Inference of Gradient-Boosted Decision Trees on Graphs for Insurance Fraud Detection},
  author={Vandervorst, F{\'e}lix Arthur and Verbeke, Wouter and Verdonck, Tim},
  journal={Available at SSRN 4887265}
}

```