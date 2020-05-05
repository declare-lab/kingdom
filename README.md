# KinGDOM: Knowledge-Guided DOMain adaptation for sentiment analysis (ACL 2020)

[_KinGDOM_](https://arxiv.org/abs/2005.00791.pdf) takes a novel perspective on the task of domain adaptation in sentiment analysis by exploring the role of external commonsense knowledge. It utilizes the ConceptNet knowledge graph to enrich the semantics of a document by providing both domain-specific and domain-general background concepts. These concepts are learned by training a graph convolutional autoencoder that leverages inter-domain concepts in a domain-invariant manner. Conditioning a popular domain-adversarial baseline method with these learned concepts helps improve its performance over state-of-the-art approaches, demonstrating the efficacy of the proposed framework.

![Alt text](KinGDOM.jpeg?raw=true "KinGDOM framework")

### Requirements
scipy==1.3.1
gensim==3.8.1
torch==1.4.0
numpy==1.18.2
scikit_learn==0.22.2.post1

### Execution
`python train.py`

### Citation

Please cite the following paper if you find this code useful in your work.

`KinGDOM: Knowledge-Guided DOMain adaptation for sentiment analysis. D. Ghosal, D. Hazarika, N. Majumder, A. Roy, S. Poria, R. Mihalcea. ACL 2020.`
