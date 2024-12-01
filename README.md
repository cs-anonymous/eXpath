<meta name="robots" content="noindex">

# eXpath

<p align="center">
<img width="50%" alt="eXpath_logo" src="https://user-images.githubusercontent.com/6909990/124291133-87fa5d80-db54-11eb-9db9-62ca9bd3fe6f.png">
</p>

eXpath is a post-hoc local explainability tool specifically tailored for embedding-based models that perform Link Prediction (LP) on Knowledge Graphs (KGs).

eXpath provides a simple and effective interface to identify the most relevant facts to any prediction; intuitively, when explaining a _tail_ prediction <_h_, _r_, _t_>, eXpath identifies the smallest set of training facts mentioning _h_ that are instrumental to that prediction; analogously, a _head_ prediction <_h_, _r_, _t_> would be explained with a combination of the training facts mentioning _t_.


## eXpath Structure 

eXpath is structured in a simple architecture based on the interaction of three modules. 
When explaining a _tail_ prediction <_h_, _r_, _t_>:

* a **Pre-Filter** narrows down the space of the candidate explanations (i.e., combinations of training facts mentioning _h_) by identifying and keeping only the most promising _h_ facts;
* an **Explanation Builder** governs the search in the resulting space of the candidate explanations, i.e., the combinations of promising facts obtained from the Pre-Filtering step;
* a **Relevance Engine** is used to estimate the *relevance* of any candidate explanation that the Explanation Builder wants to verify.

The modules would work analogously to explain a _head_ prediction. The only module that requires awareness of how the original Link Prediction model is trained and implemented is the Relevance Engine. While theoreticaly specific connectors could be developed to to adapt to pre-existing models, in our research we have found it easier to make the Relevance Engine directly interact with eXpath-compatible implementations of the models.

<p align="center">
<img width="60%" alt="eXpath_structure" src="https://user-images.githubusercontent.com/6909990/140399831-0c368ac2-7cf4-48dc-bb73-eaf00f7fde52.png">
</p>


## eXpath Explanations 

Under the broad definition described above, eXpath supports two explanation scenarios: _necessary_ explanations and _sufficient_ explanations.

* Given a _tail_ prediction <_h_, _r_, _t_>, a **necessary explanation** is the smallest set of training facts featuring _h_ such that, if we remove those facts from the training set and re-train the model from scratch, the model will not be able to identify _t_ as the top-ranking tail. In other words, a _necessary_ explanation is the smallest set of _h_ facts that have made possible for the model to pick the correct tail. An analogous definition can be derived for head predictions. 

* Given a _tail_ prediction <_h_, _r_, _t_>, a **sufficient explanation** is the smallest set of training facts featuring _h_ such that, if we add those facts to random entities _c_ for which the model does not predict <_c_, _r_, _t_>, and we retrain the model from scratch, the model will start predicting <_c_, _r_, _t_> too. In other words, a _sufficient_ explanation is the smallest set of _h_ facts make it possible to extend the prediction to any other entity in the graph. An analogous definition can be derived for head predictions.


## Environment and Prerequisites

We have run all our experiments on an Ubuntu 18.04.5 environment using Python 3.7.7, CUDA Version: 11.2 and Driver Version: 460.73.01.
eXpath requires the following libraries: 
- PyTorch (we used version 1.7.1);
- numpy;
- tqdm;
- matplotlib;
- reportlab;

## Models and Datasets

The formulation of eXpath supports any Link Prediction models based on embeddings. For the sake of simplicity in our implementation we focus on models that train on individual facts, as these are the vast majority in literature. Nonetheless, our implementation can be extended to identify fact-based explanations for other models too, e.g., models that leverage contextual information such as paths, types, or temporal data.

We run our experiments on three models that rely on very different architectures: `ComplEx`, `ConvE` and `TransE`. 
We provide implementations for these models in this repository.
We explaining their predictions on the 5 best-established datasets in literature, i.e., `FB15k`, `WN18`, `FB15k-237`, `WN18RR` and `YAGO3-10`.
The training, validation and test sets of such datasets are distributed in this repository in the `data` folder.


## Training and Testing Our Models

For the sake of reproducibility, we make available through FigShare [the `.pt` model files](https://figshare.com/s/ede27f3440fe742de60b) resulting from training each system on each dataset. To run any of the experiments of our paper, the `.pt` files of all the trained models should bw downloaded and stored in a new folder `eXpath/stored_models`.

For our models and datasets we use following hyperparameters, which we have found to lead to the best performances.

<p align="center">
<img width="90%" alt="hyperparams" src="https://user-images.githubusercontent.com/6909990/124291956-66e63c80-db55-11eb-9aa6-9892ee24afc2.png">
</p>

Note that: 
* *D* is the embedding dimension (in the models we use, entity and relation embeddings always have same dimension);
* *LR* is the learning rate;
* *B* is the batch size;
* *Ep* is the number of epochs;
* *γ* is the margin in Pairwise Ranking Loss;
* *N* is the number of negative samples generated for each positive training sample;
* *Opt* is the used Optimizer (either `SGD`, or `Adagrad`, or `Adam`);
* *Reg* is the regularizer weight;
* *Decay* is the applied learning rate Decay;
* *ω* is the size of the convolutional kernels;
* *Drop* is the training dropout rate:
    * *in* is the input dropout;
    * *h* is the dropout applied after a hidden layer;
    * *feat* is the feature dropout;

After the models have been trained, their evaluation yields the following metrics:

<p align="center">
<img width="60%" alt="model_results" src="https://user-images.githubusercontent.com/6909990/135614004-db1cff3a-68db-447d-bb9c-3c7f05426957.png">
</p>

The training and evaluation processes can be launched with the commands reported in our [training and testing section](#training-and-testing-models-1).
