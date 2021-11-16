
First, get started with Metaflow by executing these simple flows:

1. `helloworld.py` - a simple hello world flow
2. `counter_branch.py` - test artifacts
3. `parameters.py` - test parameters
4. `foreach.py` - test foreaches (parallel tasks)

After these simple examples, you can take a look at a more realistic case:

In this tutorial, we'll fine-tune a sentiment analysis model on top of
HuggingFace's DistilBERT model with the IMDB dataset.

First, we'll show how to do it without Metaflow.
1. sent_analysis_train.py is the training code (6-7 minutes on the small dataset of 100 samples on my Mac)
2. sent_analysis_predict.py is the prediction code (30 seconds)

We'll do live coding to show how to convert the training code to Metaflow.
See sent_analysis_metaflow.py for instructions.

We'll run `python sent_analysis_metaflow.py --no-pylint run --mode small`
to train a model on 100 samples locally.

We'll show how Metaflow automatically saves trained models which we can access for predictions.


