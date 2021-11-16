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

We'll use @batch to train the full dataset (40,000 samples) on AWS.
We'll need GPU since it'll take a while for the full data on CPU.
