"""We'll use a small dataset (100 samples) to validate that the flow works locally.
This workflow takes about 7 minutes on a Mac, of which 6 minutes are training.

Without Metaflow, we have a few problems:
1. How to use multiple workers to speed up the process?
2. If we want to run the code on AWS, we'll have to create containers
3. Have to explicitly save model/tokenizer and load them again for prediction
(see sent_analysis_predict.py)

When the model is training, let's show people how to convert this to Metaflow.
(see sent_analysis_metaflow.py)
"""

import torch

import utils

FNAME = "data/imdb_kaggle.csv"
BASE_MODEL = "distilbert-base-uncased"

# Convert to Torch dataset
class IMDbDataset(torch.utils.data.Dataset):
    @utils.timeit
    def __init__(self, tokenizer, split):
        self.encodings = tokenizer(list(split.review), truncation=True, padding=True)
        self.labels = list(split.sentiment)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SentimentAnalysis(object):
    """
    The workflow performs the following steps:
    1) Ingest a CSV into a Pandas Dataframe and split it into a train, eval, and test split 
    2) Tokenize the data
    3) Fine-tune DistilBERT on the train split and using eval split for evaluation

    """
    
    def __init__(self, mode="small"):
        self.mode = mode

    def load_data(self):
        """
        The start step:
        1) Loads the movie metadata into pandas dataframe.
        2) Finds all the unique genres.
        3) Launches parallel statistics computation for each genre.

        """
        import pandas as pd
        
        # Load the data set into a pandas dataframe.
        df = pd.read_csv(utils.script_path(FNAME))
        
        # Drop duplicates
        df = df.drop_duplicates(subset="review")
        
        # Convert "positive" and "negative" labels into numeric labels
        df["sentiment"] = df["sentiment"].apply(utils.cat2num)

        if self.mode == "small":
            self.train_df = df[:100]
            self.eval_df = df[40000:40010]
            self.test_df = df[45000:45010]
        else:
            self.train_df = df[:40000]
            self.eval_df = df[40000:45000]
            self.test_df = df[45000:]

    def tokenize(self):
        """
        Tokenize the data
        """
        from transformers import DistilBertTokenizerFast
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

        self.eval_dataset = IMDbDataset(self.tokenizer, self.eval_df)
        self.test_dataset = IMDbDataset(self.tokenizer, self.test_df)
        self.train_dataset = IMDbDataset(self.tokenizer, self.train_df)

    def train(self):
        """
        Train on train and evaluate on eval
        """
        from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
        
        self.training_args = TrainingArguments(
            output_dir="./results",          # output directory
            num_train_epochs=1,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=32,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir="./logs",            # directory for storing logs
            logging_steps=5,
        )

        self.model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL)
        
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

        # Run small_train_dataset (100 samples) on a mac would take 7 minutes
        self.trainer.train()
        self.trainer.evaluate()
        trainer.save_model("checkpoints/saved_model_v0")

if __name__ == "__main__":
    model = SentimentAnalysis("small")
    model.load_data()
    model.tokenize()
    model.train()
    model.predict("This movie is awesome!")