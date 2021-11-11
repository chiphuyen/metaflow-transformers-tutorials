"""After showing people the sentiment analysis model without Metaflow (sent_analysis_train.py),
we'll show how Metaflow can make their lives easier.

Live coding to show the following steps of converting the old file into Flow:

1. Get the class to inherit from FlowSpec
2. Remove the __init__ function and use Parameter to define the mode.
    "small" means training with only 100 samples. "full": 40,000 samples
3. Use IncludeFile (StringIO is needed): why?
4. Define the workflow using @step and self.next
    Require `start` and `end` steps.
5. Remove the model/tokenizer saving & loading code (no need to pass around awkward configs)

Metaflow will automatically take care of the following:
1. Automatically serialize and save your model and tokenizer (and any object assigned to self.)
2. Create containers to do training on the full dataset to AWS (change mode="full")
3. Examine past flows and saved model/tokenizer (see inspect_flow.ipynb)


Note: all attributes of a FlowSpec (self.) will be pickled, so don't assign complex objects to self.
    Q: What if I need to?
"""
from metaflow import FlowSpec, IncludeFile, Parameter, step

import torch

import utils

FNAME = "data/imdb_kaggle.csv"
BASE_MODEL = "distilbert-base-uncased"

# Convert to Torch dataset
class IMDbDataset(torch.utils.data.Dataset):
    @utils.timeit
    def __init__(self, tokenizer, split):
        print(type(split))
        print(split.head())
        self.encodings = tokenizer(list(split.review), truncation=True, padding=True)
        self.labels = list(split.sentiment)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SentimentAnalysisFlow(FlowSpec):
    """
    A sentiment analysis model: fine-tuned DistilBERT on IMDb dataset.
    
    The workflow performs the following steps:
    1) Ingest a CSV into a Pandas Dataframe and split it into a train, eval, and test split 
    2) Tokenize the data
    3) Fine-tune DistilBERT on the train split and using eval split for evaluation

    """
    mode = Parameter("mode", default="small")
    
    fname = IncludeFile(
        "fname",
        help="The path to a movie metadata file.",
        default=utils.script_path("data/imdb_kaggle.csv"),
    )

    @step
    def start(self):
        """
        The start step:
        1) Loads the movie metadata into pandas dataframe.
        2) Finds all the unique genres.
        3) Launches parallel statistics computation for each genre.

        """
        from io import StringIO # why is this needed?
        import pandas as pd
        
        # Load the data set into a pandas dataframe.
        df = pd.read_csv(StringIO(self.fname))
        
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
            
        self.next(self.tokenize)

    @step
    def tokenize(self):
        """
        Tokenize the data using the tokenizer for the `base_model`.
        Output train_dataset, eval_dataset, and test_data
        """
        from transformers import DistilBertTokenizerFast
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

        self.eval_dataset = IMDbDataset(self.tokenizer, self.eval_df)
        self.test_dataset = IMDbDataset(self.tokenizer, self.test_df)
        self.train_dataset = IMDbDataset(self.tokenizer, self.train_df)
        self.next(self.train)

    @step
    def train(self):
        """
        Fine-tune the `base_model` on train_dataset and evaluate on eval_dataset. 
        """
        import torch
        from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
        
        training_args = TrainingArguments(
            output_dir="./results",          # output directory
            num_train_epochs=1,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=32,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir="./logs",            # directory for storing logs
            logging_steps=2,
        )

        self.model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

        # Run small_train_dataset (1000 samples) on a mac would take 37 minutes
        trainer.train()
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == "__main__":
    SentimentAnalysisFlow()
