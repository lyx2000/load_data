import pandas as pd
from transformers import GPT2TokenizerFast
import openai
from openai.embeddings_utils import get_embedding
import time

openai.api_key = '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'

input_datapath = 'fine_food_reviews_1k.csv'  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]
df = df.dropna()
df['combined'] = "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
print(df.head(2))

# subsample to 1k most recent reviews and remove samples that are too long
df = df.sort_values('Time').tail(1_100)
df.drop('Time', axis=1, inplace=True)


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# remove reviews that are too long
df['n_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))
df = df[df.n_tokens<8000].tail(1_000)
print(len(df))



# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage
def my_function(x):
    time.sleep(10) 
    get_embedding(x, engine='text-embedding-ada-002')


# This will take just between 5 and 10 minutes
df['ada_similarity'] = df.combined.apply(my_function)
df['ada_search'] = df.combined.apply(my_function)
df.to_csv('fine_food_reviews_with_embeddings_1k.csv')


