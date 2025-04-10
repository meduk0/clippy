# Install the datasets library if you don't have it
# pip install datasets

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("agentlans/wikipedia-paragraphs")

# You now have access to the dataset
train_data = dataset["train"]

# View a sample
print(train_data[0])

# Save to a local file (optional)
train_data.to_json("wikipedia_paragraphs_train.json")
