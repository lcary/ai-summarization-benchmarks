import random

from datasets import load_dataset


def get_summary(data: dict) -> list:
    return data["url"]


def has_document(data: dict) -> bool:
    return bool(data["article"]["document"])


num_documents = 1000

dataset = load_dataset("wiki_lingua", "english")
dataset = dataset["train"]
dataset = list(filter(has_document, dataset))
random.shuffle(dataset)
urls = [data["url"] for data in dataset][:num_documents]

articles_file = f"data/{num_documents}_articles.txt"
with open(articles_file, "w") as f:
    for url in urls:
        f.write(f"{url}\n")

print(f"Wrote: {articles_file}")
