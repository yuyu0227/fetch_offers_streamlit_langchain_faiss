import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
import pandas as pd


openai.api_key = os.environ["OPENAI_API_KEY"]


# combine data
df_offer_retailer = pd.read_csv('data/offer_retailer.csv')
df_brand_category= pd.read_csv('data/brand_category.csv')
df_categories = pd.read_csv('data/categories.csv')
df_ = df_categories.merge(df_brand_category, left_on='PRODUCT_CATEGORY',right_on='BRAND_BELONGS_TO_CATEGORY')
df_all = df_offer_retailer.merge(df_,left_on='BRAND',right_on='BRAND')
df = df_all.drop(columns=['BRAND_BELONGS_TO_CATEGORY','CATEGORY_ID','RECEIPTS'])
new_df = df.groupby('OFFER').agg(lambda x: set(x))

# create documents
documents = []
for i in range(len(new_df)):
  record = new_df.iloc[i]
  meta = {}
  for i in range(4):
    if record[i]  and type(list(record[i])[0]) == str:
      li = [w.lower() for w in record[i]]
      meta[new_df.columns[i].lower()] = ','.join(li) 
  doc = Document(
      page_content = record.name,
      metadata=meta,
  )
  documents.append(doc)

# create FAISS index
db = FAISS.from_documents(documents, OpenAIEmbeddings())

# save FAISS index for offer data
db.save_local("offers_faiss_index")