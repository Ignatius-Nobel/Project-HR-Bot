from langchain.document_loaders import DirectoryLoader
from dotenv import load_dotenv
load_dotenv()


directory = 'data/'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
# print(len(documents))

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
# print(len(docs))
# print(docs[23].page_content)

from langchain_community.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# import pinecone 
from langchain_community.vectorstores import Pinecone
# from pinecone import Pinecone
import pinecone
# initialize pinecone
pinecone.init(
    api_key="2be35523-81bb-4f80-b192-edb4351b9ed1",  # find at app.pinecone.io
    environment="us-east-1"  # next to api key in console
)

index_name = "hr"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
