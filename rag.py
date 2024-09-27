import streamlit as st
import os 
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import DocArrayInMemorySearch, InMemoryVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import textwrap
from IPython.display import display, Markdown, Latex



OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.write(f"My environment variable: {my_env_var}")
st.write(f"My API key: {api_key}")


st.write('Hello World')
