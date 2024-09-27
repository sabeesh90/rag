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


# Access the API key from secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Set the API key in the environment
os.environ["OPENAI_API_KEY"] = api_key

# defining the model here
model = ChatOpenAI(model="gpt-4-turbo")

# parser is to parse the output contents into more meaninful ormat
parser = StrOutputParser()


template = """
There is a context i will prodive you based on that you can answer some questions. The answer should be detailed and scientiic. if possible inclcude equations.
Context : {context}
question : {question}
"""

prompt  = ChatPromptTemplate.from_template(template)
st.write(prompt.format(context = 'context', question  = 'question'))
st.write('Hello World')
