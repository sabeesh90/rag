import streamlit as st
import streamlit.components.v1 as components
import os 
import re
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
import streamlit.components.v1 as components

# Access the API key from secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Set the API key in the environment
os.environ["OPENAI_API_KEY"] = api_key

# defining the model here
model = ChatOpenAI(model="gpt-4o")

# parser is to parse the output contents into more meaninful ormat
parser = StrOutputParser()

# this is the template that would takee an input and output an answer
template = """
You are a PhD research student. Your understanding of subjects is at the research level. Your output also should be at an advanced level. You are trained on two broad subjects \
Cognitive neuroscience and Computational models in EEG 
There is a context i will prodive you based on that you can answer some questions. The answer should be detailed and scientiic. if possible include equations and images
Context : {context}
question : {question}
chat_history {chat_history}
"""
prompt  = ChatPromptTemplate.from_template(template)
embeddings = OpenAIEmbeddings()
def display_latex_content(content):
    html_content = f"""
    <html>
        <head>
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f9f9f9; /* Light background for better visibility */
                    color: #333; /* Dark font color for contrast */
                }}
                .scrollable {{
                    max-height: 400px; /* Set maximum height for scrolling */
                    overflow-y: auto; /* Enable vertical scrolling */
                    padding: 10px;
                    border: 1px solid #ccc; /* Optional border for visual separation */
                    background-color: #ffffff; /* White background for the text area */
                }}
            </style>
        </head>
        <body>
            <div class="scrollable">
                {content}
            </div>
        </body>
    </html>
    """
    components.html(html_content, height=450)

def format_chat_history(messages):
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
def format_output(text):
    # Wrap the text to a fixed width (e.g., 80 characters)
    wrapped_text = textwrap.fill(text, width=80)
    print(wrapped_text)

# function to read a pdf file
def read_pdfs(uploaded_files):
    combined_pdfs = []  
    for pdf_file in uploaded_files:
        temp_file_path = f"./temp_{pdf_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        try:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load() 
            st.subheader(pdf_file.name)
            pdf_text = loader.load()
            combined_pdfs.extend(pdf_text)
        except Exception as e:
            st.error(f"Error loading {pdf_file.name}: {e}")

        os.remove(temp_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap = 20)
    documents = text_splitter.split_documents(combined_pdfs)
    return documents
    
def create_combined_input(context, question, chat_history):
        return f"Context: {context}\nQuestion: {question}\nChat History: {chat_history}"

# Streamlit app layout
st.title("Upload Multiple PDF Files")

# File uploader for PDFs
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Check if files are uploaded
if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded.")
    documents = read_pdfs(uploaded_files)  # Read and display the content of the PDFs
    vs2 = DocArrayInMemorySearch.from_documents(documents, embeddings)
    # {"context": vs2.as_retriever(), "question" : RunnablePassthrough(), "chat_history": RunnablePassthrough()}
    chain = (
     {"context": vs2.as_retriever(), "question" : RunnablePassthrough(), "chat_history": RunnablePassthrough()}
    | prompt
    | model
    | parser
    )

  
else:
    st.warning("Please upload one or more PDF files.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Hei Sabeesh!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    chat_history_str = format_chat_history(st.session_state.messages)

    combined_input = create_combined_input(
        context=vs2.as_retriever(),
        question=prompt,
        chat_history=chat_history_str
    )
    response = chain.invoke(combined_input)
    # response = chain.invoke(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
        # display_latex_content(response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    
