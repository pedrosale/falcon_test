from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter  # Importe esta linha
from langchain.vectorstores import FAISS
from streamlit_chat import message
import os
import requests
import streamlit as st
import tempfile
import urllib.request




load_dotenv()

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Como posso te ajudar?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá, sou assistente do Pedro."]

def conversation_chat(query, chain, history):
    result = chain.run(query)
    history.append((query, result))
    return result

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("question:", placeholder="Me pergunte sobre o(s) conjunto(s) de dados pré-carregados", key='input')
            submit_button = st.form_submit_button(label='Enviar')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

def create_conversational_chain(vector_store):
    # Obter o token de API do ambiente
    huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "SEU_TOKEN_PADRAO_AQUI")
    
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.3, "max_new_tokens": 2000}, api_key=huggingface_token)
    prompt = """
You are an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's question
Question: {question}\n\nAnswer: Let's think step by step."""
    template = PromptTemplate(template=prompt, input_variables=["question"])
    chain = LLMChain(prompt=template, llm=llm)
    return chain

def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title('[Versão 4.0] 🦙💬 Falcon Chatbot desenvolvido por Pedro Sampaio Amorim.')
    # URL direta para a imagem hospedada no GitHub
    image_url = 'https://raw.githubusercontent.com/pedrosale/bot2/168f145c9833dcefac6ccab4c351234e819a5e97/fluxo%20atual.jpg'
    # Exibir a imagem usando a URL direta
    st.image(image_url, caption='Arquitetura atual: GitHub + Streamlit')
    st.markdown('**Esta versão contém:**  \nA) Modelo FALCON com refinamento de parâmetros;  \nB) Conjunto de dados pré-carregados do CTB [1. Arquivo de Contexto](https://raw.githubusercontent.com/pedrosale/bot2/main/CTB3.txt) e [2. Reforço de Contexto](https://raw.githubusercontent.com/pedrosale/bot2/main/CTB2.txt);  \nC) Processamento dos dados carregados (em B.) com uso da biblioteca Langchain.')
    # Carrega o arquivo diretamente (substitua o caminho do arquivo conforme necessário)

    # Carrega o primeiro arquivo diretamente
    file_path1 = "https://raw.githubusercontent.com/pedrosale/bot2/main/CTB3.txt"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
        temp_file1.write(urllib.request.urlopen(file_path1).read())
        temp_file_path1 = temp_file1.name

    text1 = []
    loader1 = TextLoader(temp_file_path1)
    text1.extend(loader1.load())
    os.remove(temp_file_path1)
    
    # Carrega o segundo arquivo diretamente
    file_path2 = "https://raw.githubusercontent.com/pedrosale/bot2/main/CTB2.txt"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
        temp_file2.write(urllib.request.urlopen(file_path2).read())
        temp_file_path2 = temp_file2.name

    text2 = []
    loader2 = TextLoader(temp_file_path2)
    text2.extend(loader2.load())
    os.remove(temp_file_path2)
    
    # Combina os textos carregados dos dois arquivos
    text = text1 + text2

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    # Create the chain object
    chain = create_conversational_chain(vector_store)

    display_chat_history(chain)

if __name__ == "__main__":
    main()
