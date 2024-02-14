import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter  # Importe esta linha
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile
import urllib.request
import requests

load_dotenv()


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Como posso te ajudar?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá, sou modelo que deve responder de maneira aaleatória!"]

def conversation_chat(query, chain, history):
    prompt = """
    Você é um assistente que possui as duas funções a seguir:
    1. Você é um assistente que só conversa no idioma português do Brasil (você nunca, jamais conversa em outro idioma que não seja o português do Brasil). 
    2. Você é um assistente que possui como objetivo falar coisas aleatórias a respeito de alucinação de modelos, como se fosse um papagaio estocástico fazendo isso. 
    """  # Adicionando prompt para indicar o idioma
    query_with_prompt = prompt + query
    result = chain({"question": query_with_prompt, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("question:", placeholder="Me pergunte sobre os conjuntos de dados pré-carregados", key='input')
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
    load_dotenv()

    llm = Replicate(
        streaming = True,
        model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
        #model = "tomasmcm/towerinstruct-7b-v0.1",         
        language="pt-BR",  
        callbacks=[StreamingStdOutCallbackHandler()],
        input = {"temperature": 0.01, "max_length" :500,"top_p":1})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain


def main():
    load_dotenv()
# Initialize session state
    initialize_session_state()
    st.title('Que tal conversarmos sobre imprecisão das respostas geradas pelos modelos de Large Language Models ?')
    st.markdown('**Esta versão contém:**  \nConjuntos de dados pré-carregados referente ao tema de imprecisão das respostas geradas pelos modelos Large Language Models [Veja os dados aqui](https://raw.githubusercontent.com/pedrosale/papagaio_estocastico/main/AI-Hallucinations-A-Misnomer-Worth-Clarifying.txt), [aqui](https://raw.githubusercontent.com/pedrosale/papagaio_estocastico/main/What%20are%20AI%20hallucinations_%20_%20IBM.txt) e [aqui](https://raw.githubusercontent.com/pedrosale/papagaio_estocastico/main/A%20Survey%20on%20Hallucination%20in%20Large%20Language%20Models.pdf).')
    st.markdown("""
Você é um assistente que possui as duas funções a seguir:
1. Você é um assistente que só conversa no idioma português do Brasil (você nunca, jamais conversa em outro idioma que não seja o português do Brasil). 
2. Você é um assistente que possui como objetivo falar coisas aleatórias a respeito de alucinação de modelos, como se fosse um papagaio estocástico fazendo isso.
""")

    



    # Carrega o primeiro arquivo diretamente
    file_path1 = "https://raw.githubusercontent.com/pedrosale/papagaio_estocastico/main/AI-Hallucinations-A-Misnomer-Worth-Clarifying.txt"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
        temp_file1.write(urllib.request.urlopen(file_path1).read())
        temp_file_path1 = temp_file1.name

    text1 = []
    loader1 = TextLoader(temp_file_path1)
    text1.extend(loader1.load())
    os.remove(temp_file_path1)
    
    # Carrega o segundo arquivo diretamente
    file_path2 = "https://raw.githubusercontent.com/pedrosale/papagaio_estocastico/main/What%20are%20AI%20hallucinations_%20_%20IBM.txt"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
        temp_file2.write(urllib.request.urlopen(file_path2).read())
        temp_file_path2 = temp_file2.name

    text2 = []
    loader2 = TextLoader(temp_file_path2)
    text2.extend(loader2.load())
    os.remove(temp_file_path2)

    # Carrega o segundo arquivo diretamente

    file_path3 = "https://raw.githubusercontent.com/pedrosale/papagaio_estocastico/main/A%20Survey%20on%20Hallucination%20in%20Large%20Language%20Models.pdf"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file3:
        temp_file3.write(urllib.request.urlopen(file_path3).read())
        temp_file_path3 = temp_file3.name

    text3 = PyPDFLoader(temp_file_path3).load()
    os.remove(temp_file_path3)
        
    # Combina os textos carregados dos dois arquivos
    text = text1 + text2 + text3

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
