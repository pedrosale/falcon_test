import streamlit as st
from langchain import HuggingFaceHub
from apikey_hungingface import apikey_hungingface
from langchain import PromptTemplate, LLMChain
import os

# Set Hugging Face Hub API token
# Certifique-se de armazenar seu token API no arquivo `apikey_hungingface.py`
os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikey_hungingface

# Set up the language model using the Hugging Face Hub repository
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.3, "max_new_tokens": 2000})

# Set up the prompt template
template = """
You are an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's question
Question: {question}\n\nAnswer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

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
                st.text_input("Pergunta:", value=st.session_state["past"][i], key=str(i) + '_user', disabled=True)
                st.text_input("Resposta:", value=st.session_state["generated"][i], key=str(i), disabled=True)

def main():
    initialize_session_state()
    st.title('[Versão 3.0] 🦙💬 FALCON Chatbot desenvolvido por Pedro Sampaio Amorim.')
    image_url = 'https://raw.githubusercontent.com/pedrosale/bot2/168f145c9833dcefac6ccab4c351234e819a5e97/fluxo%20atual.jpg'
    st.image(image_url, caption='Arquitetura atual: GitHub + Streamlit')
    st.markdown('**Esta versão contém:**  \nA) Modelo FALCON;  \nB) Conjunto de dados pré-carregados do CTB [1. Arquivo de Contexto](https://raw.githubusercontent.com/pedrosale/bot2/main/CTB3.txt) e [2. Reforço de Contexto](https://raw.githubusercontent.com/pedrosale/bot2/main/CTB2.txt);  \nC) Processamento dos dados carregados (em B.) com uso da biblioteca Langchain.')

    display_chat_history(llm_chain)

if __name__ == "__main__":
    main()
