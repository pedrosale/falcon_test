import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import Dict
import json
from io import StringIO
from random import randint
from transformers import AutoTokenizer

st.set_page_config(page_title="Document Analysis", page_icon=":robot:")
st.header("Chat with your document 📄  (Model: Falcon-40B-Instruct)")

endpoint_name = "falcon-40b-instruct-48xl"

model = "tiiuae/falcon-40b"
tokenizer = AutoTokenizer.from_pretrained(model)



@st.cache_resource
def load_chain():
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.3, "max_new_tokens": 2000})
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

# this is the object we will work with in the ap - it contains the LLM info as well as the memory
chatchain = load_chain()

# initialise session variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    chatchain.memory.clear()
if 'widget_key' not in st.session_state:
    st.session_state['widget_key'] = str(randint(1000, 100000000))

# Sidebar - the clear button is will flush the memory of the conversation
st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['widget_key'] = str(randint(1000, 100000000))
    chatchain.memory.clear()

# upload file button
uploaded_file = st.sidebar.file_uploader("Upload a txt file", type=["txt"], key=st.session_state['widget_key'])

# this is the container that displays the past conversation
response_container = st.container()
# this is the container with the input text box
container = st.container()

with container:
    # define the input text box
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    # when the submit button is pressed we send the user query to the chatchain object and save the chat history
    if submit_button and user_input:
        output = chatchain(user_input)["response"]
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
    # when a file is uploaded we also send the content to the chatchain object and ask for confirmation
    elif uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        content = "=== BEGIN FILE ===\n"
        content += stringio.read().strip()
        content += "\n=== END FILE ===\nPlease confirm that you have read that file by saying 'Yes, I have read the file'"
        output = chatchain(content)["response"]
        st.session_state['past'].append("I have uploaded a file. Please confirm that you have read that file.")
        st.session_state['generated'].append(output)

    history = chatchain.memory.load_memory_variables({})["history"]
    tokens = tokenizer.tokenize(history)
    st.write(f"Number of tokens in memory: {len(tokens)}")

# this loop is responsible for displaying the chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
