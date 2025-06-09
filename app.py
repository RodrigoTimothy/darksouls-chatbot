import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage

st.title("Somente as chamas eternas respondem... O que deseja saber?")

with st.form("llm-form"):
    text = st.text_area("Pergunta:")
    submit = st.form_submit_button("Enviar")

def generate_response(input_text):
    model = ChatOllama(model="llama3.2:1b")
    
    messages = [
        SystemMessage(content=(
            "Você é a Guardiã do Fogo de Dark Souls. "
            "Fale como ela: solene, misteriosa e poética. "
            "Seja sempre gentil, mas um pouco melancólica, "
            "e responda às perguntas como uma protetora das chamas eternas. "
            "Forneça informações confiáveis sobre o universo de Dark Souls, "
            "mas nunca quebre a personagem da Guardiã."
        )),
        HumanMessage(content=input_text)
    ]
    
    response = model.invoke(messages)
    return response.content

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

if submit and text:
    with st.spinner("Buscando as chamas antigas..."):
        response = generate_response(text)
        st.session_state['chat_history'].append({"user": text, "guardiã do fogo": response})
        st.write(response)

st.write("## Histórico")
for chat in st.session_state['chat_history']:
    st.write(f"**Você:** {chat['user']}")
    st.write(f"**Guardiã do Fogo:** {chat['guardiã do fogo']}")
    st.markdown("---")