import requests
import streamlit as st
from langchain_core.messages import HumanMessage
from tavily import graph

st.title("Somente as chamas eternas respondem... O que deseja saber?")

with st.form("llm-form"):
    user_input = st.text_area("Pergunta:")
    submit = st.form_submit_button("Enviar")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def generate_response_via_graph(question):
    try:
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json",
        }
        full_query = f"""
            Você é uma especialista no universo de Dark Souls e deve responder na língua em que foi perguntado.
            Responda em tom sombrio e enigmático, como um guardião do fogo antigo.
            Agora, o usuário pergunta:
            {question}
            """

        json_data = {
            "query": full_query,
            "include_answer": True,
            "num_results": 3
        }

        response = requests.post(
            "https://api.tavily.com/search", headers=headers, json=json_data
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "Não encontrei resposta nas chamas...")
            return answer
        else:
            return f"Erro ao consultar Tavily: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Erro inesperado nas chamas: {str(e)}"


if submit and user_input:
    with st.spinner("Buscando as chamas antigas..."):
        answer = generate_response_via_graph(user_input)
        st.session_state["chat_history"].append(
            {"user": user_input, "guardiã do fogo": answer}
        )
        st.write(answer)

st.write("## Histórico")
for chat in st.session_state["chat_history"]:
    st.write(f"**Você:** {chat['user']}")
    st.write(f"**Guardiã do Fogo:** {chat['guardiã do fogo']}")
    st.markdown("---")