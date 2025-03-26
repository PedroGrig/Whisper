%%writefile app.py
import streamlit as st

st.title("Transcrição de Áudio 🎤")
st.write("Bem-vindo ao sistema de transcrição automática!")

# Upload do arquivo
arquivo = st.file_uploader("Envie um arquivo de áudio", type=["mp3", "wav", "ogg"])

if arquivo is not None:
    st.success("Arquivo recebido! Iniciando transcrição...")
    # Aqui você pode chamar o Whisper para processar o arquivo
    # Exemplo: transcricao = transcrever_audio(arquivo)
    transcricao = "Texto transcrito aqui..."  # Apenas um placeholder
    st.text_area("Transcrição:", transcricao)

st.sidebar.markdown("Feito com ❤️ usando Streamlit")
