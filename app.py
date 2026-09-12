import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Manacá Lab",
    page_icon="🌳",
    layout="centered"
)

MODEL_ID = "menezesbruno/manaca-1b-base"

@st.cache_resource(show_spinner=False)
def carregar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model.eval()
    return tokenizer, model

st.title("🌳 Manacá Lab")
st.subheader("Experimentação com IA Brasileira")

st.markdown(
    """
Interface experimental para exploração do **Manacá-1B**.

O modelo é um **LLM base**, então funciona melhor quando você escreve
o início de uma frase ou parágrafo para que ele continue.
"""
)

with st.expander("ℹ️ Sobre este experimento"):
    st.markdown(
        """
Este projeto utiliza o **Manacá-1B**, modelo de linguagem em português brasileiro.

Possíveis aplicações de estudo:
- Inteligência Artificial;
- Ciência da Informação;
- Recuperação da Informação;
- Informação Jurídica;
- Bibliotecas e unidades de informação.

**Aviso:** os resultados são experimentais e podem conter informações incorretas.
"""
    )

try:
    with st.spinner("Carregando o Manacá-1B... isso pode levar alguns minutos na primeira execução."):
        tokenizer, model = carregar_modelo()

    st.success("Modelo carregado.")

    prompt = st.text_area(
        "Digite o início do texto:",
        placeholder="Ex.: A inteligência artificial aplicada à recuperação da informação jurídica pode...",
        height=180
    )

    col1, col2 = st.columns(2)

    with col1:
        max_tokens = st.slider(
            "Tamanho da geração",
            min_value=50,
            max_value=300,
            value=120,
            step=10
        )

    with col2:
        temperatura = st.slider(
            "Criatividade",
            min_value=0.1,
            max_value=1.5,
            value=0.7,
            step=0.1
        )

    if st.button("🌳 Gerar texto", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Digite um texto para começar.")
        else:
            with st.spinner("Gerando..."):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperatura,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id
                    )

                resultado = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

            st.markdown("### Resultado")
            st.text_area(
                "Texto gerado",
                value=resultado,
                height=320
            )

    st.markdown("---")
    st.markdown("### Exemplos para testar")

    exemplos = [
        "A inteligência artificial aplicada à recuperação da informação jurídica pode",
        "No contexto das bibliotecas universitárias, a inteligência artificial pode auxiliar",
        "A utilização de inteligência artificial no Poder Judiciário brasileiro apresenta",
        "A recuperação da informação jurídica consiste em",
    ]

    for exemplo in exemplos:
        st.code(exemplo)

except Exception as e:
    st.error(
        "Não foi possível carregar ou executar o modelo neste ambiente. "
        "Isso pode ocorrer por limite de memória ou recursos da hospedagem."
    )
    st.exception(e)

st.markdown("---")
st.caption(
    "Manacá Lab — interface experimental. "
    "Projeto baseado no Manacá-1B."
)
