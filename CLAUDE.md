# Manacá Lab

Interface experimental para testar o Manacá-1B (`menezesbruno/manaca-1b-base`), modelo de linguagem base em português do Instituto de IA / LNCC, com foco em Ciência da Informação, informação jurídica e bibliotecas.

## Estrutura
- `manaca_lab.ipynb`: versão principal, para o Google Colab com GPU.
- `manaca_lab_demo.ipynb`: notebook de demonstração.
- `app.py`: versão em Streamlit (carrega o modelo com `transformers`; ver `README_streamlit.md`).
- `index.html`: página de apresentação do projeto.

## Como rodar
- Colab: abrir `manaca_lab.ipynb`, escolher GPU e executar tudo.
- Local: `pip install -r requirements.txt` e `streamlit run app.py`. Precisa de torch e de memória suficiente; sem GPU, a geração fica lenta.

## Convenções
- O Manacá-1B é um modelo base, não um chatbot: os prompts devem ser escritos como texto a ser continuado, não como perguntas e instruções.
- As respostas são experimentais e podem ter alucinações; manter esse aviso na interface.
- Próximas etapas previstas: PDFs, RAG, informação jurídica, avaliação de alucinações, comparação com outros modelos.
- Não há testes automatizados.
