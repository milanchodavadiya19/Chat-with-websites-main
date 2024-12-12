# RAG Chain to chat with any website using LangChain v0.1.0
Retrieval Augmented Generation to chat with any website using:
- Langchain v0.1.0
- LangChain Expression Language (LCEL)
- Chroma Vector Database
- Streamlit

## Features
- **Interaction:** Utilizing LangChain's latest version to engage with websites, facilitating information extraction from diverse online sources.
- **Integration:** Compatible with models including GPT-4, Mistral, Llama2, and ollama. Currently employing GPT-4.
- **Streamlit GUI:** Neat, user-friendly interface developed using Streamlit.

## GUI Snapshot
![UI (Chat-with-websites)](https://github.com/SidEnigma/Chat-with-websites/assets/19359983/619b62c2-c971-4fe2-80bd-cfd77711ca81)

## Typical Retrieval Step for RAG
RAG is an acronym denoting Retrieval-Augmented Generation and operates on the principle of enhancing the knowledge base of an LLM by integrating additional information provided within the prompt and chat history. This augmentation process involves vectorizing the textual data, subsequently identifying the most closely similar text segments to the given prompt within the vectorized corpus. These selected segments are then utilized as prefixes for the LLM, thereby enriching its contextual understanding and potentially improving the quality of generated responses.

![retrieval_step](https://github.com/SidEnigma/Chat-with-websites/assets/19359983/d2a35d15-e292-45d0-a5a0-010a9090df4f)
