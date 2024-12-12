import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI()
# print(llm.invoke("What is FAISS? How is it different from other vector stores?"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an English-French translator that return whatever the use says in French."),
    ("user", "{input}")
])

# Adding ouput parser to the chain
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# Retreival chain
loader = WebBaseLoader("https://blog.langchain.dev/langchain-v0-1-0/")
docs = loader.load()

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()

# Create vector store with shorter documents (relevant for our question)
documents = text_splitter.split_documents(docs)
# print(documents)
vectorstore = FAISS.from_documents(documents, embeddings)

# Create chain for documents
template = """"Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
"""

# creating a LCEL chain (new function in LangChain v0.1.0)
# prompt = ChatPromptTemplate.from_template(template)
# document_chain = create_stuff_documents_chain(llm, prompt)

# document_chain.invoke({
#     "input": "what is langchain 0.1.0?",
#     "context": [Document(page_content="langchain 0.1.0 is the new version of a llm app development framework.")]
# })

# Create retrieval chain for fetching url content similar to question & use that as context to answer the question
retriever = vectorstore.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# response = retrieval_chain.invoke({
#     "input": "what is new in langchain 0.1.0"
# })
# print(response['answer'])

# Creating conversational retrieval chain for remembering past conversation
# prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="chat_history"),  # populates prompt wiht chat history if variable is defined
#     ("user", "{input}"),
#     ("system", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation") # summarise conversation into a prompt

# ])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# mock chat history
chat_history = [
    HumanMessage(content="Is ther anything new about langchain 0.1.0?"),
    AIMessage(content="Yes")
]

# print(retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me more about it!"
# }))

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)

converstaional_retriever_chain = create_retrieval_chain(retriever_chain, document_chain)

response = converstaional_retriever_chain.invoke({
    'chat_history': chat_history,
    "input": "Tell me more about it!"
})

print(response['answer'])