import streamlit as st
import pickle
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_community.llms import HuggingFaceEndpoint
import os


load_dotenv(find_dotenv(), override=True)
os.environ.get('HUGGINGFACEHUB_API_TOKEN')

handler = StdOutCallbackHandler()

def main():
    st.header("Chat with your pdf by asking it questions")
    
    pdf = st.file_uploader("Upload your pdf file", type = "pdf")
    
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()
            
        # split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text = text)
        
        # store the pdf name as pickle if it has been selected and load it for the embeddings
        store_name = pdf.name[:-4]
        st.write(store_name)
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = HuggingFaceEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings Created')
        query = st.text_input("Ask Question from your PDF File")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                max_length=512,
                temperature=0.8,
                # huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            )
            chain = load_qa_chain(llm, chain_type = 'stuff')
            response = chain.invoke({"input_documents": docs, "question": query}, {"callbacks":[handler]})["output_text"]
            st.write(response)
            
if __name__ == '__main__':
    main()