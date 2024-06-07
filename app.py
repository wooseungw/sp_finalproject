import streamlit as st
import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

import time
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from chatbot_class import Chatbot

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

# https://velog.io/@wonjun12/Streamlit-%ED%8C%8C%EC%9D%BC-%EC%97%85%EB%A1%9C%EB%93%9C
# 디렉토리 이름, 파일을 주면 해당 디렉토리에 파일을 저장해주는 함수
def save_uploaded_file(directory, file):
    # 1. 저장할 디렉토리(폴더) 있는지 확인
    #   없다면 디렉토리를 먼저 만든다.
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 2. 디렉토리가 있으니, 파일 저장
    with open(os.path.join(directory, f"{directory.split('/')[-1]}.pdf"), 'wb') as f:
        f.write(file.getbuffer())
    return st.success('파일 업로드 성공!')

# PDF 파일들을 로드하여 분할한 뒤 텍스트를 벡터로 변환하여 DB에 저장
def load_and_index_documents(pdf_directory, text_directory, api_key):
    documents = []

    # PDF 파일들을 로드하여 분할
    pdf_files = glob(os.path.join(pdf_directory, '*.pdf'))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        documents.extend(pdf_documents)
    print(f"Loaded {len(documents)} documents from {pdf_directory}")
    # 텍스트 파일들을 로드하여 분할
    text_files = glob(os.path.join(text_directory, '*.txt'))
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as file:
            text = file.read()
            # 텍스트를 Document 객체로 변환
            text_document = Document(page_content=text, metadata={"source": text_file})
            documents.append(text_document)
    print(f"Loaded {len(documents)} documents from {text_directory}")
    # 분할된 텍스트를 벡터로 변환하여 ChromaDB에 저장
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = chunk_splitter.split_documents(documents)
    print(f"Split {len(chunks)} chunks from {len(documents)} documents")
    print("Embedding documents...",end="")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    print("done.")
    print("Indexing documents...",end="")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("done.")
    retriever = vectordb.as_retriever()
    
    return retriever

def pdf_load(dir):
    input_docs = []
    # Load all PDF files using PyPDFLoader
    input_pdf_files = glob(os.path.join(dir, '*.pdf'))
    for pdf_file in input_pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        input_docs.extend(pdf_documents)
        
    return input_docs
        

documents = []

# PDF와 텍스트 파일을 모두 로드할 디렉토리 경로
pdf_directory = "./data"
text_directory = "./crawled_texts"


# Load and index documents from both PDF and text files



if "OPENAI_API" not in st.session_state:
    st.session_state["OPENAI_API"] = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else ""
    # print(st.session_state["OPENAI_API"])
# 기본 모델을 설정합니다.
if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o"
# 채팅 기록을 초기화합니다.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "sevice" not in st.session_state:
    st.session_state["service"] = "수업"

if "previous" not in st.session_state:
    st.session_state["previous"] =""

if "current" not in st.session_state:
    st.session_state["current"] = ""

#################################################
if "prompt" not in st.session_state:
    st.session_state["prompt"] = ''' 
    수업 프롬프트는 여기에 입력
    
    ''' if  st.session_state["service"] == "수업" else '''
    
    졸업 프롬프트는 여기에 입력
    '''
#################################################

if "retriever" not in st.session_state:
    st.session_state.retriever = load_and_index_documents(pdf_directory, text_directory, st.session_state["OPENAI_API"])
    
# pdf를 사용해서 pdf(논문)을 모두 로드

if __name__ == '__main__':
    # 설정을 위한 사이드바를 생성합니다.
    with st.sidebar:
        st.title("설정")
        st.session_state["OPENAI_API"] = st.text_input("Enter API Key", st.session_state["OPENAI_API"], type="password")
        #모델을 선택합니다.
        st.session_state["model"] = st.selectbox("모델", options=["gpt-4o", "gpt-3.5-turbo"])
    
        #라디오 버튼을 사용하여 서비스를 선택합니다.
        st.session_state["service"] = st.radio("서비스", options=["수업", "졸업"])
        st.write()
        if st.session_state["service"] =="졸업":
            st.markdown("### 파일 입력")
            with st.expander("파일 찾는 방법", expanded=False):
                st.markdown('''
                            #### 개인별 수강 과목 리스트
                            - 학생 본인의 수강 과목 리스트를 업로드해주세요.
                            - PDF 파일만 업로드 가능합니다.
                            - 강남대학교 홈페이지 -> 
                            - 종합정보시스템 -> 학적관리 -> 개인별이수과목출력 -> PDF 저장
                            #### 현재 시간표
                            - 학생 본인의 현재 시간표를 업로드해주세요.
                            - PDF 파일만 업로드 가능합니다.
                            - 강남대학교 홈페이지 ->
                            - 종합정보시스템 -> 수업관리 -> 수강신청조회 -> 조회 -> PDF저장
                            ''')
            previous = st.file_uploader("개인별 수강 과목 리스트", type=["pdf"])
            if previous is not None:
                save_uploaded_file("./previous", previous)
                st.session_state["previous"] = pdf_load("./previous")
            current = st.file_uploader("현재 시간표", type=["pdf"])
            if current is not None:
                save_uploaded_file("./current", current)
                st.session_state["current"] = pdf_load("./current")
        if st.button("초기화"):
            st.session_state.chat_history = []
            st.session_state["service"] = "수업"
            st.session_state["previous"] = ""
            st.session_state["current"] = ""
            previous = ""
            current = ""
            st.rerun()
            
            
    if st.session_state["service"] == "수업":
        st.title("강남대학교 수업지원 챗봇")
    if st.session_state["service"] == "졸업":
        st.title("강남대학교 졸업지원 챗봇")
    ###############################
    # Create a sidebar for API key and model selection
    with st.expander("챗봇 사용법", expanded=False):
        st.markdown("""
                    - 강남대학교 학사지원을 위한 챗봇입니다.
                    - 답변 내용은 학사지원 메뉴얼을 기반으로 합니다.
                    """)
    ################# 설정을 위한 사이드바를 생성합니다. 여기서 api키를 받아야 실행됩니다. ##########################################
    
            
        
            
    
    # Chatbot을 생성합니다.
    chatbot = Chatbot(api_key=st.session_state["OPENAI_API"],
                       retriever=st.session_state.retriever,
                       sys_prompt=st.session_state["prompt"],
                       model_name=st.session_state["model"])



    ############################################ 실제 챗봇을 사용하기 위한 Streamlit 코드 ###################################################
    for content in st.session_state.chat_history:
        with st.chat_message(content["role"]):
            st.markdown(content['message'])    
    ### 사용자의 입력을 출력하고 생성된 답변을 출력합니다.
    if prompt := st.chat_input("질문을 입력하세요."):
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("ai"):
            
            
            response = chatbot.invoke(str(st.session_state.chat_history[-2:])+f"\n\n{prompt}")
            
            st.write_stream(stream_data(response))
            
    st.session_state.chat_history.append({"role": "user", "message": prompt})
    st.session_state.chat_history.ap