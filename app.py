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
        
def update_prompt(service):
    if service=="학사":
        file_path="prompt_education_courses.txt"
        return prompt_load(file_path)
    elif service=="로드맵":
        file_path="prompt_major_roadmap.txt"
        return prompt_load(file_path)
    elif service=="졸업":

        prompt = '''
    너는 사용자가 강남대학교에서 졸업할 수 있는지 등을 물어보았을때 성실하게 답변해주는 학사지원 인공지능 챗봇이야.
    사용자가 자신의 수강과목에 대한 정보를 제공하면, 나는 그 정보를 기반으로 사용자가 졸업할 수 있는지, 졸업하기 위해 어떤게 더 필요한지 여부를 알려줄 수 있어.
    교육과정표를 보고 앞으로 어떤 과목을 들어야 하는지 알려줄 수 있어.
    교양 ,주전공, 복수전공, 등을 나눠서 설명해줘야해
    주전공에서 전기는 전공 기초고 전선은 전공 선택이야. 
    복수전공은 복기라고 된것과 복수 라고된것이 있어.
    마이크로 전공은 부가정보라서 주전공이나 복수전공으로 포함되지 않아.
    
    총학점은 졸업이수학점에서 확인할 수 있어. 
    
    졸업계획을 세울때는 조기졸업과 일반졸업을 고려해야해. 단 조기 졸업은 사용자의 전체 학점이 4.0 이상일때만 작성해.
    그리고 들어야할 과목은 학생의 입학년도에 맞는 교육과정표에 있는 과목으로, 사용자가 현재 듣고 있는 과목을 기반으로 추천해야해.
    예를들어 
    1. 20학년도 편입생이고 현재 3학년 2학기를 하고있다면, 2020년도 입학자 적용 교육과정표에 있는 과목중 4학년 2학기 과목이나, 이전과목중 추가로 수강하면 좋은 과목을 추천해야해. 
    2. 22학년도 신입생이고 현재 3학년 1학기를 하고있다면, 2022년도 입학자 적용 교육과정표에있는 과목 중 3학년 2학기 과목이나, 이전과목중 추가로 수강하면 좋은 과목을 추천해야해.
    
    [사용자 이름] 개인별 수강 과목 리스트에서 성명으로 적혀있어
    [사용자의 학부] 개인별 수강 과목 리스트에서 학부(과)를 참고해
    [사용자의 학과] 개인별 수강 과목 리스트에서 전공을 참고해
    [사용자의 전공 구분] 개인별 수강 과목 리스트에서 전공구분을 참고해
    [사용자의 복수전공,부전공] 개인별 수강 과목 리스트에서 복수전공, 부전공을 참고해 둘중 하나만 있어
    [입학년도]:사용자의 학번을 보고 2022로 시작하면 22학년도 입학자 2020이면 20년도 입학자야
    
    예를들어 사용자가 졸업 요건을 확인하고 싶어할 때의 출력 양식은 다음과 같아
    
    학점은 졸업이수학점에서 [입학년도]와 [사용자의 학부],[사용자의 학과],[사용자의 전공 구분]  맞게 선택되야해. 
    예를들어 2022학년도에 입학한 ICT융합공학부 일반전공 학생은 2021학년도이후 ICT융합공학부 일반전공 입학자는 복수전공이 필수로 선택되야해서 교양학점의 기초교양에 17학점(채플4회, 인성과학문 4회), 균형교양이 15학점(5개 영역에서 각1개), 주전공(1전공 학점)이 57학점, 복수전공 학점이 36학점이 되는거고 최소 졸업학점이 130학점이 되야하는거야
    
    
    [사용자 이름] 학생님의 강남대학교 졸업요건 및 현재 수강 과목들을 기반으로 졸업 가능 여부를 평가하고 필요한 부분을 안내드리겠습니다.

    ## 졸업 필요 학점 및 과목 요건
    ### 강남대학교 [사용자의 학부] [사용자의 학과] [사용자의 전공 구분]의 [입학년도]학년도 입학자의 졸업요건은 다음과 같습니다:

    |구분|이수 기준 학점| 현재 취득 학점 | 학점 평균 |현재 수강 학점 | 남은 학점 | 비고|
    |---|---|---|---|---|---|---|
    |기초교양| 00학점[[사용자의 학부][사용자의 전공 구분]의 기초교양 이수 기준 학점] | 00학점[개인별 수강 과목 리스트의 이수 구분 '기초'의 학점 합계]|  0.0 [개인별 수강 과목 리스트의 이수 구분 '기초'의 성적 평균 이때 P,F,R인 과목은 제외하고 학점 계산]| [현재 수강 정보에서 '기초' 학점]  | [기초 교양 학점 합계 - 개인별 수강 과목 리스트의 '기초'의 학점 합계 - 현재 수강 중인 과목 중 이수구분 '기초'의 학점 합계] | 채플 [수강 횟수]/[이수 기준]회, 인성과학문 [수강 횟수]/[이수 기준]회 |
    |균형교양| 00학점[[사용자의 학부][사용자의 전공 구분]의 균형교양 이수 기준 학점] | 00학점[개인별 수강 과목 리스트의 이수 구분 '균형'의 학점 합계] | 0.0 [개인별 수강 과목 리스트의 이수 구분 '균형'의 성적 평균]|  |  [사용자가 현재 듣고있지 않거나 들은적 없는 균형교양의 영역 표시 없으면 비우기]|
    |주전공([사용자의 전공 구분])| 00학점[[사용자의 학부][사용자의 전공 구분]의 제1전공 이수 기준 학점] |  | 0.0 [개인별 수강 과목 리스트의 이수 구분 '전선','전기'의 합의 성적 평균] | [주전공의 이수 기준학점 - 현재 취득학점 - 현재 수강학점] |  |
    |복수전공| 00학점[[사용자의 학부][사용자의 전공 구분]의 복수(부)전공 이수 기준 학점] |  |  |  [(부)복수 전공의 이수 기준학점 - 현재 취득학점 - 현재 수강학점] |  |
    |총 이수학점| 00학점[[사용자의 학부][사용자의 전공 구분]의 이수 기준 학점] |  |  |  |  [개인별 수강 과목 리스트에서 천체 합계 + 현재 수강중인 신청학점|
    
    추가적인 소프트웨어 관련 과목들을 계속해서 이수하는 것을 권장드립니다.
    이제 35 학점을 더 채우기 위한 구체적인 계획을 세우는 것이 좋겠습니다. 매 학기 18-21 학점씩 수강하면, 남은 2-3학기 내 졸업이 가능할 것입니다.
    ## 조기 졸업 계획
    조기 졸업을 위해서는 다음과 같은 계획을 수립하시는 것이 좋습니다:
    
    
    ## 일반 졸업 계획
    일반적인 졸업을 위해서는 다음과 같은 계획을 수립하시는 것이 좋습니다:
    매 학기 15-18 학점씩 수강하여, 남은 학기 내에 졸업 가능합니다.
    구체적으로는 다음과 같습니다.
    
    - 2024년 1학기: OO 학점 
    
    - 2024년 2학기: OO 학점 
    
    - 2025년 1학기: OO 학점 
    
    - 2025년 2학기: OO 학점 
    
    ## 종합
    현재 진행한 및 계획한 수강 과목들을 잘 이수하면 졸업요건을 충족할 수 있습니다.
    또한, 졸업종합평가(P/F) 과목을 반드시 심사 시기에 맞춰 이수하세요.
    지금까지의 성적관리와 학점 이수를 기반으로 충분히 졸업 가능성을 보유하고 있습니다. 앞으로도 성실히 수강 과목들을 완료하시면 무난하게 졸업할 수 있을 것입니다.
    '''
    return prompt

documents = []

# PDF와 텍스트 파일을 모두 로드할 디렉토리 경로
pdf_directory = "./data"
text_directory = "./crawled_texts"



# Load and index documents from both PDF and text files


#Load Prompt from a txt file
def prompt_load(file_path):
    file_content=""
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content


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

################################################
if "prompt" not in st.session_state:

    if st.session_state["service"] == "졸업":
        file_path="prompt_graduation.txt"
        st.session_state["prompt"] = prompt_load(file_path)
        
    elif st.session_state["service"] == "학사":
        file_path="prompt_education_courses.txt"
        st.session_state["prompt"] = prompt_load(file_path)
                    
    elif st.session_state["service"] == "로드맵":
        file_path="major_roadmap.txt"
        st.session_state["prompt"] = prompt_load(file_path)
        st.session_state["prompt"] = ''' 
        로드맵 프롬프트는 여기에 입력
    
    '''
    else:
        st.session_state["prompt"] = '''
        서비스가 선택되지 않았습니다.
    '''    
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
        st.session_state["service"] = st.radio("서비스", options=["졸업","학사","로드맵"])
        st.session_state["prompt"] = update_prompt(st.session_state["service"])
        st.write()
        if st.session_state["service"] =="졸업":
            st.markdown("### 파일 입력")
            with st.expander("파일 찾는 방법", expanded=False):
                st.markdown('''
                            #### 개인별 수강 과목 리스트
                            - 학생 본인의 수강 과목 리스트를 업로드해주세요.
                            - PDF 파일만 업로드 가능합니다.
                            - 강남대학교 홈페이지 -> 
                            - 종합정보시스템 -> 개인별이수과목출력 -> PDF 저장
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
            
    if st.session_state["service"] == "졸업":
        st.title("강남대학교 졸업지원 챗봇")       
    if st.session_state["service"] == "학사":
        st.title("강남대학교 학사지원 챗봇")
    if st.session_state["service"] == "로드맵":
        st.title("강남대학교 로드맵지원 챗봇")
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

    ###1.학사지원 서비스 
    if st.session_state["service"] == "학사":
        if prompt := st.chat_input("질문을 입력하세요."):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("ai"):
            
                response = chatbot.generate(str(st.session_state.chat_history[-2:])+f"\n\n{prompt}")
            
                st.write_stream(stream_data(response))
            st.session_state.chat_history.append({"role": "user", "message": prompt})
            st.session_state.chat_history.append({"role": "ai", "message": response})
            
            
    if st.session_state["service"] == "졸업":
        if prompt := st.chat_input("질문을 입력하세요."):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("ai"):
                if len(st.session_state.chat_history) > 2:
                    querry = "\n\n이건 내가 이전에 수강했던 과목들이야."+str(st.session_state["previous"])+"\n\n이건 내가 지금 듣고 있는 과목들이야."+ str(st.session_state["current"]) + "\n\n 이건 이전대화야"+str(st.session_state.chat_history[-2:]) + f"\n\n 이건 내 질문이야.{prompt}"
                    response = chatbot.generate(querry)
                else:
                    querry = "\n\n이건 내가 이전에 수강했던 과목들이야."+str(st.session_state["previous"])+"\n\n이건 내가 지금 듣고 있는 과목들이야."+ str(st.session_state["current"])  + f"\n\n 이건 내 질문이야.{prompt}"
                    response = chatbot.generate(querry)
                
                st.write_stream(stream_data(response))
            
            st.session_state.chat_history.append({"role": "user", "message": prompt})
            st.session_state.chat_history.append({"role": "ai", "message": response})
 
    if st.session_state["service"] == "로드맵":
        if prompt := st.chat_input("질문을 입력하세요."):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("ai"):
                response = chatbot.generate(str(st.session_state.chat_history[-2:])+f"\n\n{prompt}")
            
                st.write_stream(stream_data(response))

            st.session_state.chat_history.append({"role": "user", "message": prompt})
            st.session_state.chat_history.append({"role": "ai", "message": response})
        