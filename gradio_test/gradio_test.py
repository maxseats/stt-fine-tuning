from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import os
import yaml 

# clova  사용 용 파일 임포트
from chat_completions import CompletionExecutor,parse_response,RequestData

# .env 파일에서 환경 변수 로드
from dotenv import load_dotenv
import os



# 아 pip install -U langchain-community 해야하는데 이때 클로바 사용시의 requrment 랑 좀 달라질수도 있음 기억해

# YAML 파일 경로
# yaml_file = '/mnt/a/yeh-jeans/gradio_test/secrets.yaml'

# # YAML 파일 읽기
# with open(yaml_file, 'r', encoding='utf-8') as file:
#     yaml_content = yaml.safe_load(file)


# # 특정 키 값 읽기, 실험용 GPT key 값
# gpt_key = yaml_content.get('gpt_key')
# api_key = yaml_content['test'].get('api_key')

# os.environ["OPENAI_API_KEY"] = gpt_key

# llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')


load_dotenv()
API_KEY = os.getenv("API_KEY")
API_KEY_PRIMARY_VAL = os.getenv("API_KEY_PRIMARY_VAL")
REQUEST_ID = os.getenv("REQUEST_ID")
TEST_APP_ID = os.getenv("TEST_APP_ID")

# 새로운 response 
def clova_response(user_input,history,additional_input_info):
    preset_text = [
        {"role": "system", "content": "사용자의 질문에 답변합니다."},
        {"role": "user", "content":user_input }, # 사용자가 textbox 에서 입력한 내용을 content 로
    ]
    
    request_data = RequestData(messages=preset_text).to_dict()

    completion_executor = CompletionExecutor(
        api_key=API_KEY,
        api_key_primary_val=API_KEY_PRIMARY_VAL,
        request_id=REQUEST_ID,
        test_app_id=TEST_APP_ID,
    )
    
    response = completion_executor.execute(request_data)
    response = parse_response(response)
    return response



def response(message, history, additional_input_info):
        history_langchain_format = []
        # additional_input_info로 받은 시스템 프롬프트를 랭체인에게 전달할 메시지에 포함시킨다.
        history_langchain_format.append(SystemMessage(content= additional_input_info))
        for human, ai in history:
                history_langchain_format.append(HumanMessage(content=human))
                history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = llm(history_langchain_format)
        return gpt_response.content
'''
    중요!!! gradio 의 매개변수 맵핑 방법!!
    이 fn 아래로, 사용된 변수들이 차례차례 맵핑된다. 물론 맵핑될수없는 애는 건너뛴다.
    예를 들어, 이 아래의 textbox, chatbot,additional_inputs 변수는 fn의 
    def method (val1(textbox 값), val2(chatbot값 ), val3(additional_inputs값))
    이런식으로 매칭된다. title 이나 description 같은 입력이 없는 변수들은 매개변수로 맵핑되지 않는다.
'''
gr.ChatInterface(
        fn=clova_response, # 사용자가 인터페이스에 입력을 제출할때 호출되는 함수
       
        textbox=gr.Textbox(placeholder="말걸어주세요..", container=False, scale=7),
        
        # 채팅창의 크기를 조절 및 자동으로 history 라는 변수에 대화 내용을 저장하고 관리하는 메서드
        chatbot=gr.Chatbot(height=1000), 
        title="어떤 챗봇을 원하심미까?",
        description="물어보면 답하는 챗봇임미다.",
        theme="soft",
        examples=[["안뇽"], ["요즘 덥다 ㅠㅠ"], ["점심메뉴 추천바람, 짜장 짬뽕 택 1"]],
        retry_btn="다시보내기 ↩",
        undo_btn="이전챗 삭제 ❌",
        clear_btn="전챗 삭제 💫",
        additional_inputs=[
            gr.Textbox("", label="System Prompt를 입력해주세요", placeholder="I'm lovely chatbot.")
        ]
).launch()
