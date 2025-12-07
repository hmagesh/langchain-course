from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """Elon Musk is a South African-born American entrepreneur known for co-founding PayPal and leading major tech companies like Tesla (electric vehicles) and SpaceX (aerospace), aiming to revolutionize transport and energy. He also acquired Twitter (now X), co-founded Neuralink (brain-computer interfaces), and The Boring Company (tunneling). Musk is famous for ambitious goals, such as colonizing Mars, using "first principles" thinking, and transforming industries, 
    despite often controversial management and public behavior."""
    summary_template = """
    Summarize the following information:
    {information} in 10 words or less and also give an interestig fact
    """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    #llm =  ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0)
    llm = ChatOllama(model="gemma3:270m",temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
if __name__ == "__main__":
    main()
