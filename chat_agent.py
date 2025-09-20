# chat_agent.py

from smolagents import CodeAgent, LiteLLMModel, GoogleSearchTool
from dotenv import load_dotenv
import os

def initialize_chat_agent():
    """
    Loads API keys, initializes the model and tools, 
    and returns the configured ChatAgent AND the system prompt.
    """
    load_dotenv()
    key = os.getenv("KEY")
    Search_key = os.getenv("skey")

    model = LiteLLMModel(model_id="gpt-5-mini", api_key=key)
    os.environ["SERPER_API_KEY"] = Search_key

    MENTAL_HEALTH_SYSTEM_PROMPT = """
    You are an AI Mental Health Checker designed to act like a supportive mental health professional. 
    Your goals:
    1. Engage the user in a safe, respectful, and empathetic way, just like a mental health doctor or counselor would.
    2. Use active listening, empathy, and validation. For example: 
       - "That sounds really difficult, I understand why you feel this way."
       - "It’s okay to experience these emotions, let’s work through them together."
    3. Provide thoughtful guidance and gentle coping strategies, but do not claim to replace professional therapy.
    4. Never dismiss or judge the user’s feelings. Always encourage healthy coping mechanisms such as journaling, exercise, talking to loved ones, or professional therapy if necessary.
    5. If the user asks: "I want YouTube videos" or "I want articles on the web for motivation", 
       THEN:
         - Use the Web Search Agent tool.
         - Retrieve high-quality motivational YouTube videos or self-help articles.
         - Return results in this format:
           Title: <title of the video/article>
           Link: <URL>
           Summary: <short 2-3 sentence summary of content>
    6. Keep the conversation natural and supportive. Only use the Web Search tool when explicitly asked for external resources.
    7. If the query is sensitive (like suicide/self-harm), encourage the user to immediately reach out to a trusted person or a professional hotline.
    8. Answer in 2-3 lines like a normal person would answer, if needed then answer in more words.
    9. Talk in English until user ask you to speak other language
    """

    WebAgent = CodeAgent(
        model=model,
        tools=[GoogleSearchTool(provider="serper")],
        stream_outputs=False,
        name="Web_search_agent",
        description='It do web search for you, give it query as its input argument'
    )

    ChatAgent = CodeAgent(
        tools=[],
        model=model,
        stream_outputs=False,
        managed_agents=[WebAgent],
        name="Mental_Health_Agent",
        description="Empathetic mental health checker"
        # The 'system_prompt' argument has been removed from here
    )
    
    # We now return both the agent and the prompt text
    return ChatAgent, MENTAL_HEALTH_SYSTEM_PROMPT

# This part is for testing the agent directly
if __name__ == '__main__':
    print("Testing Chat Agent...")
    agent, system_prompt = initialize_chat_agent()
    test_query = "hello, I'm feeling a bit down today."
    full_prompt = f"{system_prompt}\nUser Query: {test_query}"
    response = agent.run(full_prompt)
    print(f"Query: {test_query}")
    print(f"Response: {response}")