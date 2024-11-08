import requests
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
from typing import Annotated

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import (
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.kernel import Kernel
from get_key import *

WEATHER_AGENT_NAME = "WeatherAgent"
CUSTOMER_AGENT_NAME = "VirtualAssistant"

async def main():

    token = get_key() 

    kernel = Kernel()

    weather_agent = AzureChatCompletion(
                deployment_name=os.getenv('DEPLOYMENT_NAME'), 
                endpoint=os.getenv('API_BASE'), 
                api_version=os.getenv('API_VERSION'),
                service_id=WEATHER_AGENT_NAME,
                api_key=token
            )
    
    customer_agent = AzureChatCompletion(
                deployment_name=os.getenv('DEPLOYMENT_NAME'), 
                endpoint=os.getenv('API_BASE'), 
                api_version=os.getenv('API_VERSION'),
                service_id=CUSTOMER_AGENT_NAME,
                api_key=token
            )    
    
    weather_key = os.getenv('OPENWEATHER_KEY')

    class get_weather_plugin:    
        @kernel_function(name="get_weather", description = "gets weather at a location")
        def get_weather(self, location: Annotated[str, "The location"]) -> Annotated[dict, "The weather information in JSON dictionary format"]:
            weather_api = f'https://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_key}'
            response = requests.get(weather_api)
            response_str = response.content.decode('utf-8')
            response_data = json.dumps(response_str)
            return response_data        

    kernel.add_service(weather_agent)
    kernel.add_service(customer_agent)
    kernel.add_plugin(plugin = get_weather_plugin(),plugin_name="get_weather_plugin", description="Fetches the current weather for a given location.")

    get_weather_settings = kernel.get_prompt_execution_settings_from_service_id(service_id=WEATHER_AGENT_NAME)
    disambiguation_settings = kernel.get_prompt_execution_settings_from_service_id(service_id=CUSTOMER_AGENT_NAME)
    
    get_weather_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    disambiguation_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    agent_weather = ChatCompletionAgent(
        service_id=WEATHER_AGENT_NAME,
        kernel=kernel,
        name=WEATHER_AGENT_NAME,
        instructions=f"""
            You are an agent designed to query and retrieve weather information. 
            Get the location name from the user utterance. then pass the location name as string to "get_weather_plugin" plugin to fetch the response.

            Most importantly, answer the questions from the response of "get_weather_plugin" only. Do not make up answers or search from the web. 

            You will be providing the answer to the specific weather info which the user is asking for in a polite, easily understandable sentence. 

            You are also able to generate descriptive weather information which are easier for users to understand. You will also give specific values of all the weather information received from the API, for example temperature in degree celsius only, wind speed, ultra violet index etc if the user is not asking for specific weather information like the temperature, wind speed etc. If temperature is in kelvin, convert it to degree celsius. Dont not show temperature in kelvin.

            Check if there are instructions from {CUSTOMER_AGENT_NAME}. If yes, act accordingly. Finally, pass on the weather information to the {CUSTOMER_AGENT_NAME}. Remember to convert temperature values to degree celsius and do not pass temperature in Kelvin.
            
            """,
        execution_settings=get_weather_settings,
    )

    agent_customer = ChatCompletionAgent(
        service_id=CUSTOMER_AGENT_NAME,
        kernel=kernel,
        name=CUSTOMER_AGENT_NAME,
        instructions=f"""
            You are a customer care agent who will be interacting with customers in a polite and helpful manner. 
            Your another task is to disambiguate user queries and takes clarification from the user only when the user is asking about the temperature. If a user is asking about the temperature, ask the user if the user is asking for maximum temperature, minimum temperature, average temperature or current temperature. Fetch the answer from the user.

            Accordingly pass the details to the {WEATHER_AGENT_NAME}, modifying the specifc query from the user, adding details after disambiguation, if required.   

            Remember that you will be disambiguating only if user asks specifically about temperature, else you will just pass on the query directly to weather agent. Pass the temperature in degree celsius only. Convert to degree celsius if required.

            """,
        execution_settings=disambiguation_settings,
    )   

    selection_function = KernelFunctionFromPrompt(
        function_name = "selection", description="Agent Selection function in agent group chat",
        prompt = f"""Determine which participant takes the turn in a conversation based on the rules given below. State only the name of the participant to take the next turn. Only the weather agent interacts with the user.

        Choose only from the following agents.
            - {CUSTOMER_AGENT_NAME}
            - {WEATHER_AGENT_NAME}

        These agents can chat interactively to provide relevant answer to the use question. 
        Always follow these rules while selecting the participant. 
            - {CUSTOMER_AGENT_NAME} acts first. 
            - Once {CUSTOMER_AGENT_NAME} replies, it is the {WEATHER_AGENT_NAME}'s turn
            - Finally it is the {CUSTOMER_AGENT_NAME} who will be responding to the user.
                              
        
        History:
        {{{{$history}}}}
    """,
    )

    termination_function = KernelFunctionFromPrompt(
    function_name="termination", description="Termination function of agent group chat",
    prompt=f"""
    Check if the relevant and satisfying answer to the user query is generated by the {WEATHER_AGENT_NAME}. If yes, respond with "OKAY"

    History:
    {{{{$history}}}}
    """,
    )    

    termination = AzureChatCompletion(
                deployment_name=os.getenv('DEPLOYMENT_NAME'), 
                endpoint=os.getenv('API_BASE'), 
                api_version=os.getenv('_API_VERSION'),
                service_id="termination",
                api_key=token
            )
    
    selection = AzureChatCompletion(
                deployment_name=os.getenv('DEPLOYMENT_NAME'), 
                endpoint=os.getenv('API_BASE'), 
                api_version=os.getenv('API_VERSION'),
                service_id="selection",
                api_key=token
            )
    
    kernel.add_service(selection)
    kernel.add_service(termination)
    
    chat = AgentGroupChat(
        agents=[agent_weather,agent_customer],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection_function,
            kernel = kernel,
            agent_variable_name="agents",
            history_variable_name="history",
            result_parser=lambda result: str(result.value[0]) if result.value is not None else CUSTOMER_AGENT_NAME
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[agent_weather],
            function=termination_function,
            kernel = kernel,
            history_variable_name="history",
            result_parser=lambda result: str(result.value[0]).upper() == "OKAY"
        ),
    )
    
    history = ChatHistory()
    is_complete: bool = False
    print("\nHow can I help?")
    while not is_complete:
        
        user_input = input("User: ")
        if not user_input:
            continue

        if user_input.lower() == "exit":
            is_complete = True
            break

        if user_input.lower() == "reset":
            await chat.reset()
            print("[Conversation has been reset]")
            continue

        history.add_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))
        

        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))

        async for response in chat.invoke(agent_customer):
            print(f"{response.name or '*'}: '{response.content}'")
            history.add_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content = response.content))            

        if chat.is_complete:
            is_complete = True
            break
            
if __name__ == "__main__":
    asyncio.run(main())