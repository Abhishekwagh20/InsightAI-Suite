import asyncio
from openai import OpenAI
import config
import os
import requests
import json
from bs4 import BeautifulSoup

news_api_key = config.news_api_key


# Function to scrape the website
def scrape_website(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        content = "\n".join(f"<p>{p.get_text()}</p>" for p in paragraphs)
        
        # Wrap in basic HTML structure
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Scraped Content</title>
        </head>
        <body>
            {content}
        </body>
        </html>
        """
        
        return html_content
    else:
        return f"<p>Failed to retrieve content from {url}. Status code: {response.status_code}</p>"



def get_news(topic):
    """Get news information based on a given topic."""
    url = (
        f"https://newsapi.org/v2/everything?q={topic}&apikey={news_api_key}&pageSize=5"
    )
    try:
        response = requests.get(url)
        
        news = json.dumps(response.json(), indent=3)
        news_json = json.loads(news) 
        data = news_json
        
        articles = data["articles"]
        final_news = []
        
        for article in articles:
            source_name = article["source"]["name"]
            author = article["author"]
            title = article["title"]
            description = article["description"]
            url = article["url"]
            
            title_description = f""" 
                Title: {title},
                Author: {author},
                Source: {source_name},
                Description: {description},
                URL: {url}
            """
            final_news.append(title_description)
        return final_news
            
    except Exception as e:
        print("Something went wrong:", e)

def createAssistant(file_ids, title):
    # Create the OpenAI Client Instance
    client = OpenAI(api_key=config.API_KEY)

    instructions = """
    You are a helpful assistant. Use your knowledge base to answer user questions.
    """

    # The GPT model for the Assistant 
    model = "gpt-4o-mini" 

    tools = [{"type": "file_search"},
             {"type": "code_interpreter"},
             {"type":"function", 
             "function":{
                "name": "get_news",
                "description": "Get news title, author, description and URL which is similar to the news topic provided by the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "news topic provided by the user",
                        },
                     },
                    "required": ["topic"],
                }
             }},{
             "type":"function",
             "function":{
                 "name":"scrape_website",
                 "description":"Scrap the provided websited based on the url and store it as a html",
                 "parameters":{
                     "type":"object",
                     "properties":{
                         "url":{
                             "type":"string",
                             "description":"url of the websit to scrap"
                         },
                     },
                     "requires":["url"],
                 },
             }
             }]

    # CREATE VECTOR STORE
    vector_store = client.beta.vector_stores.create(name=title, file_ids=file_ids)
    tool_resources = {"file_search": {"vector_store_ids": [vector_store.id]}}

    # Create the Assistant
    assistant = client.beta.assistants.create(
        name=title,
        instructions=instructions,
        model=model,
        tools=tools,
        tool_resources=tool_resources
    )

    # Return the Assistant ID
    return assistant.id, vector_store.id

def saveFileOpenAI(location):
    # Create OpenAI Client
    client = OpenAI(api_key=config.API_KEY)
    
    # Open the file
    with open(location, "rb") as f:
        # Send file to OpenAI
        file = client.files.create(file=f, purpose='assistants')
    
    # Ensure the file is closed before attempting to remove it
    os.remove(location)

    # Return File ID
    return file.id

def startAssistantThread(prompt, vector_id):
    # Initiate Messages
    messages = [{"role": "user", "content": prompt}]
    # Create the OpenAI Client
    client = OpenAI(api_key=config.API_KEY)
    # Create the Thread
    tool_resources = {"file_search": {"vector_store_ids": [vector_id]}}
    thread = client.beta.threads.create(messages=messages, tool_resources=tool_resources)

    return thread.id

def runAssistant(thread_id, assistant_id):
    client = OpenAI(api_key=config.API_KEY)
    response_message = "No valid response generated."
    run = client.beta.threads.runs.create_and_poll(thread_id=thread_id, assistant_id=assistant_id, poll_interval_ms=1000)

    if run.status == 'requires_action':
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            if tool_call.function.name == "get_news":
                raw_arguments = tool_call.function.arguments
                try:
                    arguments = json.loads(raw_arguments)
                    output = get_news(arguments["topic"])
                    response_message = json.dumps(output)
                except json.JSONDecodeError as e:
                    response_message = "JSONDecodeError: " + str(e)
                except KeyError as e:
                    response_message = f"Missing required argument: {e}"
                finally:
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=run.thread_id,
                        run_id=run.id,
                        tool_outputs=[{
                            "tool_call_id": tool_call.id,
                            "output": response_message
                        }]
                    )
                
            elif tool_call.function.name == "scrape_website":
                raw_arguments = tool_call.function.arguments
                arguments = json.loads(raw_arguments)
                try:
                    output = scrape_website(arguments["url"])
                    response_message = output
                except Exception as e:
                    response_message = f"Error during scraping: {e}"
                
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=run.thread_id,
                    run_id=run.id,
                    tool_outputs=[{
                        "tool_call_id": tool_call.id,
                        "output": response_message
                    }]
                )
    return {"response": retrieveThread(run.thread_id)}


def retrieveThread(thread_id):
    client = OpenAI(api_key=config.API_KEY)
    thread_messages = client.beta.threads.messages.list(thread_id)
    list_messages = thread_messages.data 
    thread_messages = []
    for message in list_messages:
        obj = {}
        obj['content'] = message.content[0].text.value
        obj['role'] = message.role
        thread_messages.append(obj)
    return thread_messages[::-1]

def addMessageToThread(thread_id, prompt):
    client = OpenAI(api_key=config.API_KEY)
    thread_message = client.beta.threads.messages.create(thread_id, role="user", content=prompt)
