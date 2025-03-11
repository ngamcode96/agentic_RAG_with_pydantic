from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List
from utils import get_embedding

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(model_name=llm)

logfire.configure(send_to_logfire='if_token_present')

# setup Agent Dependencies
@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI


# system prompt
system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""


# define agent
pydantic_ai_expert = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)


## tools
# define retrieval tool
@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        #Get the user query embedding
        query_embedding = await get_embedding(user_query)

        #Query Supabase for relevant document
        result = ctx.deps.supabase.rpc(
            fn="match_site_pages",
            params={
                "query_embedding": query_embedding,
                "match_count": 5,
                'filter': {'source': 'pydantic_ai_docs'}

            }
        ).execute()

        if not result.data:
            return "No relevant documentation found"
        
        #Formating the result
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
            # {doc['title']}

            {doc['content']}
            """
            formatted_chunks.append(chunk_text)
        
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"
    


@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) ->List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages').select('url').eq('metadata->>source', 'pydantic_ai_docs').execute()
        
        if not result.data:
            return []
        
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []
    

@pydantic_ai_expert.tool
async  def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """

    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
        
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
        
            
        # Join everything together
        return "\n\n".join(formatted_content)
    
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
