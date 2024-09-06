import os
from typing import Optional, Any, List, Dict
from operator import itemgetter
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from anthropic import Anthropic as AnthropicClient
from .base_provider import BaseProvider

class Anthropic(BaseProvider):
    DEFAULT_MODEL_KWARGS = {"max_tokens": 300, "temperature": 0}

    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        model_kwargs: dict = None
    ):
        super().__init__(model_name, model_kwargs or self.DEFAULT_MODEL_KWARGS, 'NIAH_MODEL_API_KEY')
        self.client = AnthropicClient(api_key=self.api_key)
        self.tokenizer = self.client.get_tokenizer()

    def evaluate_model(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.model_kwargs.get("max_tokens", 300),
            temperature=self.model_kwargs.get("temperature", 0),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.content[0].text

    def generate_prompt(self, context: str, retrieval_question: str) -> str:
        return f"""You are a helpful AI bot that answers questions for a user. Keep your response short and direct.

Context:
{context}

Question: {retrieval_question}

Don't give information outside the document or repeat your findings."""
    
    def encode_text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids
    
    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        return self.tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> Any:
        template = """Human: You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        <document_content>
        {context} 
        </document_content>
        Here is the user question:
        <question>
         {question}
        </question>
        Don't give information outside the document or repeat your findings.
        Assistant: Here is the most relevant information in the documents:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        model = ChatAnthropic(temperature=0, model=self.model_name)
        chain = (
            {"context": lambda x: context, "question": itemgetter("question")}
            | prompt 
            | model 
        )
        return chain
