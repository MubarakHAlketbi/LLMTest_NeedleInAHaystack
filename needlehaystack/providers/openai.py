from typing import Optional, Any, List, Dict
from operator import itemgetter
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import tiktoken
from .base_provider import BaseProvider

class OpenAI(BaseProvider):
    DEFAULT_MODEL_KWARGS: Dict[str, Any] = {"max_tokens": 300, "temperature": 0}

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0125",
        model_kwargs: Dict[str, Any] = None
    ):
        super().__init__(model_name, model_kwargs or self.DEFAULT_MODEL_KWARGS, 'NIAH_MODEL_API_KEY')
        self.model = AsyncOpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    def evaluate_model(self, prompt: List[Dict[str, str]]) -> str:
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            **self.model_kwargs
        )
        return response.choices[0].message.content
    
    def generate_prompt(self, context: str, retrieval_question: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"
            }
        ]
    
    def encode_text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        return self.tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> Any:
        template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        \n ------- \n 
        {context} 
        \n ------- \n
        Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        model = ChatOpenAI(temperature=0, model=self.model_name)
        chain = (
            {"context": lambda x: context, "question": itemgetter("question")}
            | prompt 
            | model 
        )
        return chain
