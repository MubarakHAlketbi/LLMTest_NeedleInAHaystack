from typing import Optional, Any, Tuple, List, Dict
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from cohere import AsyncClient, Client
from .base_provider import BaseProvider

class Cohere(BaseProvider):
    DEFAULT_MODEL_KWARGS = {"max_tokens": 50, "temperature": 0.3}

    def __init__(
        self,
        model_name: str = "command-r",
        model_kwargs: dict = None
    ):
        super().__init__(model_name, model_kwargs or self.DEFAULT_MODEL_KWARGS, 'NIAH_MODEL_API_KEY')
        self.client = Client(api_key=self.api_key)

    def evaluate_model(self, prompt: tuple[str, list[dict[str, str]]]) -> str:
        message, chat_history = prompt
        response = self.client.chat(
            message=message,
            chat_history=chat_history,
            model=self.model_name,
            **self.model_kwargs
        )
        return response.text

    def generate_prompt(self, context: str, retrieval_question: str) -> Tuple[str, List[Dict[str, str]]]:
        return (
            f"{retrieval_question} Don't give information outside the document or repeat your findings", 
            [{
                "role": "System",
                "message": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "User",
                "message": context
            }]
        )
    
    def encode_text_to_tokens(self, text: str) -> List[int]:
        if not text:
            return []
        return Client().tokenize(text=text, model=self.model_name).tokens

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        return Client().detokenize(tokens=tokens[:context_length], model=self.model_name).text

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
        
        model = ChatCohere(cohere_api_key=self.api_key, temperature=0.3, model=self.model_name)
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        chain = (
            {"context": lambda x: context, "question": itemgetter("question")}
            | prompt 
            | model 
        )
        return chain
