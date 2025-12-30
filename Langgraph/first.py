from typing import Annotated, TypedDict
from langgraph import Graph, Node, Field, String, Integer
from langgraph.graph.message import add_message

class TestGraph(TypedDict):
    question: Annotated[str, add_message]
    context: Annotated[str, "Context for the question"]
    answer: Annotated[str, "Answer generated from the context"]
    message: Annotated[list, add_message]
    relevance: Annotated[str, "Relevance of the answer"]