from src.api.v1.agents.agents import run_search_agent
from src.core.guardrails import guard_input, guard_output

def query_documents(query: str):
    guard_input(query)

    result = run_search_agent(query)
    #return run_search_agent(query)

    if isinstance(result, dict) and result.get("answer"):
      result["answer"] = guard_output(result["answer"])
    return result
