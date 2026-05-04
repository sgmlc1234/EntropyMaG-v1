import json
import random
import logging
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langsmith.run_helpers import get_current_run_tree, tracing_context

from tools import extract_json_from_text, run_python_code
from config import get_llm_config
from deepagent.tracing import invoke_structured_with_slim_trace
from prompts import (
    build_orchestrator_context_block,
    GeneratedProblemSchema,
    MODIFIER_EXECUTION_SYSTEM_PROMPT,
    OrchestratorDispatchSchema,
    OrchestratorPlanSchema,
    build_modifier_prompt,
)

# 로거 설정
logger = logging.getLogger("modifier")

# Initialize LLM with OpenRouter config
config = get_llm_config("modifier")
llm = ChatOpenAI(
    model=config["model"],
    temperature=0.7,
    max_tokens=config.get("max_tokens", 2000),
    timeout=config.get("timeout", 120),
    max_retries=config.get("max_retries", 3),
    api_key=config["api_key"],
    base_url=config["base_url"]
)




def _format_tool_evidence(messages) -> str:
    snippets = []
    for msg in messages or []:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "tool")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            snippets.append(f"[{name}]\n{content[:800]}")
    return "\n\n".join(snippets[-4:]) if snippets else "No tool evidence captured."

def modify_problem(problem: Dict, user_request: str, invoke_config: Dict = None) -> Dict:
    """
    Modifies a problem based on user request using an Agent with Code Execution tool.
    
    Args:
        problem: The problem dict to modify
        user_request: User's modification request
    
    Returns:
        Modified problem dict, or None if modification failed
    """
    # Create an agent with code execution tool
    tools = [run_python_code]
    agent_executor = create_react_agent(llm, tools)
    
    # 간결한 문제 요약
    prob_summary = f"""ID: {problem.get('id')}
Statement: {problem.get('statement', '')[:200]}...
Answer: {problem.get('answer', '')}"""

    prompt_content = build_modifier_prompt(
        problem_summary=prob_summary,
        user_request=user_request,
        difficulty_hint=str(problem.get("difficulty", "Medium")),
    )
    
    plan_payload = OrchestratorPlanSchema(
        stage="modifier_stage",
        objective="Apply the requested modification while preserving mathematical correctness.",
        strategy_summary="Modify the target problem conservatively unless the request explicitly asks for a deeper rewrite.",
        hard_constraints=[
            "Preserve the target problem's mathematical identity unless explicitly told otherwise.",
            "Use run_python_code when calculations are needed.",
            "Return JSON only.",
        ],
        success_criteria=[
            "Return a valid GeneratedProblemSchema object.",
            "Keep statement, answer, solution, and code aligned.",
            "Do not output prose outside the JSON object.",
        ],
    ).model_dump()
    dispatch_payload = OrchestratorDispatchSchema(
        agent_role="modifier",
        slot=0,
        op_type="modify",
        parent_ids=[problem.get("id", "")],
        invariant_notes="Preserve the original problem's mathematical identity unless the dispatch request explicitly allows a rewrite.",
        task_payload={
            "problem_id": problem.get("id", ""),
            "user_request": user_request,
            "difficulty_hint": str(problem.get("difficulty", "Medium")),
        },
        finish_when=[
            "Return a modified problem JSON object.",
            "Keep mathematical correctness intact.",
            "Use run_python_code when verification is needed.",
        ],
    ).model_dump()
    orchestrator_context = build_orchestrator_context_block(plan_payload, dispatch_payload)
    messages = [
        SystemMessage(content=MODIFIER_EXECUTION_SYSTEM_PROMPT),
        HumanMessage(content=orchestrator_context + "\n\n" + prompt_content)
    ]
    
    try:
        logger.info(f"🔧 Applying modification: '{user_request}'")
        print(f"\n    [MODIFIER] 🔧 Applying modification: '{user_request}'")
        run_config = {"recursion_limit": 40}
        if invoke_config:
            run_config.update(invoke_config)
        
        with tracing_context(parent=get_current_run_tree()):
            response = agent_executor.invoke(
                {"messages": messages},
                config=run_config
            )
        evidence = _format_tool_evidence(response.get("messages", []))
        structured_llm = llm.with_structured_output(GeneratedProblemSchema, include_raw=True)
        modified_problem = invoke_structured_with_slim_trace(
            structured_llm,
            [
                SystemMessage(content=MODIFIER_EXECUTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=orchestrator_context
                    + "\n\n"
                    + prompt_content
                    + "\n\nSupporting tool evidence:\n"
                    + evidence
                    + "\n\nReturn one valid structured modified problem."
                ),
            ],
            trace_name=((invoke_config or {}).get("run_name") or "orchestrator.modifier.llm"),
            tags=list((invoke_config or {}).get("tags", []) or []),
            metadata=dict((invoke_config or {}).get("metadata", {}) or {}),
            summary_inputs={
                "problem_id": problem.get("id", ""),
                "user_request": user_request,
            },
            output_key="modified_problem",
        )

        if "solution" not in modified_problem and modified_problem.get("solution_sketch"):
            modified_problem["solution"] = modified_problem.get("solution_sketch", "")
        modified_problem = GeneratedProblemSchema.model_validate(
            {
                "statement": modified_problem.get("statement", ""),
                "answer": str(modified_problem.get("answer", "")),
                "answer_type": modified_problem.get("answer_type", problem.get("answer_type", "integer")),
                "difficulty": modified_problem.get("difficulty", problem.get("difficulty", "Medium")),
                "difficulty_label": modified_problem.get("difficulty_label", problem.get("difficulty_label", "medium")),
                "solution": modified_problem.get("solution", ""),
                "code": modified_problem.get("code", ""),
                "variation_axis_used": modified_problem.get("variation_axis_used", problem.get("variation_axis_used", "preserve the original variation axis")),
                "difficulty_strategy": modified_problem.get("difficulty_strategy", problem.get("difficulty_strategy", "unspecified")),
                "dispatch_rationale_echo": modified_problem.get("dispatch_rationale_echo", problem.get("dispatch_rationale_echo", "Preserve the original modification intent.")),
                "evidence_summary": modified_problem.get("evidence_summary", problem.get("evidence_summary", "Updated with modifier-provided reasoning and verification evidence.")),
            }
        ).model_dump()

        # Preserve/update metadata
        modified_problem["id"] = f"mod_{problem['id']}_{random.randint(1000, 9999)}"
        modified_problem["parent_ids"] = [problem["id"]]
        modified_problem["type"] = "modified"
        modified_problem["modification_request"] = user_request
        
        logger.info(f"✅ Modification complete: {modified_problem['id']}")
        print(f"    [MODIFIER] ✅ Modification complete: {modified_problem['id']}")
        return modified_problem
        
    except Exception as e:
        logger.error(f"❌ Error modifying problem {problem.get('id')}: {e}")
        print(f"    [MODIFIER] ❌ Error modifying problem {problem.get('id')}: {e}")
        return None
