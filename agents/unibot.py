from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
import sys
from pathlib import Path
import time
import uuid
import asyncio
import os
from datetime import datetime

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from agents.tools import get_rag_tools
from retrieval.rag_system import RAGSystem


load_dotenv(override=True)


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


class EvaluatorOutput(BaseModel):
    feedback: str = Field(
        description="Feedback on the assistant's response"
    )
    success_criteria_met: bool = Field(
        description="Whether the success criteria have been met"
    )
    user_input_needed: bool = Field(
        description=(
            "True if more input is needed from the user, "
            "or clarifications, or the assistant is stuck"
        )
    )


class UniBot:
    def __init__(self, college_website_url: Optional[str] = None):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.llm_with_tools = None
        self.graph = None
        self.unibot_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None
        self.rag_system = RAGSystem()
        self.college_website_url = college_website_url

    async def setup(self):
        # Verify API key is set
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables."
            )

        # Initialize tools.
        # Trimmed to just the RAG tool set: this is a college-FAQ bot, not
        # a general agent, so it doesn't need generic web browsing
        # (Playwright toolkit - was also launching a full Chromium browser
        # on every startup for no benefit), a Python REPL, file read/write,
        # or push notifications. Keeping the tool surface narrow also
        # matters for a live demo/interview - "why does a college chatbot
        # have code execution and file access" is not a question you want
        # to be answering on the spot.
        self.tools = get_rag_tools(self.rag_system)

        # Disable live website scraping during normal chatbot queries
        self.tools = [
            tool
            for tool in self.tools
            if getattr(tool, "name", "") != "scrape_college_website"
        ]

        # Worker model
        # Using Groq instead of Gemini: meaningfully more generous
        # free-tier rate limits and much faster inference (Groq's custom
        # LPU hardware) - directly addresses the stalls and 429 rate-limit
        # errors this project was hitting on Gemini's free tier.
        # Using gpt-oss-20b (not the larger 120b) for both Worker and
        # Evaluator: 20b's smaller context need keeps requests comfortably
        # under Groq's free-tier 8,000 TPM limit even with multiple
        # retrieved chunks in the prompt - 120b was hitting 413 "request
        # too large" errors under the same load. Response times dropped
        # from ~45s to ~1.5s as a result.
        worker_model = "openai/gpt-oss-20b"

        self.worker_llm = ChatGroq(
            model=worker_model,
            temperature=0.2
        )

        self.worker_llm_with_tools = self.worker_llm.bind_tools(
            self.tools
        )

        # Evaluator model
        evaluator_llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.1
        )

        self.evaluator_llm_with_output = (
            evaluator_llm.with_structured_output(
                EvaluatorOutput
            )
        )

        await self.build_graph()

    def worker(self, state: State) -> Dict[str, Any]:
        system_message = f"""You are UniBot, a college query assistant for answering questions about the college.

You have access to a knowledge base containing information scraped from the college website.

CRITICAL RULE:
For any question about the college, courses, admissions, faculty, facilities, policies, events, or other college-related information, you MUST call the query_college_knowledge_base tool before providing an answer.

WORKFLOW:
1. Identify the user's college-related question.
2. Search the college knowledge base using query_college_knowledge_base ONCE.
3. Once you receive search results from query_college_knowledge_base, immediately synthesize and provide the final answer with source citations. Do NOT search again.
4. If the retrieved information does not contain the answer, state that clearly.
5. Do not invent unsupported information.
6. Include the source links provided by the knowledge base.

SCRAPING:
Do NOT scrape the college website during normal questions.
Only use the scrape_college_website tool if the user explicitly asks you to:
- scrape the website
- update the knowledge base
- refresh the knowledge base

If the knowledge base does not contain enough information for the user's question, do not invent an answer and do not suggest scraping unless the user explicitly requested it.

SOURCE CITATION:
When you provide information from the knowledge base, ALWAYS include the source links from the tool output.

The knowledge base tool provides a "Sources:" section with URLs formatted as markdown links like:
[URL](URL)

Copy the source links from the tool output accurately.
Do not rename, abbreviate, or invent source URLs.

Example:
Sources:
1. [https://bmsit.ac.in/admissions](https://bmsit.ac.in/admissions)
2. [https://bmsit.ac.in/public/assets/pdf/fee/structure.pdf](https://bmsit.ac.in/public/assets/pdf/fee/structure.pdf)

Do not use generic names such as "BMSIT Administration" when the exact source URL is available.

DEPARTMENT HEADS & LEADERSHIP:
When answering questions about Heads of Department (HODs), Deans, Principal, or college leadership:
- Always use the official Academic Council Members list (2024-25 to 2026-27) or official department directory provided in the knowledge base.
- Match each person carefully to their specific department (e.g. Dr. Satish Kumar T is HOD of Computer Science & Engineering (CSE); Dr. Surekha K B is HOD of Information Science & Engineering (ISE)).
- Do NOT cite or use old student newsletters or event issues from past years for current HOD positions.

You keep working on the task until either:
- a clarification is genuinely required from the user, or
- the success criteria is met.

The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This is the success criteria:
{state['success_criteria']}

RESPONSE STYLE:
- Answer the user's question directly and clearly.
- Use information supported by the knowledge base.
- Do not expose internal reasoning or thought processes.
- Do not mention evaluator feedback, previous rejection, or internal processing.
- Do not ask unnecessary follow-up questions.
- If clarification is genuinely necessary, clearly state the question.
"""

        # Add evaluator feedback when refinement is required
        if state.get("feedback_on_work"):
            system_message += f"""
Previously, the response did not satisfy the success criteria.

Evaluator feedback:
{state['feedback_on_work']}

Use this feedback to improve the answer.
Do not mention the feedback, rejection, or internal reasoning in your response.
Provide the corrected answer directly.
"""

        # Work on a copy so the original state is not modified in place
        messages = list(state["messages"])

        # Add/update system message
        found_system_message = False

        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        if not found_system_message:
            messages = [
                SystemMessage(content=system_message)
            ] + messages

        # -------------------------------------------------------------
        # IMPORTANT:
        # When the Evaluator sends the task back to the Worker, the
        # previous message may be an AIMessage. Gemini requires the
        # final conversational turn to be a user/function message.
        #
        # Add evaluator feedback as a HumanMessage so the retry has
        # a valid final turn.
        # -------------------------------------------------------------
        if state.get("feedback_on_work"):
            messages.append(
                HumanMessage(
                    content=(
                        "Evaluator feedback for the previous answer:\n\n"
                        f"{state['feedback_on_work']}\n\n"
                        "Please revise the answer using this feedback. "
                        "Return only the corrected answer."
                    )
                )
            )

        # Check if tools were already executed in this conversation turn
        last_human_idx = -1
        for idx, m in enumerate(messages):
            if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
                last_human_idx = idx

        already_used_tools = any(
            getattr(m, "type", "") == "tool" or type(m).__name__ == "ToolMessage"
            for m in messages[last_human_idx + 1:]
        ) if last_human_idx >= 0 else False

        # If tools were already run for this turn, invoke raw worker_llm so it synthesizes the answer
        llm_to_invoke = self.worker_llm if already_used_tools else self.worker_llm_with_tools

        # Invoke Worker LLM
        start_time = time.perf_counter()

        response = llm_to_invoke.invoke(
            messages
        )

        elapsed = time.perf_counter() - start_time

        tool_calls = getattr(response, "tool_calls", [])

        print(
            f"[TIMING] Worker LLM: {elapsed:.2f}s | "
            f"tool_calls={len(tool_calls)}"
        )

        if tool_calls:
            print(
                "[TIMING] Worker requested tools: "
                + ", ".join(
                    call.get("name", "unknown")
                    for call in tool_calls
                )
            )

        return {
            "messages": [response],
        }

    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]

        if (
            hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            return "tools"

        return "evaluator"

    def format_conversation(
        self,
        messages: List[Any]
    ) -> str:
        conversation = "Conversation history:\n\n"

        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += (
                    f"User: {message.content}\n"
                )

            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += (
                    f"Assistant: {text}\n"
                )

        return conversation

    def evaluator(self, state: State) -> State:
        if not state["messages"]:
            return {
                "feedback_on_work": "No messages found",
                "success_criteria_met": False,
                "user_input_needed": True
            }

        last_message = state["messages"][-1]

        # Handle message objects and dictionaries
        if hasattr(last_message, "content"):
            last_response = last_message.content or ""

        elif isinstance(last_message, dict):
            last_response = last_message.get(
                "content",
                ""
            )

        else:
            last_response = (
                str(last_message)
                if last_message
                else ""
            )

        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.

Assess the Assistant's last response based on the given success criteria.

You must provide:
1. Feedback on the response.
2. Whether the success criteria have been met.
3. Whether additional user input is required.

Be objective and evaluate whether the answer actually satisfies the user's request.
"""

        user_message = f"""You are evaluating a conversation between the User and Assistant.

The conversation is:

{self.format_conversation(state['messages'])}

The success criteria for this task is:

{state['success_criteria']}

The final response from the Assistant is:

{last_response}

Determine whether the response satisfies the success criteria.

Also determine whether more user input is required because:
- clarification is genuinely needed,
- the Assistant asked a necessary question,
- or the Assistant is unable to complete the task.

The Assistant has access to tools, including file-writing tools.
If the Assistant states that it completed an action, give it the benefit of the doubt unless there is a clear reason to reject the response.

Reject the response only when meaningful improvement is required.
"""

        # Previous feedback is retained for later refinement
        # but is not added as an assistant message.
        if state.get("feedback_on_work"):
            user_message += (
                f"\nPrevious evaluator feedback:\n"
                f"{state['feedback_on_work']}\n"
            )

        evaluator_messages = [
            SystemMessage(
                content=system_message
            ),
            HumanMessage(
                content=user_message
            )
        ]

        # Invoke evaluator
        start_time = time.perf_counter()

        eval_result = (
            self.evaluator_llm_with_output.invoke(
                evaluator_messages
            )
        )

        print(
            f"[EVALUATOR] success="
            f"{eval_result.success_criteria_met} "
            f"| user_input_needed="
            f"{eval_result.user_input_needed}"
        )

        safe_feedback = str(eval_result.feedback).encode('ascii', 'backslashreplace').decode('ascii') if eval_result.feedback else ""
        print(
            f"[EVALUATOR] feedback="
            f"{safe_feedback}"
        )

        elapsed = time.perf_counter() - start_time

        print(
            f"[TIMING] Evaluator LLM: "
            f"{elapsed:.2f}s | "
            f"success="
            f"{eval_result.success_criteria_met} | "
            f"user_input_needed="
            f"{eval_result.user_input_needed}"
        )

        # IMPORTANT:
        # Do NOT append evaluator feedback as an AIMessage.
        #
        # Keeping it only in state.feedback_on_work prevents the
        # next Gemini Worker request from ending with an assistant
        # message, which previously caused the prefilling error.
        return {
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": (
                eval_result.success_criteria_met
            ),
            "user_input_needed": (
                eval_result.user_input_needed
            )
        }

    def route_based_on_evaluation(
        self,
        state: State
    ) -> str:

        if (
            state["success_criteria_met"]
            or state["user_input_needed"]
        ):
            return "END"

        return "worker"

    async def build_graph(self):
        # Set up graph builder
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node(
            "worker",
            self.worker
        )

        graph_builder.add_node(
            "tools",
            ToolNode(tools=self.tools)
        )

        graph_builder.add_node(
            "evaluator",
            self.evaluator
        )

        # Worker routing
        graph_builder.add_conditional_edges(
            "worker",
            self.worker_router,
            {
                "tools": "tools",
                "evaluator": "evaluator"
            }
        )

        # Tool results go back to Worker
        graph_builder.add_edge(
            "tools",
            "worker"
        )

        # Evaluator routing
        graph_builder.add_conditional_edges(
            "evaluator",
            self.route_based_on_evaluation,
            {
                "worker": "worker",
                "END": END
            }
        )

        graph_builder.add_edge(
            START,
            "worker"
        )

        # Compile graph
        self.graph = graph_builder.compile(
            checkpointer=self.memory
        )

    async def run_superstep(
        self,
        message,
        success_criteria,
        history,
        thread_id: Optional[str] = None
    ):
        config = {
            "configurable": {
                # Use the caller-supplied session id so each browser tab /
                # user gets its own LangGraph thread. Falls back to the
                # instance-wide id only when no session id is provided
                # (e.g. direct/script usage outside the API server).
                "thread_id": thread_id or self.unibot_id
            }
        }

        # Convert history to LangChain messages
        messages = []

        if history:
            for msg in history:
                role = msg.get("role", "")
                content = msg.get(
                    "content",
                    ""
                )

                if role == "user":
                    messages.append(
                        HumanMessage(
                            content=content
                        )
                    )

                elif role == "assistant":
                    messages.append(
                        AIMessage(
                            content=content
                        )
                    )

        # Add current user message
        if isinstance(message, str):
            messages.append(
                HumanMessage(
                    content=message
                )
            )

        elif isinstance(message, list):
            messages.extend(message)

        else:
            messages.append(
                HumanMessage(
                    content=str(message)
                )
            )

        state = {
            "messages": messages,
            "success_criteria": (
                success_criteria
                or "The answer should be clear and accurate"
            ),
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False
        }

        # Execute graph
        result = await self.graph.ainvoke(
            state,
            config=config
        )

        # Extract user message content for history
        user_content = (
            message
            if isinstance(message, str)
            else (
                messages[0].content
                if messages
                else str(message)
            )
        )

        user = {
            "role": "user",
            "content": user_content
        }

        # -------------------------------------------------------------
        # Get the latest actual assistant answer.
        #
        # Do not assume evaluator feedback is a message anymore.
        # Also ignore AI messages that only contain tool calls.
        # -------------------------------------------------------------
        assistant_message = None

        for msg in reversed(result["messages"]):
            if not isinstance(msg, AIMessage):
                continue

            if not msg.content:
                continue

            tool_calls = getattr(
                msg,
                "tool_calls",
                []
            )

            if tool_calls:
                continue

            assistant_message = msg
            break

        if assistant_message:
            assistant_content = (
                assistant_message.content
            )

            if isinstance(
                assistant_content,
                list
            ):
                text_parts = []

                for item in assistant_content:
                    if (
                        isinstance(item, dict)
                        and "text" in item
                    ):
                        text_parts.append(
                            item["text"]
                        )

                    elif isinstance(item, str):
                        text_parts.append(item)

                assistant_reply = (
                    "\n".join(text_parts)
                    if text_parts
                    else str(assistant_content)
                )

            elif isinstance(
                assistant_content,
                dict
            ):
                assistant_reply = (
                    assistant_content.get(
                        "text",
                        str(assistant_content)
                    )
                )

            else:
                assistant_reply = (
                    str(assistant_content)
                    if assistant_content
                    else ""
                )

        else:
            assistant_reply = ""

        reply = {
            "role": "assistant",
            "content": assistant_reply
        }

        # Return only user and assistant messages
        return history + [user, reply]

    def cleanup(self):
        """Cleanup browser and playwright resources"""

        if self.browser or self.playwright:
            try:
                loop = asyncio.get_running_loop()

                # If an async loop is running,
                # schedule cleanup
                if self.browser:
                    loop.create_task(
                        self.browser.close()
                    )

                if self.playwright:
                    loop.create_task(
                        self.playwright.stop()
                    )

            except RuntimeError:

                # If no loop is running, create one
                async def cleanup_async():
                    if self.browser:
                        await self.browser.close()

                    if self.playwright:
                        await self.playwright.stop()

                try:
                    asyncio.run(
                        cleanup_async()
                    )

                except Exception as e:
                    print(
                        f"Warning: Error during cleanup: {e}"
                    )