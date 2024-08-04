from langchain.tools import BaseTool

# 天气查询工具，无论查询什么都返回Sunny
class WeatherTool(BaseTool):
    name = "Weather"
    description = "useful for when you want to know about weather"

    def _run(self, query: str) -> str:
        return 'Sunny^_^'

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError('BingSearchRun does not support async')
    
# 计算工具，返回3
class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math."

    def _run(self, query: str) -> str:
        return "3"
    async def _arun(self, query: str) -> str:
        return NotImplementedError("BindSearchRun does not support async")
    

# 针对工具的简单调用：使用openai的temperature=0
from langChain.agents import initialize_agent
from langChain.llms import OpenAI
from CustomTools import WeatherTool
from CustomTools import CustomCalculatorTool

llm = OpenAI(temperature=0)

tools = [WeatherTool(), CustomCalculatorTool()]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("Query the weather of this week, and how old will i be in ten years?the year i am 28")

# langchain 是一个基于LLM的编程框架，旨在帮助开发人员使用llm构建端到端的应用程序。
# 它提供了一套工具、组件和接口，可以简化创建由llm和聊天模型提供支持的应用程序的过程，langChain由几大组件组成，包括models，prompts，chains，memory和agent

# agent是一个代理，接受用户的输入，采取相应的行动然后返回行动的结果，agent可以看作是一个自带路由消费的chains的代理，基于MRKL和ReAct的基本原理，agent可以使用工具和自然语言处理问题。
# agent的作用是代表用户或其他系统完成任务，如数据收集、数据处理、数据决策等

# ReAct是一个结合了推理和行动的语言模型，虽然LLM在语言理解和交互决策制定方面展现出了令人印象深刻的能力，但它们的推理（例如链式思考提示）和行动（例如行动计划生成）的能力主要被视为两个独立的主题
# 借助于ReAct，不仅能执行任务，还会告诉你如何思考和决策的。

