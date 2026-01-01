from google.adk.tools import function_tool

def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

# Create tools that the Agent can use
add_tool = function_tool.FunctionTool(add)
multiply_tool = function_tool.FunctionTool(multiply)
