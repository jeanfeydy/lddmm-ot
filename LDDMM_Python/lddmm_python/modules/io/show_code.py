import inspect
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter, Terminal256Formatter

def show_code(func):
	if type(func) is str :
		code = func
	else :
		code = inspect.getsourcelines(func)[0]
		code = ''.join(code)
	print(highlight(code, PythonLexer(), Terminal256Formatter()))

