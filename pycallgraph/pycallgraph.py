"""Module for call graph visualization using PyCallGraph.

The idea was to create a decorator to visualize the code structure.

Requirements:
-install pycallgraph2
-install graphviz according to https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224
and add graphviz path ("C:/Program Files/Graphviz/bin") to System Path in Environment Variables.

You can specify the maximum depth your visualization, the memory record
and the file name of your output .png file.

"""

from functools import wraps
import psutil

from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from pycallgraph2 import memory_profiler as pycallgraph_Profiler
from pycallgraph2 import Config as pycallgraph_Config

def graph_call_path_factory(max_depth, memory_flag, file_name):
    def graph_call_path(my_function):
        """ Utility function that works as decorator for graphing single function call path. """
        @wraps(my_function)
        def function_wrapper_for_graph_function_call_path(*args, **kwargs):
            """ Inner function for the time measuring utility decorator. """
            if memory_flag:
                pycallgraph_config = pycallgraph_Config(max_depth=max_depth, memory=True)
                pycallgraph_Profiler._get_memory = graph_call_memory_monkey_patch
            else:
                pycallgraph_config = pycallgraph_Config(max_depth=max_depth)
            graphviz = GraphvizOutput(output_file=file_name+'.png')
            with PyCallGraph(output=graphviz, config=pycallgraph_config):
                result = my_function(*args, **kwargs)
            return result
        return function_wrapper_for_graph_function_call_path
    return graph_call_path


def graph_call_memory_monkey_patch(pid):
    """ Monkey patch function to correct memory info method in pycallgraph. """
    process = psutil.Process(pid)
    try:
        mem = float(process.memory_info()[0]) / (1024 ** 2)
    except psutil.AccessDenied:
        mem = -1
    return mem