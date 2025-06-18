from dotenv import load_dotenv
from openai import OpenAI
from openai import AzureOpenAI
import json
import os
import datetime
import inspect

import python_code_notebook

load_dotenv()
deployment_name = os.getenv("DEPLOYMENT_NAME")

class PythonCodeInterpreter():
    def __init__(self, deployment_name: str):

        # self.client = OpenAI()
        self.client = AzureOpenAI()

        self.system_message = True
        self.deployment_name = deployment_name
        self.current_messages_index = 1
        self.ipynb_result_dir = "results"
        self.ipynb_prefix = os.path.join(os.path.dirname(__file__), self.ipynb_result_dir, "running_")
        self.ipynb_file = ""
        self.result_file = ""
        self.messages = []
        self.messages_system = [{
            "role": "system",
            "content": (
                f"You are interacting with {deployment_name}, a large language model trained by OpenAI. "
                "The model is based on ReAct technology and uses Python for data analysis and visualization.\n"
                "When a message containing Python code is sent to Python, it is executed in the state-preserving "
                "Jupyter notebook environment. Python returns the results of the execution. "
                "'/mnt/data' drive can be used to store and persist user files.\n"
                "Python is used to analyze, visualize, and predict the data. If you provide a data set, "
                "we will analyze it and create appropriate graphs for visualization. Additionally, "
                "we can extract trends from the data and provide future projections.\n"
                "We can also provide information on a wide range of scientific topics, "
                "including natural language processing (NLP), machine learning, mathematics, physics, chemistry, "
                "and biology. Let us know what questions you have, what your research needs are, or what problems "
                "you need solved.\n"
                "When a user hands you a file, first understand the type of data you are dealing with, its structure "
                "and characteristics, and tell me its contents. Use clear text and sometimes diagrams.\n"
            )
        }]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    # "description": "When there is information I don't know, I run some Python code to get the results.",
                    "description": "If some information is unknown, run Python code to get the data from outside, do calculations, etc., to get the results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "python_code": {
                                "type": "string",
                                "description": (
                                    "The python program code. The python is used to execute the python code created "
                                    "by you for all information. python code must not perform any directory or file "
                                    "operations outside of the current directory."
                                ),
                            }
                        },
                        "required": ["python_code"],
                    },
                }
            },
        ]
        self.available_functions = {
            "run_python": self.run_python_code_in_notebook,
        } 
    
    def ask_continue(self):
        # result = False
        # user_input = input("Do you want to continue running this program?(yes/no): ").strip().lower()
        # if user_input == "yes":
        #     result = True
        result = True
        return result

    # helper method used to check if the correct arguments are provided to a function
    def check_args(self, function, args):
        sig = inspect.signature(function)
        params = sig.parameters

        # Check if there are extra arguments
        for name in args:
            if name not in params:
                return False
        # Check if the required arguments are provided 
        for name, param in params.items():
            if param.default is param.empty and name not in args:
                return False
        return True

    # Run Python code in the notebook
    def run_python_code_in_notebook(self, code: str, messages):
        if code.startswith('{"python_code":'):
            code = json.loads(code)["python_code"]

        # Pause to review the program
        print(f"----------\n{code}\n----------")
        if not self.ask_continue():
            quit()

        # Run
        results, self.ipynb_file = python_code_notebook.run_all(
            code,
            messages = messages,
            prepared_notebook=self.ipynb_file,
            result_ipynb_prefix=self.ipynb_prefix,
            remove_result_ipynb=False
            )
        result = results[-1]
        print(result)
        result_strs =  [x['text/plain'] for x in result if x.get('text/plain')]
        result_str = "\n".join(result_strs)
        return result_str

    def write_messages_in_notebook(self, messages):
        result, self.ipynb_file = python_code_notebook.run_all(
            "",
            messages = messages,
            prepared_notebook=self.ipynb_file,
            result_ipynb_prefix=self.ipynb_prefix,
            remove_result_ipynb=False
            )
        return

    def run_conversation(self, message):
        result_name = f'result_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        if self.system_message is True:
            self.messages.extend(self.messages_system)
        self.messages.append({"role": "user", "content": message})
        max_loops = 200
        tool_choice_flag = False
        finish_flag = False
        for i in range(max_loops):
            print(f"Loop {i+1}/{max_loops}")
            completion = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=self.messages,
                tools=self.tools,
                # parallel_tool_calls=False,
                # tool_choice="auto" if tool_choice_flag else "none", 
                # temperature=0,
                # seed=100,

            )
            response_message = completion.choices[0].message
            response_reason = completion.choices[0].finish_reason
            response_role = response_message.role

            if response_reason == 'tool_calls':
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_arguments = tool_call.function.arguments

                    if function_name not in self.available_functions:
                        return f"Function {function_name} does not exist."

                    function_to_call = self.available_functions[function_name]
                    if response_message.content is None:
                        content_messages = self.messages[self.current_messages_index:]
                    else:
                        content_messages = [response_message]
                    function_response = function_to_call(function_arguments, content_messages)

                    # special treatment. For some reason, an error occurs when inserting a figure strings
                    if function_response.startswith('<Figure size'):
                        function_response = "Omitted due to the large size of the image."

                    self.messages.append(response_message)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": function_response,
                        }
                    )
            elif response_reason == 'stop':
                self.write_messages_in_notebook([response_message])
                self.messages.append(
                    {
                        "role": response_role,
                        "content": response_message.content,
                    }
                )
                print(f"Response: {response_message.content}")
                user_input = input("Please write the message you want to send. If you want to finish the conversation, type 'exit': ").strip().lower()
                if user_input == 'exit':
                    finish_flag = True
                else:
                    self.write_messages_in_notebook([{"role": "user", "content": user_input}])
                    self.messages.append({"role": "user", "content": user_input})
            elif response_reason == 'length':
                print("Response length exceeded the limit. Please try again with a shorter message.")
            else:
                print(f"Unexpected response reason: {response_reason}")
            self.current_messages_index = len(self.messages)
            if finish_flag:
                break
        
        # write results
        current_ipynb_file = self.ipynb_file
        self.ipynb_file = os.path.join(os.path.dirname(__file__), self.ipynb_result_dir, f'{result_name}.ipynb')
        os.rename(current_ipynb_file, self.ipynb_file)
        self.result_file = os.path.join(os.path.dirname(__file__), self.ipynb_result_dir, f'{result_name}.json')
        with open(self.result_file, 'w') as f:
            print(self.messages, file=f)
        return self.messages

if __name__ == '__main__':
    message = (
        "半導体イオン注入装置におけるアーク放電の発生回数が増加する原因を以下の DATA_DIRECTORY の各FILE_TYPEの全てのデータから調べて下さい。\n"
        "ドメイン知識も活用してください。アーク放電の発生回数が増加する原因を知りたいです。\n"
        "- DATA_DIRECTORY には、 ヘッダ名が異なるデータのグループがディレクトリ毎に各格納されています。\n"
        "- FILE_TYPE毎に異なるデータが入っていますが、timestampは全てのFILE_TYPEに含まれていますので、indexとして使ってください。\n"
        "- アーク放電の発生回数の累計を示すデータのヘッダ名は、 *ArcingCount* ですが、全てのFILE_TYPEには含まれていません。\n"
        "- 各ファイルは全て同一の時間インデックス（timestamp）を持ち、計測データの種類ごとにカラムを分割して複数ファイルに分けて保存しているという構造です。\n"
        "- 目的変数は 複数の*ArcingCount* ヘッダになります。また、類型になるため、前ステップとの差を求めるとアーク放電の発生時間を知ることができます。\n"
        "- アーク放電の発生回数が増加する説明変数から 複数の*ArcingCount* ヘッダ名を除いてください。\n"
        "- 数値に変換できない場合はカテゴリ変数として enum 数値に変換してください。\n"
        "- function_callingのpythonにコメントを記載てください。\n"
        "\n"
        "# DATA_DIRECTORY\n"
        "/mnt/data/em02_data/csvlog_202209_rs\n"
        "\n"
        "# FILE_TYPE\n"
        "sv_history_f8_1000_01_*.csv\n"
        "sv_history_f8_60000_01_*.csv\n"
        "sv_history_f8_0050_05_*.csv\n"
        "sv_history_f8_1000_03_*.csv\n"
        "sv_history_bo_0100_03_*.csv\n"
        "sv_history_f8_0050_01_*.csv\n"
        "sv_history_i4_0050_01_*.csv\n"
        "sv_history_bo_0050_01_*.csv\n"
        "sv_history_f8_0100_02_*.csv\n"
        "sv_history_bo_0050_03_*.csv\n"
        "sv_history_bo_0100_01_*.csv\n"
        "sv_history_bo_0100_02_*.csv\n"
        "sv_history_bo_1000_01_*.csv\n"
        "sv_history_f8_0050_04_*.csv\n"
        "sv_history_bo_0050_04_*.csv\n"
        "sv_history_a_0100_01_*.csv\n"
        "sv_history_bo_0050_02_*.csv\n"
        "sv_history_a_1000_01_*.csv\n"
        "sv_history_f8_1000_02_*.csv\n"
        "sv_history_i4_1000_01_*.csv\n"
        "sv_history_f8_0050_02_*.csv\n"
        "sv_history_i4_0100_01_*.csv\n"
        "sv_history_bo_1000_02_*.csv\n"
        "sv_history_f8_0100_01_*.csv\n"
        "sv_history_f8_0050_03_*.csv\n"
        "\n"
        "# FILE_FORMAT\n"
        "FILE_TYPE_YYYY-MM-DD.csv\n"
        "## 例\n"
        "sv_history_f8_0050_04_2023_11_06.csv\n"
        "\n"
        "# アーク放電の発生回数ヘッダ名\n"
        "## 例\n"
        "PM.BeamSystem.EnergyController.ExtractionForHighEnergy.ArcingCount_mean\n"
        "PM.BeamSystem.EnergyController.Decel.ArcingCount_mean\n"
        "PM.BeamSystem.BeamCorrector.EBendOuter.ArcingCount_mean\n"
        "PM.BeamSystem.BeamCorrector.DecelFocus.ArcingCount_mean\n"
        "PM.BeamSystem.BeamCorrector.EBendMid.ArcingCount_mean\n"
        "PM.BeamSystem.BeamCorrector.EBendInner.ArcingCount_mean\n"
        "\n"
        )


    print(f"Message: {message}")
    # Initialize the Python Code Interpreter
    pci = PythonCodeInterpreter(deployment_name)
    pci.system_message = True
    assistant_response = pci.run_conversation(message)
    print(assistant_response)
