from modelscope.hub.api import HubApi

# api = HubApi()
# api.upload_folder(repo_id="alephpi98/FormulaNet", folder_path="./model")

api = HubApi()
# 列出仓库文件（包括隐藏文件）
files = api.get_model_files(model_id="alephpi98/FormulaNet")
print(files)