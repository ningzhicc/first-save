from transformers import AutoTokenizer, AutoModel
model_dir = r"../downloaded_plms/qwen/base"
tok = AutoTokenizer.from_pretrained(model_dir)
mdl = AutoModel.from_pretrained(model_dir)
print(type(tok))
print(type(mdl))