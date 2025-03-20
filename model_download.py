#模型下载
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='.')