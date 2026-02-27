# 部署到 Streamlit（同事直接打开网页使用）

本项目已增加 `streamlit_app.py`，用于在 Streamlit 上托管系统。

## 1. 本地测试
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
打开终端提示的地址（通常是 http://localhost:8501 ）。

## 2. Streamlit Cloud 部署
1) 把项目推到 GitHub  
2) Streamlit Cloud -> New app  
- Main file path: `streamlit_app.py`  
3) 部署后获得网页链接，发给同事。

## 3. 内网服务器部署
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```
同事访问：`http://<服务器IP>:8501`

> 说明：Streamlit 页面会在后台启动 FastAPI（uvicorn）并用 iframe 内嵌 `/calendar` 和 `/schedule`。
