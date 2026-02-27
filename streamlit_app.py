import os
import time
import threading
import subprocess
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="审计排班系统", layout="wide")

# ---- Config ----
PORT = int(os.environ.get("PORT", "8000"))
HOST = "0.0.0.0"  # for cloud/container
BASE_URL = f"http://127.0.0.1:{PORT}"

# ---- Start FastAPI (uvicorn) in background ----
@st.cache_resource(show_spinner=False)
def start_backend():
    # Use a subprocess so it works both locally and on Streamlit hosting
    env = os.environ.copy()
    # Ensure PYTHONPATH includes project root so "app.main:app" resolves
    env["PYTHONPATH"] = os.getcwd()
    cmd = [
        "python", "-m", "uvicorn",
        "app.main:app",
        "--host", HOST,
        "--port", str(PORT),
    ]
    # On Streamlit hosting, reload is not needed and can cause issues
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Wait for boot
    deadline = time.time() + 25
    ready = False
    out_lines = []
    while time.time() < deadline and not ready:
        try:
            line = p.stdout.readline().strip()
            if line:
                out_lines.append(line)
                if ("Application startup complete" in line) or ("Uvicorn running on" in line):
                    ready = True
        except Exception:
            pass
        time.sleep(0.1)
    return p, "\n".join(out_lines[-50:])

proc, logs = start_backend()

st.title("审计排班系统（网页版）")
st.caption("此页面用于在 Streamlit 上托管原 FastAPI 系统：同事直接打开网页即可使用。")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.link_button("打开排班页面 /schedule", f"{BASE_URL}/schedule")
with col2:
    st.link_button("打开日历页面 /calendar", f"{BASE_URL}/calendar")
with col3:
    st.download_button("下载启动日志（排查用）", logs.encode("utf-8"), file_name="backend_startup_log.txt")

tabs = st.tabs(["内嵌日历（推荐）", "内嵌排班", "说明"])
with tabs[0]:
    # Embed calendar page
    components.iframe(f"{BASE_URL}/calendar", height=900, scrolling=True)
with tabs[1]:
    components.iframe(f"{BASE_URL}/schedule", height=900, scrolling=True)
with tabs[2]:
    st.markdown(
        '''
### 如何部署到 Streamlit（同事直接打开网页）

**方式 1：Streamlit Community Cloud（最省事）**
1. 把本项目上传到 GitHub（私有/公有仓库均可）
2. Streamlit Cloud 新建 App：
   - Repository：选择你的仓库
   - Branch：main
   - Main file path：`streamlit_app.py`
3. 部署完成后，会生成一个 URL，发给同事即可。

**方式 2：公司内网服务器部署（更适合内部系统）**
- 服务器装好 Python 后运行：
  - `pip install -r requirements.txt`
  - `streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0`
- 然后同事访问：`http://<服务器IP>:8501`

> 注意：本方案会在 Streamlit 容器内启动一个 FastAPI 服务（端口 8000），Streamlit 页面用 iframe 内嵌原系统页面。
'''
    )
