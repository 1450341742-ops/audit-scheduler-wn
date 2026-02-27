import os
import time
import subprocess
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="审计排班系统", layout="wide")

# ---- Config ----
PORT = int(os.environ.get("PORT", "8000"))
HOST = "0.0.0.0"
BASE_URL = f"http://127.0.0.1:{PORT}"

# ---- Start FastAPI (uvicorn) in background ----
@st.cache_resource(show_spinner=False)
def start_backend():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    cmd = [
        "python", "-m", "uvicorn",
        "app.main:app",
        "--host", HOST,
        "--port", str(PORT),
    ]

    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

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

    return p, "\n".join(out_lines[-100:])

proc, logs = start_backend()

# ===== 关键修改：默认直接进入系统，不再显示封面跳转页 =====

# 顶部简洁工具栏（可保留）
tool1, tool2, tool3 = st.columns([1, 1, 2])

with tool1:
    show_calendar = st.button("切换到日历")
with tool2:
    show_schedule = st.button("切换到排班")
with tool3:
    with st.expander("查看启动日志（排查用）"):
        st.code(logs if logs else "暂无日志")

# 用 session_state 记住当前页面，默认直接进排班
if "current_page" not in st.session_state:
    st.session_state.current_page = "schedule"

if show_calendar:
    st.session_state.current_page = "calendar"

if show_schedule:
    st.session_state.current_page = "schedule"

# 直接渲染页面内容
if st.session_state.current_page == "calendar":
    components.iframe(f"{BASE_URL}/calendar", height=1100, scrolling=True)
else:
    components.iframe(f"{BASE_URL}/schedule", height=1100, scrolling=True)
