"""
screen6_carla_launcher.py
Streamlit page for launching CARLA and ScenarioRunner automatically.
"""
import subprocess
import threading
import time
import os
import signal
import glob
import streamlit as st

# Module-level log buffers — safe to write from background threads
_log_buffers = {
    "carla_logs": [],
    "scenario_logs": [],
}

def show():
    CARLA_ROOT = "/home/trishan"
    SCENARIO_RUNNER_ROOT = "/home/trishan/Desktop/shamanth_mtech/scenario_runner"
    eggs = glob.glob(os.path.join(CARLA_ROOT, "PythonAPI/carla/dist/*py3*.egg"))
    CARLA_EGG = eggs[0] if eggs else None
    PYTHONPATH_EXTRA = ":".join(filter(None, [
        CARLA_EGG,
        os.path.join(CARLA_ROOT, "PythonAPI/carla"),
        os.path.join(CARLA_ROOT, "PythonAPI"),
    ]))

    #CARLA_CMD = [
    ##   os.path.join(CARLA_ROOT, "CarlaUE4.sh"),
    #"-RenderOffScreen",
    #   "-quality-level=Low",
    #   "-no-rendering",
    #   "-benchmark",
    #  "-fps=15"
    #]

    CARLA_CMD = [
        os.path.join(CARLA_ROOT, "CarlaUE4.sh"),
        "-quality-level=Low",
        "-carla-server",
        "-fps=20",
    ]

    defaults = {
        "carla_process": None,
        "scenario_process": None,
        "carla_logs": [],
        "scenario_logs": [],
        "carla_status": "stopped",
        "scenario_status": "stopped",
        "carla_launched": False,
        "scenario_launched": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def stream_logs(process, log_key, status_key, ready_marker=None):
        for line in iter(process.stdout.readline, b""):
            decoded = line.decode("utf-8", errors="replace").rstrip()
            _log_buffers[log_key].append(decoded)
            if ready_marker and ready_marker in decoded:
                st.session_state[status_key] = "running"
        rc = process.wait()
        if st.session_state.get(status_key) not in ("stopped",):
            st.session_state[status_key] = "finished" if rc == 0 else "crashed"

    def launch_carla():
        st.session_state["carla_status"] = "starting"
        st.session_state["carla_launched"] = True
        _log_buffers["carla_logs"] = []
        env = os.environ.copy()
        proc = subprocess.Popen(
            CARLA_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid
        )
        st.session_state["carla_process"] = proc
        threading.Thread(
            target=stream_logs,
            args=(proc, "carla_logs", "carla_status", "Server listening"),
            daemon=True
        ).start()

    def launch_scenario(xosc_path):
        st.session_state["scenario_status"] = "starting"
        st.session_state["scenario_launched"] = True
        _log_buffers["scenario_logs"] = []
        cmd = [
            "bash", "-c",
            f'source /home/trishan/Desktop/shamanth_mtech/venv37/bin/activate && '
            f'export PYTHONPATH=/home/trishan/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:/home/trishan/PythonAPI/carla:/home/trishan/PythonAPI && '
            f'cd /home/trishan/Desktop/shamanth_mtech/scenario_runner && '
            f'python3.7 scenario_runner.py --openscenario "{xosc_path}" --host 127.0.0.1 --port 2000 --timeout 60 --output --repetitions 3'
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        st.session_state["scenario_process"] = proc
        threading.Thread(
            target=stream_logs,
            args=(proc, "scenario_logs", "scenario_status"),
            daemon=True
        ).start()

    def stop_all():
        for key in ["carla_process", "scenario_process"]:
            proc = st.session_state.get(key)
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
        st.session_state["carla_status"] = "stopped"
        st.session_state["scenario_status"] = "stopped"
        st.session_state["carla_process"] = None
        st.session_state["scenario_process"] = None
        st.session_state["carla_launched"] = False
        st.session_state["scenario_launched"] = False

    STATUS_ICONS = {
        "stopped": "⚪", "starting": "🟡", "running": "🟢",
        "finished": "✅", "crashed": "🔴", "error": "🔴",
    }

    def status_badge(label, status):
        icon = STATUS_ICONS.get(status, "⚪")
        st.markdown(f"**{label}:** {icon} `{status.upper()}`")

    st.title("🚗 Launch in CARLA")
    st.markdown("Launch CARLA first, then run your generated scenario in ScenarioRunner.")
    st.divider()

    st.subheader("📁 Scenario File")
    xosc_path = st.text_input(
        "Path to generated .xosc file",
        value=st.session_state.get(
            "generated_xosc_path",
            os.path.join(SCENARIO_RUNNER_ROOT, "srunner/examples/Car-to-Car Rear Stationary.xosc")
        ),
        help="Full path to the OpenSCENARIO file generated by your RAG system"
    )

    st.divider()
    st.subheader("📊 Status")
    col1, col2 = st.columns(2)
    with col1:
        status_badge("CARLA Simulator", st.session_state["carla_status"])
    with col2:
        status_badge("Scenario Runner", st.session_state["scenario_status"])

    st.divider()
    st.subheader("🎮 Controls")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "🚀 Launch CARLA",
            type="primary",
            use_container_width=True,
            disabled=st.session_state["carla_launched"]
        ):
            launch_carla()
            st.success("✅ CARLA is starting...")
            st.rerun()

    with col2:
        carla_running = st.session_state["carla_status"] in ("running", "starting")
        if st.button(
            "🏁 Run Scenario",
            type="primary",
            use_container_width=True,
            disabled=st.session_state["scenario_launched"]
        ):
            if not os.path.exists(xosc_path):
                st.error(f"❌ XOSC file not found: {xosc_path}")
            elif st.session_state["carla_status"] == "stopped":
                st.warning("⚠️ Please launch CARLA first!")
            else:
                launch_scenario(xosc_path)
                st.success("✅ Scenario started!")
                st.rerun()

    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    with col4:
        if st.button(
            "🛑 Stop All",
            type="secondary",
            use_container_width=True,
            disabled=not (st.session_state["carla_launched"] or st.session_state["scenario_launched"])
        ):
            stop_all()
            st.success("All processes stopped.")
            st.rerun()

    st.divider()
    st.subheader("📋 Live Logs")
    log_col1, log_col2 = st.columns(2)

    with log_col1:
        st.markdown("**🖥️ CARLA Simulator Logs**")
        carla_logs = _log_buffers["carla_logs"]
        if carla_logs:
            st.code("\n".join(carla_logs[-50:]), language="bash")
        else:
            st.info("No logs yet. Launch CARLA to see output.")

    with log_col2:
        st.markdown("**🏁 Scenario Runner Logs**")
        scenario_logs = _log_buffers["scenario_logs"]
        if scenario_logs:
            st.code("\n".join(scenario_logs[-50:]), language="bash")
        else:
            st.info("No logs yet. Run Scenario to see output.")

    if st.session_state["carla_launched"] or st.session_state["scenario_launched"]:
        time.sleep(2)
        st.rerun()
