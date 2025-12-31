import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import altair as alt
import hashlib
import base64
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 1. ENHANCED SYSTEM CONFIGURATION
# =============================================
st.set_page_config(
    page_title="SCAM-EVO // SENTINEL v2.0",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://security.operations/help',
        'Report a bug': 'https://security.operations/bug',
        'About': 'SCAM-EVO Sentinel v2.0 - Production SOC Dashboard'
    }
)

# =============================================
# 2. ENTERPRISE CSS & THEMING
# =============================================
st.markdown("""
    <style>
        /* CORE THEME */
        :root {
            --primary: #00FF41;
            --secondary: #00BFFF;
            --danger: #FF0055;
            --warning: #FFAA00;
            --success: #00FF41;
            --dark-bg: #0A0A0A;
            --card-bg: #111111;
            --border: #222222;
        }
        
        /* MAIN APP */
        .stApp {
            background: linear-gradient(180deg, #000000 0%, #0A0A0A 100%);
            font-family: 'JetBrains Mono', 'SF Mono', 'Courier New', monospace;
        }
        
        /* TYPOGRAPHY */
        h1, h2, h3, h4 {
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
        }
        
        h1 { 
            color: var(--primary) !important;
            font-size: 2rem !important;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            margin-bottom: 20px !important;
        }
        
        h2 { 
            color: #FFFFFF !important;
            font-size: 1.3rem !important;
            margin-top: 25px !important;
        }
        
        h3 { 
            color: #888888 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        
        /* METRIC CARDS - ENHANCED */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #FFFFFF !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #AAAAAA !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* CUSTOM CARDS */
        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            border-color: var(--primary);
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
        }
        
        /* BUTTONS */
        .stButton > button {
            border: 1px solid var(--border) !important;
            background: linear-gradient(135deg, #111111 0%, #222222 100%) !important;
            color: #FFFFFF !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important;
            border-radius: 4px !important;
            padding: 10px 20px !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.85rem !important;
        }
        
        .stButton > button:hover {
            border-color: var(--primary) !important;
            background: linear-gradient(135deg, #222222 0%, #111111 100%) !important;
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.3) !important;
            transform: translateY(-1px);
        }
        
        .primary-btn > button {
            border-color: var(--primary) !important;
            background: linear-gradient(135deg, #003300 0%, #00FF41 100%) !important;
            color: #000000 !important;
            font-weight: 700 !important;
        }
        
        /* SIDEBAR */
        .css-1d391kg {
            background: linear-gradient(180deg, #050505 0%, #0A0A0A 100%);
            border-right: 1px solid var(--border);
        }
        
        /* DATA INPUTS */
        .stSelectbox, .stSlider, .stTextInput {
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
        }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: var(--card-bg);
            border-radius: 4px;
            padding: 5px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 2px;
            padding: 10px 20px;
            color: #888888;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary) !important;
            color: #000000 !important;
            font-weight: 700;
        }
        
        /* HIDE DEFAULT ELEMENTS */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* CUSTOM SCROLLBAR */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #111111;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #00CC33;
        }
        
        /* ANIMATIONS */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        /* STATUS BADGES */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .status-critical { background: rgba(255, 0, 85, 0.2); color: #FF0055; border: 1px solid #FF0055; }
        .status-warning { background: rgba(255, 170, 0, 0.2); color: #FFAA00; border: 1px solid #FFAA00; }
        .status-stable { background: rgba(0, 255, 65, 0.2); color: #00FF41; border: 1px solid #00FF41; }
        .status-info { background: rgba(0, 191, 255, 0.2); color: #00BFFF; border: 1px solid #00BFFF; }
        
        /* GRID LAYOUT HELPERS */
        .grid-container {
            display: grid;
            grid-gap: 15px;
            margin-bottom: 20px;
        }
        
        .grid-2 { grid-template-columns: repeat(2, 1fr); }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        .grid-4 { grid-template-columns: repeat(4, 1fr); }
    </style>
""", unsafe_allow_html=True)

# =============================================
# 2. HELPER FUNCTIONS
# =============================================
def _storage_dir() -> Path:
    return Path(__file__).resolve().parent / "storage"

def _try_read_json(path: Path) -> Dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _parse_iso_dt(value: object) -> datetime | None:
    try:
        s = str(value or "").strip()
        if not s:
            return None
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _discover_runs(storage_dir: Path) -> List[Dict[str, Any]]:
    runs_root = storage_dir / "runs"
    if not runs_root.exists() or not runs_root.is_dir():
        return []

    out: List[Dict[str, Any]] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        summary = _try_read_json(p / "summary.json")
        if not summary:
            continue
        config = _try_read_json(p / "run_config.json") or {}
        created_at = _parse_iso_dt(summary.get("created_at"))
        out.append(
            {
                "run_id": p.name,
                "run_type": str(summary.get("run_type") or ""),
                "created_at": created_at,
                "summary": summary,
                "config": config,
                "path": str(p),
            }
        )

    out.sort(key=lambda r: (r.get("created_at") or datetime.min), reverse=True)
    return out

def _format_run_label(r: Dict[str, Any]) -> str:
    run_id = str(r.get("run_id") or "")
    run_type = str(r.get("run_type") or "")
    created_at = r.get("created_at")
    created_str = created_at.isoformat(timespec="seconds") if isinstance(created_at, datetime) else ""
    s = r.get("summary") if isinstance(r.get("summary"), dict) else {}
    ds = str(s.get("dataset_id") or "")
    mid = str(s.get("model_id") or "")
    return f"{created_str} | {run_type} | run={run_id[:8]} | ds={ds[:8]} | model={mid[:8]}"

def _resolve_latest_retrain_context(storage_dir: Path, *, selected_retrain_run_id: str | None) -> Dict[str, Any] | None:

    runs = _discover_runs(storage_dir)
    retrains = [r for r in runs if str(r.get("run_type") or "") == "adversarial_retrain"]
    if not retrains:
        return None

    chosen: Dict[str, Any] | None = None
    if selected_retrain_run_id:
        for r in retrains:
            if str(r.get("run_id") or "") == str(selected_retrain_run_id):
                chosen = r
                break
    if chosen is None:
        chosen = retrains[0]

    summary = chosen.get("summary") if isinstance(chosen.get("summary"), dict) else None
    if not isinstance(summary, dict):
        return None

    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    baseline_eval = metrics.get("baseline_eval") if isinstance(metrics.get("baseline_eval"), dict) else {}
    defended_eval = metrics.get("defended_eval") if isinstance(metrics.get("defended_eval"), dict) else {}
    defended_attack = metrics.get("defended_attack") if isinstance(metrics.get("defended_attack"), dict) else {}

    dataset_id = str(summary.get("dataset_id") or "")
    baseline_model_id = str(summary.get("model_id") or "")
    defended_model_id = str(summary.get("defended_model_id") or "")
    if not dataset_id or not baseline_model_id:
        return None

    def _find_holdout_attack(model_id: str) -> Dict[str, Any] | None:
        for r in runs:
            if str(r.get("run_type") or "") != "adversarial":
                continue
            rs = r.get("summary") if isinstance(r.get("summary"), dict) else {}
            rc = r.get("config") if isinstance(r.get("config"), dict) else {}
            if str(rs.get("dataset_id") or "") != dataset_id:
                continue
            if str(rs.get("model_id") or "") != model_id:
                continue
            if str(rc.get("split") or "") != "holdout":
                continue
            return r
        return None

    baseline_holdout_attack = _find_holdout_attack(baseline_model_id)
    defended_holdout_attack = _find_holdout_attack(defended_model_id) if defended_model_id else None

    baseline_evasion = None
    defended_evasion = None
    holdout_total = 0

    if baseline_holdout_attack:
        bs = baseline_holdout_attack.get("summary") if isinstance(baseline_holdout_attack.get("summary"), dict) else {}
        baseline_evasion = bs.get("evasion_rate")
        holdout_total = int(bs.get("total_candidates", 0) or 0)

    if defended_holdout_attack:
        ds = defended_holdout_attack.get("summary") if isinstance(defended_holdout_attack.get("summary"), dict) else {}
        defended_evasion = ds.get("evasion_rate")

    if baseline_evasion is None:
        baseline_evasion = summary.get("evasion_rate", 0.0)
    if defended_evasion is None:
        defended_evasion = defended_attack.get("evasion_rate", baseline_evasion)

    baseline_f1 = float(baseline_eval.get("f1", 0.0) or 0.0)
    defended_f1 = float(defended_eval.get("f1", baseline_f1) or baseline_f1)

    ds_meta = _try_read_json(storage_dir / "datasets" / dataset_id / "meta.json")
    return {
        "retrain_run_id": str(chosen.get("run_id") or ""),
        "dataset_id": dataset_id,
        "baseline_model_id": baseline_model_id,
        "defended_model_id": defended_model_id,
        "baseline_f1": max(0.0, min(1.0, float(baseline_f1))),
        "defended_f1": max(0.0, min(1.0, float(defended_f1))),
        "baseline_evasion": max(0.0, min(1.0, float(baseline_evasion or 0.0))),
        "defended_evasion": max(0.0, min(1.0, float(defended_evasion or 0.0))),
        "holdout_attack_total": int(holdout_total),
        "ds_meta": ds_meta,
    }

# =============================================
# 3. ENHANCED STATE MANAGEMENT
# =============================================
class SentinelState:
    """Enterprise-grade state management for the Sentinel system"""
    
    @staticmethod
    def initialize():
        """Initialize or reset all session state variables"""
        defaults = {
            # Core Simulation
            "sim_active": False,
            "step": 0,
            "phase": 0.0,
            "last_update": datetime.now(),
            
            # Data Storage
            "history": [],
            "alerts": [],
            "threat_log": [],
            "performance_metrics": {
                "f1": 0.55,
                "evasion": 0.45,
                "response_time": 120.0,
                "throughput": 800.0,
                "confidence": 0.75,
            },
            "use_real_artifacts": False,
            "selected_retrain_run_id": None,
            "real_ctx": None,
            "applied_retrain_run_id": None,
            "applied_use_real_artifacts": False,
            
            # System Configuration
            "config": {
                "simulation_speed": 50,
                "alert_threshold": 0.2,
                "f1_target": 0.95,
                "max_history": 1000,
                "auto_response": True,
                "notification_level": "critical"
            },
            
            # Dashboard State
            "selected_view": "Overview",
            "alert_filter": "all",
            "time_range": "1h",
            "dark_mode": True
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

# Initialize state
SentinelState.initialize()

# =============================================
# 4. ENHANCED DATA ENGINE
# =============================================
class DataEngine:
    """Production-grade data generation and management"""
    
    @staticmethod
    def generate_manifold_data(step: int, phase: float) -> Dict[str, Any]:
        """Generate 3D manifold data with realistic threat patterns"""
        np.random.seed(step % 1000)  # Deterministic but varied
        
        # Dynamic grid resolution based on phase
        resolution = 20 + int(30 * phase)
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-5, 5, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Animated pulse effect
        pulse = np.sin(step * 0.15) * 0.3
        
        # Complex manifold equation representing model convergence
        Z = (
            np.sin(np.sqrt(X**2 + Y**2) * 1.5 - step * 0.08) * (1.0 - phase) +
            (X**2 + Y**2) * 0.08 * phase +
            0.5 * np.sin(X * 0.8 + step * 0.05) * np.cos(Y * 0.8 + step * 0.05) * (1.0 - phase) +
            pulse
        )
        
        # Normalize Z for better visualization
        Z = (Z - Z.min()) / (Z.max() - Z.min()) * 2 - 1
        
        return {"X": X, "Y": Y, "Z": Z}
    
    @staticmethod
    def generate_threat_data(step: int, phase: float) -> Dict[str, Any]:
        """Generate threat and legitimate traffic data"""
        np.random.seed(step % 1000 + 1000)
        
        # Number of points scales with phase (more detection = more identified threats)
        n_safe = 300
        n_threats = 150 + int(100 * phase)  # More threats visible as detection improves
        
        # Safe traffic - clustered in center
        safe_x = np.random.normal(0, 1.2, n_safe)
        safe_y = np.random.normal(0, 1.2, n_safe)
        safe_z = (safe_x**2 + safe_y**2) * 0.1 * phase + np.random.normal(0, 0.1, n_safe)
        
        # Threat traffic - distribution changes with phase
        threat_phase = phase
        if phase < 0.3:
            # Early phase: threats hidden among safe traffic
            threat_x = np.random.normal(0, 1.5, n_threats)
            threat_y = np.random.normal(0, 1.5, n_threats)
        elif phase < 0.7:
            # Mid phase: threats moving outward
            spread = 2.0 + 2.0 * ((phase - 0.3) / 0.4)
            threat_x = np.random.normal(0, spread, n_threats)
            threat_y = np.random.normal(0, spread, n_threats)
        else:
            # Late phase: threats isolated at perimeter
            angles = np.random.uniform(0, 2*np.pi, n_threats)
            radii = 3.5 + np.random.normal(0, 0.5, n_threats)
            threat_x = radii * np.cos(angles)
            threat_y = radii * np.sin(angles)
        
        threat_z = (threat_x**2 + threat_y**2) * 0.1 * phase + 0.5 + np.random.normal(0, 0.2, n_threats)
        
        # Assign threat types based on position
        threat_types = []
        for i in range(n_threats):
            r = np.sqrt(threat_x[i]**2 + threat_y[i]**2)
            if r < 2.0:
                threat_types.append("stealth")
            elif r < 4.0:
                threat_types.append("evasive")
            else:
                threat_types.append("aggressive")
        
        return {
            "safe": {"x": safe_x, "y": safe_y, "z": safe_z},
            "threats": {"x": threat_x, "y": threat_y, "z": threat_z, "types": threat_types}
        }
    
    @staticmethod
    def generate_metrics(step: int, phase: float) -> Dict[str, Any]:
        """Generate realistic performance metrics"""
        np.random.seed(step % 1000 + 2000)
        
        # Base metrics with noise and trends
        time_factor = step * 0.01
        
        # F1 Score - improves with phase but has realistic noise
        f1_base = 0.5 + 0.45 * phase
        f1_noise = np.random.normal(0, 0.02)
        f1_trend = 0.01 * np.sin(time_factor)  # Small cyclic trend
        f1 = min(0.98, max(0.5, f1_base + f1_noise + f1_trend))
        
        # Evasion Rate - decreases with phase
        evasion_base = 0.4 * (1.0 - phase)
        evasion_noise = np.random.normal(0, 0.015)
        evasion = max(0.01, min(0.5, evasion_base + evasion_noise))
        
        # Response Time - improves with phase
        response_base = 150 - 100 * phase
        response_noise = np.random.normal(0, 10)
        response = max(20, response_base + response_noise)
        
        # Throughput - increases with phase
        throughput_base = 500 + 1000 * phase
        throughput_noise = np.random.normal(0, 50)
        throughput = max(500, throughput_base + throughput_noise)
        
        # Confidence score
        confidence = 0.6 + 0.35 * phase + np.random.normal(0, 0.05)
        confidence = min(0.99, max(0.5, confidence))
        
        return {
            "f1": f1,
            "evasion": evasion,
            "response_time": response,
            "throughput": throughput,
            "confidence": confidence,
            "timestamp": datetime.now()
        }
    
    @staticmethod
    def generate_alerts(step: int, phase: float) -> List[Dict[str, Any]]:
        """Generate realistic security alerts"""
        np.random.seed(step % 1000 + 3000)
        
        alerts = []
        n_alerts = np.random.poisson(3)  # Average 3 alerts per step
        
        alert_types = ["Phishing", "Malware", "DDoS", "Insider", "Zero-Day", "Credential", "Lateral"]
        severities = ["low", "medium", "high", "critical"]
        
        for _ in range(n_alerts):
            alert_time = datetime.now() - timedelta(seconds=np.random.randint(0, 300))
            alert_type = np.random.choice(alert_types)
            
            # Adjust severity based on phase
            if phase < 0.3:
                severity_weights = [0.1, 0.3, 0.4, 0.2]
            elif phase < 0.7:
                severity_weights = [0.3, 0.4, 0.2, 0.1]
            else:
                severity_weights = [0.5, 0.3, 0.15, 0.05]
            
            severity = np.random.choice(severities, p=severity_weights)
            
            alerts.append({
                "id": hashlib.md5(f"{step}_{_}".encode()).hexdigest()[:8],
                "type": alert_type,
                "severity": severity,
                "timestamp": alert_time,
                "description": f"{alert_type} attempt detected",
                "source": f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}",
                "status": "new" if np.random.random() > 0.7 else "investigating"
            })
        
        return alerts

# =============================================
# 5. ENHANCED VISUALIZATION COMPONENTS
# =============================================
class VisualizationEngine:
    """Production-grade visualization components"""
    
    @staticmethod
    def create_3d_manifold(manifold_data: Dict[str, Any], threat_data: Dict[str, Any], step: int) -> go.Figure:
        """Create enhanced 3D manifold visualization"""
        fig = go.Figure()
        
        # Enhanced surface with lighting
        fig.add_trace(go.Surface(
            z=manifold_data["Z"],
            x=manifold_data["X"],
            y=manifold_data["Y"],
            colorscale=[
                [0, "#000000"],
                [0.2, "#330066"],
                [0.4, "#6600CC"],
                [0.6, "#9900FF"],
                [0.8, "#CC66FF"],
                [1, "#FFFFFF"]
            ],
            opacity=0.9,
            showscale=False,
            contours={
                "z": {"show": True, "start": -1, "end": 1, "size": 0.2, "color": "rgba(255,255,255,0.1)"}
            },
            lighting={
                "ambient": 0.4,
                "diffuse": 0.6,
                "fresnel": 0.1,
                "specular": 0.5,
                "roughness": 0.8
            },
            lightposition={"x": 100, "y": 200, "z": 100}
        ))
        
        # Safe traffic
        fig.add_trace(go.Scatter3d(
            x=threat_data["safe"]["x"],
            y=threat_data["safe"]["y"],
            z=threat_data["safe"]["z"],
            mode='markers',
            marker=dict(
                size=3,
                color='#00FF41',
                opacity=0.7,
                line=dict(width=0)
            ),
            name='Legitimate',
            hoverinfo='skip'
        ))
        
        # Threat traffic with color coding
        threat_colors = {
            "stealth": "#FFAA00",
            "evasive": "#FF5500",
            "aggressive": "#FF0055"
        }
        
        for threat_type in ["stealth", "evasive", "aggressive"]:
            mask = [t == threat_type for t in threat_data["threats"]["types"]]
            if any(mask):
                fig.add_trace(go.Scatter3d(
                    x=np.array(threat_data["threats"]["x"])[mask],
                    y=np.array(threat_data["threats"]["y"])[mask],
                    z=np.array(threat_data["threats"]["z"])[mask],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=threat_colors[threat_type],
                        opacity=0.9,
                        line=dict(width=1, color='white')
                    ),
                    name=f'Threat: {threat_type}',
                    hoverinfo='skip'
                ))
        
        # Camera animation
        camera_angle = step * 0.5
        camera_dict = dict(
            eye=dict(
                x=1.5 * np.cos(np.radians(camera_angle)),
                y=1.5 * np.sin(np.radians(camera_angle)),
                z=1.2
            ),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                camera=camera_dict,
                aspectmode='data',
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='#333333',
                borderwidth=1,
                font=dict(color='white', size=10)
            )
        )
        
        return fig
    
    @staticmethod
    def create_performance_dashboard(history: List[Dict[str, Any]]) -> go.Figure:
        """Create comprehensive performance dashboard"""
        if not history:
            return go.Figure()
        
        df = pd.DataFrame(history)
        
        fig = go.Figure()
        
        # F1 Score - primary metric
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['f1'],
            mode='lines+markers',
            name='F1 Score',
            line=dict(color='#00FF41', width=3),
            marker=dict(size=4),
            yaxis='y1'
        ))
        
        # Evasion Rate
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['evasion'],
            mode='lines',
            name='Evasion Rate',
            line=dict(color='#FF0055', width=2, dash='dot'),
            yaxis='y2'
        ))
        
        # Response Time
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['response_time'],
            mode='lines',
            name='Response Time (ms)',
            line=dict(color='#00BFFF', width=2),
            yaxis='y3',
            visible='legendonly'
        ))
        
        # Throughput
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['throughput'],
            mode='lines',
            name='Throughput (TPS)',
            line=dict(color='#FFAA00', width=2),
            yaxis='y4',
            visible='legendonly'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=40, b=40, t=40, pad=10),
            height=350,
            hovermode='x unified',
            xaxis=dict(
                showgrid=True,
                gridcolor='#333333',
                title="Time",
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(
                title="F1 Score",
                side='left',
                position=0,
                color='#00FF41',
                range=[0, 1.05],
                gridcolor='#333333'
            ),
            yaxis2=dict(
                title="Evasion Rate",
                side='right',
                overlaying='y',
                position=1,
                color='#FF0055',
                range=[0, 0.5],
                gridcolor='#333333'
            ),
            yaxis3=dict(
                title="Response Time",
                side='right',
                overlaying='y',
                position=0.85,
                color='#00BFFF',
                gridcolor='#333333',
                visible=False
            ),
            yaxis4=dict(
                title="Throughput",
                side='right',
                overlaying='y',
                position=0.7,
                color='#FFAA00',
                gridcolor='#333333',
                visible=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0.7)'
            )
        )
        
        return fig
    
    @staticmethod
    def create_threat_matrix(step: int) -> go.Figure:
        """Create advanced threat matrix visualization"""
        np.random.seed(step % 1000)
        
        # Generate more complex patterns
        size = 50
        data = np.zeros((size, size))
        
        # Add various threat patterns
        for _ in range(5):
            cx = np.random.randint(0, size)
            cy = np.random.randint(0, size)
            radius = np.random.randint(5, 15)
            intensity = np.random.random() * 0.8 + 0.2
            
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist < radius:
                        data[i, j] = max(data[i, j], intensity * (1 - dist/radius))
        
        # Add scan line
        scan_line = step % size
        data[scan_line, :] = 1.0
        
        # Add random noise
        noise = np.random.rand(size, size) * 0.1
        data = np.clip(data + noise, 0, 1)
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            colorscale=[
                [0, "#000000"],
                [0.3, "#330033"],
                [0.6, "#660066"],
                [0.8, "#990099"],
                [1, "#FF00FF"]
            ],
            showscale=False,
            zsmooth='best'
        ))
        
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False)
        )
        
        return fig
    
    @staticmethod
    def create_gauge_chart(value: float, title: str, color_scale: List[str]) -> go.Figure:
        """Create a gauge chart for single metrics"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={'suffix': '%', 'font': {'size': 24}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color_scale[1]},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333333",
                'steps': [
                    {'range': [0, 50], 'color': color_scale[0]},
                    {'range': [50, 80], 'color': color_scale[1]},
                    {'range': [80, 100], 'color': color_scale[2]}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            height=200,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig

# =============================================
# 6. SIDEBAR CONTROLS - ENHANCED
# =============================================
def render_sidebar():
    """Render the enhanced sidebar controls"""
    with st.sidebar:
        # Header with logo
        st.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <h1 style='color: #00FF41; font-size: 1.8rem;'>🛡️ SCAM-EVO</h1>
                <p style='color: #888; font-size: 0.9rem;'>SENTINEL CONTROL PANEL v2.0</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        storage_dir = _storage_dir()
        runs = _discover_runs(storage_dir)
        retrains = [r for r in runs if str(r.get("run_type") or "") == "adversarial_retrain"]

        st.markdown("### 📦 REAL ARTIFACTS")
        default_real = bool(retrains)
        if "use_real_artifacts" not in st.session_state:
            st.session_state.use_real_artifacts = default_real
        st.session_state.use_real_artifacts = st.toggle("Use Storage Runs", value=bool(st.session_state.use_real_artifacts) and default_real)

        if st.session_state.use_real_artifacts and retrains:
            run_ids = [str(r.get("run_id") or "") for r in retrains]
            run_map = {str(r.get("run_id") or ""): r for r in retrains}

            def _fmt_run(rid: str) -> str:
                rr = run_map.get(str(rid))
                return _format_run_label(rr) if rr else str(rid)

            selected = st.selectbox(
                "Latest Retrain Run",
                options=run_ids,
                format_func=_fmt_run,
                index=0,
            )
            st.session_state.selected_retrain_run_id = str(selected)
            st.session_state.real_ctx = _resolve_latest_retrain_context(
                storage_dir,
                selected_retrain_run_id=str(selected),
            )

            rc = st.session_state.real_ctx if isinstance(st.session_state.real_ctx, dict) else None
            if rc:
                current_run_id = str(rc.get("retrain_run_id") or "")
                last_run_id = str(st.session_state.get("applied_retrain_run_id") or "")
                last_enabled = bool(st.session_state.get("applied_use_real_artifacts"))
                if current_run_id and (current_run_id != last_run_id or not last_enabled):
                    st.session_state.applied_retrain_run_id = current_run_id
                    st.session_state.applied_use_real_artifacts = True
                    st.session_state.phase = 0.0

                    if not bool(st.session_state.sim_active):
                        m = DataEngine.generate_metrics(int(st.session_state.step), float(st.session_state.phase))
                        m["f1"] = float(rc.get("baseline_f1", m.get("f1")) or (m.get("f1") or 0.0))
                        m["evasion"] = float(rc.get("baseline_evasion", m.get("evasion")) or (m.get("evasion") or 0.0))
                        m["confidence"] = max(0.5, min(0.99, 0.45 + 0.55 * float(m["f1"])))
                        st.session_state.performance_metrics = {
                            "f1": float(m.get("f1", 0.0) or 0.0),
                            "evasion": float(m.get("evasion", 0.0) or 0.0),
                            "response_time": float(m.get("response_time", 0.0) or 0.0),
                            "throughput": float(m.get("throughput", 0.0) or 0.0),
                            "confidence": float(m.get("confidence", 0.0) or 0.0),
                        }

                st.caption(
                    f"ds={str(rc.get('dataset_id') or '')[:8]} | base={str(rc.get('baseline_model_id') or '')[:8]} | def={str(rc.get('defended_model_id') or '')[:8]}"
                )
        else:
            st.session_state.real_ctx = None
            st.session_state.applied_use_real_artifacts = False
        
        # System Controls
        st.markdown("### 🎛️ SYSTEM CONTROLS")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ **START**", type="primary", use_container_width=True):
                st.session_state.sim_active = True
                st.session_state.last_update = datetime.now()
                st.rerun()
        
        with col2:
            if st.button("⏸ **PAUSE**", use_container_width=True):
                st.session_state.sim_active = False
                st.rerun()
        
        if st.button("↺ **FULL RESET**", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            SentinelState.initialize()
            st.rerun()
        
        st.markdown("---")
        
        # Configuration
        st.markdown("### ⚙️ CONFIGURATION")
        
        st.session_state.config["simulation_speed"] = st.slider(
            "Clock Speed (ms)",
            10, 1000, 100,
            help="Controls simulation update frequency"
        )
        
        st.session_state.config["alert_threshold"] = st.slider(
            "Alert Threshold",
            0.0, 1.0, 0.2, 0.05,
            help="Threshold for generating security alerts"
        )
        
        st.session_state.config["f1_target"] = st.slider(
            "F1 Target",
            0.5, 1.0, 0.95, 0.01,
            help="Target F1 score for the model"
        )
        
        st.session_state.config["auto_response"] = st.toggle(
            "Auto Response",
            value=True,
            help="Enable automatic threat response"
        )
        
        st.session_state.config["notification_level"] = st.selectbox(
            "Notification Level",
            ["low", "medium", "high", "critical"],
            index=3
        )
        
        st.markdown("---")
        
        # Dashboard Settings
        st.markdown("### 📊 DISPLAY")
        
        view_options = ["Overview", "Threat Analysis", "Performance", "System Health", "Alerts"]
        st.session_state.selected_view = st.selectbox(
            "View Mode",
            view_options,
            index=view_options.index(st.session_state.selected_view) if st.session_state.selected_view in view_options else 0
        )
        
        time_ranges = ["15m", "1h", "6h", "24h", "7d"]
        st.session_state.time_range = st.selectbox(
            "Time Range",
            time_ranges,
            index=time_ranges.index(st.session_state.time_range) if st.session_state.time_range in time_ranges else 1
        )
        
        st.session_state.dark_mode = st.toggle("Dark Mode", value=True)
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 📡 SYSTEM STATUS")
        
        status_color = "#00FF41" if st.session_state.sim_active else "#FF0055"
        status_text = "**LIVE**" if st.session_state.sim_active else "**STANDBY**"
        
        st.markdown(f"""
            <div style='background: rgba(17, 17, 17, 0.8); padding: 15px; border-radius: 4px; border-left: 4px solid {status_color};'>
                <p style='margin: 0; color: {status_color}; font-weight: bold;'>{status_text}</p>
                <p style='margin: 5px 0 0 0; color: #888; font-size: 0.85rem;'>
                    Uptime: {st.session_state.step * st.session_state.config["simulation_speed"] / 1000:.1f}s<br>
                    Last update: {st.session_state.last_update.strftime('%H:%M:%S')}
                </p>
            </div>
        """, unsafe_allow_html=True)

# =============================================
# 7. MAIN DASHBOARD COMPONENTS
# =============================================
def render_overview():
    """Render the main overview dashboard"""
    
    # Header
    st.markdown(f"""
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
            <div>
                <h1>THREAT INTELLIGENCE DASHBOARD</h1>
                <p style='color: #888; margin-top: -10px;'>Real-time Adversarial Defense Monitoring</p>
            </div>
            <div style='text-align: right;'>
                <span class='status-badge status-stable'>LIVE</span>
                <p style='color: #888; font-size: 0.9rem; margin-top: 5px;'>
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Top Metrics Row
    st.markdown('<div class="grid-container grid-4">', unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="CONFIDENCE SCORE",
            value=f"{st.session_state.performance_metrics.get('confidence', 0.75)*100:.1f}%",
            delta="+2.4%" if st.session_state.sim_active else "0.0%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        evasion = float(st.session_state.performance_metrics.get('evasion', 0.45) or 0.0)
        st.metric(
            label="EVASION RATE",
            value=f"{evasion*100:.1f}%",
            delta="-" if st.session_state.phase >= 0.8 else "+"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        f1 = float(st.session_state.performance_metrics.get('f1', 0.55) or 0.0)
        st.metric(
            label="F1 SCORE",
            value=f"{f1*100:.1f}%",
            delta="+" if st.session_state.phase >= 0.8 else "Live"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[3]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        tps = float(st.session_state.performance_metrics.get('throughput', 800.0) or 0.0)
        st.metric(
            label="THROUGHPUT (TPS)",
            value=f"{tps:,.0f}",
            delta="+" if st.session_state.sim_active else "0"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    rc = st.session_state.real_ctx if isinstance(st.session_state.real_ctx, dict) else None
    if rc:
        st.markdown(
            f"<span class='status-badge status-info'>REAL</span> Retrain run: <code>{str(rc.get('retrain_run_id') or '')}</code>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.4, 1.0])
    with left:
        manifold = DataEngine.generate_manifold_data(int(st.session_state.step), float(st.session_state.phase))
        threats = DataEngine.generate_threat_data(int(st.session_state.step), float(st.session_state.phase))
        fig3d = VisualizationEngine.create_3d_manifold(manifold, threats, int(st.session_state.step))
        st.plotly_chart(fig3d, use_container_width=True)

    with right:
        tm = VisualizationEngine.create_threat_matrix(int(st.session_state.step))
        st.plotly_chart(tm, use_container_width=True)

        gauges = st.columns(2)
        with gauges[0]:
            st.plotly_chart(
                VisualizationEngine.create_gauge_chart(
                    float(st.session_state.performance_metrics.get('f1', 0.0) or 0.0),
                    "F1",
                    ["rgba(255,0,85,0.25)", "rgba(255,170,0,0.25)", "rgba(0,255,65,0.25)"],
                ),
                use_container_width=True,
            )
        with gauges[1]:
            st.plotly_chart(
                VisualizationEngine.create_gauge_chart(
                    1.0 - float(st.session_state.performance_metrics.get('evasion', 0.0) or 0.0),
                    "ROBUST",
                    ["rgba(255,0,85,0.25)", "rgba(255,170,0,0.25)", "rgba(0,255,65,0.25)"],
                ),
                use_container_width=True,
            )

    st.markdown("### 📈 PERFORMANCE")
    hist = st.session_state.history if isinstance(st.session_state.history, list) else []
    st.plotly_chart(VisualizationEngine.create_performance_dashboard(hist[-250:]), use_container_width=True)


def _tick_once() -> None:
    st.session_state.step = int(st.session_state.step) + 1
    if bool(st.session_state.sim_active):
        st.session_state.phase = float(min(1.0, float(st.session_state.phase) + 0.02))

    rc = st.session_state.real_ctx if isinstance(st.session_state.real_ctx, dict) else None
    base_f1 = None
    def_f1 = None
    base_ev = None
    def_ev = None
    if rc:
        base_f1 = float(rc.get('baseline_f1', 0.55) or 0.55)
        def_f1 = float(rc.get('defended_f1', base_f1) or base_f1)
        base_ev = float(rc.get('baseline_evasion', 0.45) or 0.45)
        def_ev = float(rc.get('defended_evasion', base_ev) or base_ev)

    m = DataEngine.generate_metrics(int(st.session_state.step), float(st.session_state.phase))
    if base_f1 is not None and def_f1 is not None and base_ev is not None and def_ev is not None:
        p = float(st.session_state.phase)
        m['f1'] = max(0.0, min(1.0, base_f1 + (def_f1 - base_f1) * p))
        m['evasion'] = max(0.0, min(1.0, base_ev + (def_ev - base_ev) * p))
        m['confidence'] = max(0.5, min(0.99, 0.45 + 0.55 * float(m['f1'])))

    st.session_state.performance_metrics = {
        'f1': float(m.get('f1', 0.0) or 0.0),
        'evasion': float(m.get('evasion', 0.0) or 0.0),
        'response_time': float(m.get('response_time', 0.0) or 0.0),
        'throughput': float(m.get('throughput', 0.0) or 0.0),
        'confidence': float(m.get('confidence', 0.0) or 0.0),
    }

    if not isinstance(st.session_state.history, list):
        st.session_state.history = []
    st.session_state.history.append(
        {
            'timestamp': m.get('timestamp', datetime.now()),
            'f1': float(m.get('f1', 0.0) or 0.0),
            'evasion': float(m.get('evasion', 0.0) or 0.0),
            'response_time': float(m.get('response_time', 0.0) or 0.0),
            'throughput': float(m.get('throughput', 0.0) or 0.0),
        }
    )
    st.session_state.history = st.session_state.history[-int(st.session_state.config.get('max_history', 1000) or 1000):]

    st.session_state.last_update = datetime.now()


render_sidebar()

if bool(st.session_state.sim_active):
    _tick_once()

if str(st.session_state.selected_view) == "Overview":
    render_overview()
else:
    st.info("This view is not implemented yet. Switch to Overview.")

if bool(st.session_state.sim_active):
    time.sleep(float(st.session_state.config.get('simulation_speed', 100) or 100) / 1000.0)
    st.rerun()