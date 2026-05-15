"""
Generate viz_data.json for the STIG-Net synthetic data dashboard.
Uses only numpy (no torch needed) since we just need raw data export.
Model predictions can be added later when torch env is fixed.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from data.synthetic import generate_all, split_data

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            if obj.dtype == bool:
                return obj.tolist()
            return np.around(obj, 4).tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _serialize(arr):
    """Convert numpy array to JSON-safe list."""
    if isinstance(arr, np.ndarray):
        if arr.dtype == bool:
            return arr.tolist()
        return np.around(arr, 4).tolist()
    return arr


def get_upstream_neighbors(edge_index, link_id):
    """Get links that have edges pointing to link_id (upstream of this link)."""
    return sorted(set(
        int(edge_index[0, e]) for e in range(edge_index.shape[1])
        if edge_index[1, e] == link_id
    ))


def get_downstream_neighbors(edge_index, link_id):
    """Get links that link_id has edges pointing to (downstream of this link)."""
    return sorted(set(
        int(edge_index[1, e]) for e in range(edge_index.shape[1])
        if edge_index[0, e] == link_id
    ))


def main():
    print("=== Phase 1: Generate synthetic data ===")
    data = generate_all(num_links=20, num_days=14, incident_days_start=7,
                        incident_days_end=10, seed=SEED)

    N = data.num_links
    T = data.num_steps
    H_ROADS = 3
    V_ROADS = 5
    N_h = H_ROADS * (V_ROADS - 1)  # 12 horizontal links
    N_v = (H_ROADS - 1) * V_ROADS  # 10 vertical links

    # --- Grid positions for topology view ---
    positions = {}
    for lid in sorted(data.network.links.keys()):
        if lid < N_h:
            row = lid // (V_ROADS - 1)
            col = lid % (V_ROADS - 1)
            x = col + 0.5
            y = float(H_ROADS - 1 - row)
        else:
            vlid = lid - N_h
            col = vlid // (H_ROADS - 1)
            row = vlid % (H_ROADS - 1)
            x = float(col)
            y = float(H_ROADS - 1 - row) - 0.5
        positions[lid] = {"x": round(x * 1.5, 3), "y": round(y * 1.5, 3)}

    # --- Edge export ---
    E = data.network.edge_index
    A = data.network.edge_attr
    turn_names = ["through", "left", "right"]
    edges = []
    for e in range(E.shape[1]):
        edges.append({
            "src": int(E[0, e]),
            "dst": int(E[1, e]),
            "turn": int(A[e, 0]),
            "turn_name": turn_names[int(A[e, 0])],
        })

    # --- Links export ---
    links_export = []
    for lid in range(N):
        link = data.network.links[lid]
        links_export.append({
            "link_id": link.link_id,
            "road_class": link.road_class,
            "road_class_name": ["expressway", "arterial", "collector", "local"][link.road_class],
            "lanes": link.lanes,
            "capacity": link.capacity,
            "free_flow_speed": link.free_flow_speed,
            "length": round(link.length, 3),
            "position": positions[lid],
            "is_horizontal": lid < N_h,
            "direction": "eastbound" if lid < N_h else "southbound",
            "upstream": get_upstream_neighbors(E, lid),
            "downstream": get_downstream_neighbors(E, lid),
        })

    # --- Precompute neighbor info for incident propagation ---
    # For each link, compute k-hop upstream neighbors
    def khop_upstream(edge_index, link_id, k):
        """Return set of upstream link_ids within k hops."""
        if k <= 0:
            return set()
        prev = set()
        current = {link_id}
        all_upstream = set()
        for _ in range(k):
            next_hop = set()
            for lid in current:
                ups = get_upstream_neighbors(edge_index, lid)
                next_hop.update(ups)
            new = next_hop - {link_id} - all_upstream
            all_upstream.update(new)
            current = next_hop
        return all_upstream

    # --- Incidents export ---
    incidents_export = []
    for ev in data.incidents:
        up_1hop = get_upstream_neighbors(E, ev.link_id)
        up_2hop = khop_upstream(E, ev.link_id, 2) - set(up_1hop) - {ev.link_id}
        up_3hop = khop_upstream(E, ev.link_id, 3) - set(up_1hop) - up_2hop - {ev.link_id}
        incidents_export.append({
            "link_id": ev.link_id,
            "start_step": ev.start_step,
            "end_step": ev.start_step + ev.duration_steps,
            "duration_steps": ev.duration_steps,
            "blocked_lanes": ev.blocked_lanes,
            "start_day": ev.start_step // 288,
            "start_hour": round((ev.start_step % 288) * 5 / 60, 2),
            "duration_min": ev.duration_steps * 5,
            "upstream_1hop": sorted(up_1hop),
            "upstream_2hop": sorted(up_2hop),
            "upstream_3hop": sorted(up_3hop),
            "upstream_all_hops": sorted(set(up_1hop) | up_2hop | up_3hop),
        })

    # --- Data quality stats ---
    flow = data.traffic[:, :, 0]
    speed = data.traffic[:, :, 1]
    ff_speeds = np.array([l.free_flow_speed for l in data.network.links.values()])
    speed_ratios = speed / (ff_speeds[:, None] + 1e-8)

    has_normal = (~data.incident_mask).sum() > 0
    has_incident = data.incident_mask.sum() > 0
    speed_drop_pct = 0
    if has_normal and has_incident:
        avg_normal = speed[~data.incident_mask].mean()
        avg_incident = speed[data.incident_mask].mean()
        speed_drop_pct = round((avg_normal - avg_incident) / max(avg_normal, 0.01) * 100, 1)

    # Weather distribution
    wtype_counts = np.zeros(len(data.weather_types))
    for t in range(min(T, data.weather.shape[0])):
        wtype_counts[int(data.weather[t, 4])] += 1
    weather_pct = {
        name: round(wtype_counts[i] / max(wtype_counts.sum(), 1) * 100, 1)
        for i, name in enumerate(data.weather_types)
    }

    # --- Split info ---
    train_data, val_data, test_data = split_data(data, train_days=8, val_days=3)

    print("=== Phase 2: Build export dict ===")
    viz_data = {
        "meta": {
            "num_links": N,
            "num_timesteps": T,
            "num_days": T // 288,
            "steps_per_day": 288,
            "step_minutes": 5,
            "num_incidents": len(data.incidents),
            "num_edges": int(E.shape[1]),
            "grid_h_roads": H_ROADS,
            "grid_v_roads": V_ROADS,
        },
        "network": {
            "links": links_export,
            "edges": edges,
        },
        "data_quality": {
            "flow": {
                "mean": round(float(flow.mean()), 2),
                "std": round(float(flow.std()), 2),
                "min": round(float(flow.min()), 2),
                "max": round(float(flow.max()), 2),
            },
            "speed": {
                "mean": round(float(speed.mean()), 2),
                "std": round(float(speed.std()), 2),
                "min": round(float(speed.min()), 2),
                "max": round(float(speed.max()), 2),
            },
            "speed_ratio_mean": round(float(speed_ratios.mean()), 3),
            "speed_ratio_std": round(float(speed_ratios.std()), 3),
            "speed_drop_incident_pct": speed_drop_pct,
            "weather_pct": weather_pct,
        },
        "traffic": {
            "flow": _serialize(flow),
            "speed": _serialize(speed),
            "delay": _serialize(data.traffic[:, :, 2]),
        },
        "incidents": {
            "events": incidents_export,
            "mask": _serialize(data.incident_mask),
            "features": _serialize(data.incident_features),
        },
        "static_features": {
            "data": _serialize(data.static_features),
            "columns": ["road_class_norm", "capacity_norm", "ff_speed_norm", "length_norm"],
        },
        "weather": {
            "data": _serialize(data.weather),
            "columns": ["temperature", "precipitation", "visibility", "wind_speed", "weather_type_idx"],
            "types": data.weather_types,
        },
        "split_info": {
            "train_steps": 8 * 288,
            "val_steps": 3 * 288,
            "test_steps": 3 * 288,
            "incident_mask_train": _serialize(train_data["incident_mask"]),
            "incident_mask_val": _serialize(val_data["incident_mask"]),
            "incident_mask_test": _serialize(test_data["incident_mask"]),
        },
        "predictions": None,  # Will be added when torch env is fixed
    }

    print(f"=== Phase 3: Write JSON ({N} links × {T} steps) ===")
    json_path = os.path.join(OUT_DIR, "viz_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(viz_data, f, ensure_ascii=False, cls=NumpyEncoder)
    size_mb = os.path.getsize(json_path) / 1024 / 1024
    print(f"Exported: {json_path}")
    print(f"Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
