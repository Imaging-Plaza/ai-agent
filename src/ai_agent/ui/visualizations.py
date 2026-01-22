import plotly.graph_objects as go
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime


def create_tool_usage_chart(tool_calls: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a bar chart showing tool call frequency.
    """
    if not tool_calls:
        # Empty state
        fig = go.Figure()
        fig.add_annotation(
            text="No tool calls yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    
    # Count tool calls
    tool_names = [tc.get("tool", "unknown") for tc in tool_calls if "tool" in tc]
    counts = Counter(tool_names)
    
    # Sort by call sequence (order of first appearance) instead of frequency
    # This is more meaningful when all tools are called once
    seen = {}
    for i, tc in enumerate(tool_calls):
        tool = tc.get("tool", "unknown")
        if tool not in seen:
            seen[tool] = i
    
    sorted_tools = sorted(counts.items(), key=lambda x: seen.get(x[0], 999))
    tools = [t[0] for t in sorted_tools]
    frequencies = [t[1] for t in sorted_tools]
    
    # Color scheme matching Imaging Plaza green theme
    colors = ["#00A991"] * len(tools)
    
    fig = go.Figure(data=[
        go.Bar(
            x=frequencies,
            y=tools,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{count}×" for count in frequencies],
            textposition='outside',
            textfont=dict(size=12),
            hovertemplate='<b>%{y}</b><br>Called %{x} time(s)<extra></extra>',
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=f"Tool Calls ({len(tool_calls)} total, {len(tools)} unique)",
            font=dict(size=13, color="#333"),
            x=0,
            xanchor='left',
        ),
        height=max(150, 40 + len(tools) * 30),  # Dynamic height based on tool count
        margin=dict(l=10, r=50, t=35, b=30),
        xaxis=dict(
            title="Number of Calls",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[0, max(frequencies) * 1.2] if frequencies else [0, 1],
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        showlegend=False,
    )
    
    return fig


def create_tool_timeline(tool_calls: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a timeline visualization of tool calls in sequence.
    """
    if not tool_calls:
        # Empty state
        fig = go.Figure()
        fig.add_annotation(
            text="No tool calls yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    
    # Extract tool names and create sequence
    tool_names = []
    statuses = []
    timestamps = []
    for tc in tool_calls:
        tool = tc.get("tool", "unknown")
        tool_names.append(tool)
        
        # Capture timestamp if available
        ts = tc.get("timestamp", "")
        timestamps.append(ts)
        
        # Determine status
        if tc.get("blocked"):
            statuses.append("blocked")
        elif tc.get("error"):
            statuses.append("error")
        else:
            statuses.append("success")
    
    # Color mapping
    color_map = {
        "success": "#00A991",  # Imaging Plaza green
        "error": "#FF6B6B",
        "blocked": "#FFA500",
    }
    colors = [color_map.get(s, "#CCCCCC") for s in statuses]
    
    # Create scatter plot as timeline
    x_positions = list(range(1, len(tool_names) + 1))
    
    fig = go.Figure()
    
    # Add trace for each status type
    for status, color in color_map.items():
        indices = [i for i, s in enumerate(statuses) if s == status]
        if indices:
            # Format timestamps for display
            display_timestamps = []
            for i in indices:
                ts = timestamps[i]
                if ts:
                    try:
                        # Parse ISO format and format as HH:MM:SS
                        dt = datetime.fromisoformat(ts)
                        display_timestamps.append(dt.strftime("%H:%M:%S"))
                    except:
                        display_timestamps.append(ts[:19])  # Fallback to raw string
                else:
                    display_timestamps.append("N/A")
            
            fig.add_trace(go.Scatter(
                x=[x_positions[i] for i in indices],
                y=[tool_names[i] for i in indices],
                mode='markers',
                name=status.capitalize(),
                marker=dict(
                    size=12,
                    color=color,
                    line=dict(width=1, color='white'),
                ),
                customdata=display_timestamps,
                hovertemplate='<b>%{y}</b><br>Call #%{x}<br>Time: %{customdata}<extra></extra>',
            ))
    
    fig.update_layout(
        title=dict(
            text=f"Call Sequence ({len(tool_names)} calls)",
            font=dict(size=13, color="#333"),
            x=0,
            xanchor='left',
        ),
        height=max(150, 40 + len(set(tool_names)) * 30),  # Dynamic height
        margin=dict(l=10, r=20, t=60, b=30),
        xaxis=dict(
            title="Order",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            dtick=1,
            range=[0.5, len(tool_names) + 0.5],
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
    )
    
    return fig


def create_disabled_tools_display(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Create a text summary of disabled/blocked tools.
    """
    blocked = [
        tc for tc in tool_calls 
        if tc.get("blocked") or tc.get("reason") == "quota"
    ]
    
    if not blocked:
        return "✅ No tools disabled"
    
    lines = ["⚠️ **Disabled Tools:**\n"]
    for tc in blocked:
        tool_name = tc.get("tool", "unknown")
        reason = tc.get("reason", "unknown")
        cap = tc.get("cap", "?")
        lines.append(f"- `{tool_name}`: {reason} (limit: {cap})")
    
    return "\n".join(lines)