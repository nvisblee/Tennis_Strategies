import streamlit as st
from google.api_core import client_options as client_options_lib
import google.generativeai as genai
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
import tempfile
import numpy as np
import json

# ----------------------------
# Secrets / Keys
# ----------------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(layout="wide", page_title="Tennis Strategy Analyzer")
st.title("Tennis Point Strategy Analyzer")


# ----------------------------
# Helpers
# ----------------------------
def validate_coordinates(coords, coord_type="position"):
    if coords is None:
        return None

    if isinstance(coords, dict):
        coords = [coords.get("x"), coords.get("y")]

    if not isinstance(coords, (list, tuple)) or len(coords) != 2:
        return None

    if coords[0] is None or coords[1] is None:
        return None

    try:
        x, y = float(coords[0]), float(coords[1])
    except Exception:
        return None

    # Visualization bounds (expanded) so we can still draw if slightly off
    x_min_viz, x_max_viz = -10, 50
    y_min_viz, y_max_viz = -10, 90

    # Court bounds for "end" checks
    if "end" in coord_type.lower():
        court_width = 36
        court_length = 78
        if not (0 <= x <= court_width and 0 <= y <= court_length):
            st.info(
                f"{coord_type} at [{x:.1f}, {y:.1f}] is outside the court boundaries."
            )

    if x < x_min_viz or x > x_max_viz or y < y_min_viz or y > y_max_viz:
        st.warning(
            f"{coord_type} coordinates [{x:.1f}, {y:.1f}] are far outside the court. "
            "The visualization may be skewed, but proceeding."
        )

    return [x, y]


def validate_and_fix_player_positions(player1_pos, player2_pos):
    if player1_pos is None or player2_pos is None:
        return player1_pos, player2_pos

    if player1_pos[0] == player2_pos[0] and player1_pos[1] == player2_pos[1]:
        st.error(
            "Both players detected at identical positions. This likely indicates an extraction failure."
        )
        return None, None

    # Heuristic: Player 1 should usually be closer to camera (lower y)
    if player1_pos[1] > player2_pos[1] and player1_pos[1] > 45 and player2_pos[1] < 30:
        st.warning("Player positions appear swapped based on y-coordinates. Correcting.")
        return player2_pos, player1_pos

    return player1_pos, player2_pos


def correct_shot_start_position(shot_start, player_pos):
    if shot_start is None or player_pos is None:
        return shot_start

    distance = np.sqrt(
        (shot_start[0] - player_pos[0]) ** 2 + (shot_start[1] - player_pos[1]) ** 2
    )

    if distance > 8.0:
        st.info(
            f"Correcting shot start position from [{shot_start[0]:.1f}, {shot_start[1]:.1f}] "
            f"to player position [{player_pos[0]:.1f}, {player_pos[1]:.1f}]"
        )
        return player_pos.copy()

    return shot_start


def create_tennis_court():
    fig_plot, ax_plot = plt.subplots(figsize=(5, 10))
    ax_plot.set_facecolor("#88b2b9")

    court_width = 36
    court_length = 78

    court = patches.Rectangle(
        (0, 0),
        court_width,
        court_length,
        linewidth=2,
        edgecolor="white",
        facecolor="#3a9e5c",
    )
    ax_plot.add_patch(court)

    # Baselines
    plt.plot([0, court_width], [0, 0], "white", linewidth=2)
    plt.plot([0, court_width], [court_length, court_length], "white", linewidth=2)

    # Singles sidelines
    singles_width = 27
    margin = (court_width - singles_width) / 2
    plt.plot([margin, margin], [0, court_length], "white", linewidth=2)
    plt.plot([court_width - margin, court_width - margin], [0, court_length], "white", linewidth=2)

    # Service lines
    service_line_dist = 21
    plt.plot([margin, court_width - margin], [service_line_dist, service_line_dist], "white", linewidth=2)
    plt.plot([margin, court_width - margin], [court_length - service_line_dist, court_length - service_line_dist], "white", linewidth=2)

    # Center service line
    plt.plot([court_width / 2, court_width / 2], [service_line_dist, court_length - service_line_dist], "white", linewidth=2)

    # Center marks
    center_mark_width = 0.5
    plt.plot(
        [court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2],
        [0, 0],
        "white",
        linewidth=2,
    )
    plt.plot(
        [court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2],
        [court_length, court_length],
        "white",
        linewidth=2,
    )

    # Net line (for visualization)
    plt.plot([0, court_width], [court_length / 2, court_length / 2], "white", linewidth=3)

    plt.xlim(-5, court_width + 5)
    plt.ylim(-5, court_length + 5)
    ax_plot.set_xticks([])
    ax_plot.set_yticks([])
    plt.axis("equal")

    return fig_plot, ax_plot


def parse_keyshot_timestamp(key_shot_text: str):
    """
    Returns (key_shot_timestamp_str or None, key_shot_text_without_timestamp_line)
    """
    if not key_shot_text:
        return None, key_shot_text

    lines = [ln.strip() for ln in key_shot_text.split("\n") if ln.strip()]
    ts_line = None
    for ln in lines:
        if ln.lower().startswith("key shot timestamp:"):
            ts_line = ln
            break

    if not ts_line:
        return None, "\n".join(lines)

    timestamp_str = ts_line.split(":", 1)[1].strip()
    remaining = [ln for ln in lines if ln != ts_line]
    return timestamp_str, "\n".join(remaining)


@st.cache_resource
def get_genai_client():
    genai.configure(
        api_key=GOOGLE_API_KEY,
        transport="rest",
        client_options=client_options_lib.ClientOptions(
            api_endpoint=os.getenv("GOOGLE_API_ENDPOINT"),
        ),
    )
    return genai


genai_client = None
try:
    genai_client = get_genai_client()
except Exception as e:
    st.error(f"Failed to initialize Google GenAI Client: {e}")


# ----------------------------
# UI
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a Tennis Point Video (MP4)", type=["mp4", "mov", "avi", "webm", "mkv"]
)

if uploaded_file is None:
    st.info("Please upload a tennis point video to start the analysis.")
    st.markdown(
        """
## How This Works
1. Upload a Tennis Point Video
2. AI Analysis:
   - Overall game strategy and patterns
   - Court positioning and movement
   - Shot selection (high-level; not a shot-by-shot narration)
3. Results:
   - Personalized coaching analysis
   - Identification of a key shot that influenced the point (if possible)
   - Tactical shot recommendation with a visualization
"""
    )
    st.stop()

if not genai_client:
    st.error("API client could not be initialized. Please check API keys/configurations and restart.")
    st.stop()

st.video(uploaded_file)

# Save upload to temp file (for local preview only; Gemini uses uploaded file URI after genai upload)
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    video_file_path = tmp_file.name

try:
    # Size checks
    if uploaded_file.size > 150 * 1024 * 1024:
        st.warning("This video is too large to process. Please upload a smaller video.")
        st.stop()
    elif uploaded_file.size > 52 * 1024 * 1024:
        st.warning("Large video detected. Processing may take longer.")

    st.info("Video uploaded. Analyzing...")

    analysis_successful = False

    # ----------------------------
    # 1) Upload file to Google AI
    # ----------------------------
    with st.spinner("Uploading file to Google AI..."):
        video_file_obj = genai_client.upload_file(path=video_file_path)

        processing_placeholder = st.empty()
        progress_counter = 0
        while video_file_obj.state.name == "PROCESSING":
            processing_placeholder.text(f"Processing video... {progress_counter}s")
            progress_counter += 2
            time.sleep(2)
            video_file_obj = genai_client.get_file(name=video_file_obj.name)

        processing_placeholder.empty()

        if video_file_obj.state.name == "FAILED":
            st.error(f"Video processing failed: {video_file_obj.state.name}")
            st.stop()

        st.success("File successfully processed by Google AI.")

    # ----------------------------
    # 2) Coaching Analysis (Gemini 3)
    # ----------------------------
    with st.spinner("Generating coaching analysis..."):
        coach_prompt = """
You are an elite tennis strategy coach with computer vision capabilities.
Analyze ONLY the player CLOSEST TO THE CAMERA.
Speak directly to them as if you were their coach between games.

STRICT RULES:
- Do NOT describe or invent individual shots.
- Ignore the first shot completely.
- Base feedback ONLY on clear, observable patterns from the footage.
- If the clip is too short or unclear, reply with:
  "Not enough reliable data in this clip to give feedback."
- Skip categories you cannot confirm. Never speculate.
- You cannot judge depth (distance to baseline). Only analyze lateral (x-axis) movement.

What to Analyze
1) Strategic Patterns and Shot Selections
- What overall shot choices or rally patterns did the player lean on?
- Were they dictating play, reacting defensively, or staying neutral?
- Did these patterns help or hurt them?

2) Court Positioning and Movement
- Where did they usually stand (behind baseline, hugging baseline, moving forward)?
- Did they recover to an effective court position after shots?
- Was footwork efficient or costing them opportunities?

3) Strategic Effectiveness
- Did their choices open up the court or surrender control?
- Did they make visible tactical adjustments during the rally?

4) Personalized Coaching Feedback (most important)
- Speak directly to the player (“You should…”, “Next time, focus on…”).
- Connect advice to what you saw.
- Suggest concrete drills or habits that fit the observed issues.
- Highlight one or two strengths, but focus mainly on improvement.
"""

        model_coach = genai_client.GenerativeModel("gemini-3-pro-preview")
        response_coach = model_coach.generate_content(
            contents=[
                {"parts": [
                    {"file_data": {"file_uri": video_file_obj.uri, "mime_type": video_file_obj.mime_type}},
                    {"text": coach_prompt},
                ]}
            ],
            generation_config={"temperature": 0.2},
        )

        if not getattr(response_coach, "parts", None) or response_coach.candidates[0].finish_reason != 1:
            st.error("The AI analysis was blocked. Try another video segment.")
            st.stop()

        coach_text = response_coach.text.strip()

    st.subheader("Coach's Analysis")
    st.markdown(coach_text)

    # ----------------------------
    # 3) Key Shot Identification (Gemini 3) - Video native
    # ----------------------------
    with st.spinner("Identifying key shot..."):
        key_shot_prompt = """
You are a tennis coach using computer vision to analyze a single tennis point.

Your task is to identify and clearly describe the single most pivotal groundstroke (not a serve) in this point.
This must be a shot hit by the player closest to the camera only. Do not include or describe any shots from the opponent.

Constraints:
- Pick ONE key groundstroke only.
- Do NOT describe the full rally.
- Be precise and unambiguous so another model can locate this moment.
- If the clip is too short or unclear to pick a pivotal groundstroke, respond exactly:
  "Not enough reliable data in this clip to identify a key shot."

Format your response exactly like this (no extra text):

Player 1: [where the player closest to the camera was when hitting the key shot]
Key Shot: [what shot they hit — brief but specific; do NOT include any timestamps here]
Impact: [why this shot was pivotal]
Key Shot Timestamp: [approx time in seconds from the start of the video, like "5.4s"]
"""

        model_keyshot = genai_client.GenerativeModel("gemini-3-pro-preview")
        response_keyshot = model_keyshot.generate_content(
            contents=[
                {"parts": [
                    {"file_data": {"file_uri": video_file_obj.uri, "mime_type": video_file_obj.mime_type}},
                    {"text": key_shot_prompt},
                ]}
            ],
            generation_config={"temperature": 0.25},
        )

        if not getattr(response_keyshot, "parts", None) or response_keyshot.candidates[0].finish_reason != 1:
            st.error("The AI key-shot identification was blocked. Try another video segment.")
            st.stop()

        key_shot_text_full = response_keyshot.text.strip()

    st.subheader("Key Shot")
    st.markdown(key_shot_text_full)

    if key_shot_text_full.strip() == "Not enough reliable data in this clip to identify a key shot.":
        st.info("Try a slightly longer clip that includes a clear rally and a decisive moment.")
        st.stop()

    key_shot_timestamp_str, key_shot_text_for_gemini = parse_keyshot_timestamp(key_shot_text_full)
    if key_shot_timestamp_str:
        st.info(f"Key shot identified around: {key_shot_timestamp_str}")
    else:
        st.warning("Could not find a timestamp line. Proceeding without timestamp guidance.")

    # ----------------------------
    # 4) Extract Positions and Trajectories (Gemini 3 JSON)
    # ----------------------------
    if analysis_successful is False:
        with st.spinner("Extracting positions and trajectories..."):
            json_prompt = f"""
You are a tennis analysis system. Using both the video and the key shot description below, identify the moment of the key shot and extract the player positions and shot trajectory coordinates at that moment.

Key shot identification:
({key_shot_text_for_gemini})
"""
            if key_shot_timestamp_str:
                json_prompt += f'\nThe key shot occurs at approximately "{key_shot_timestamp_str}" into the video. Use this to pinpoint the exact moment.\n'

            json_prompt += """
Court Coordinate System for Estimation:
- Origin (0,0) is the bottom-left corner of the court from the camera's perspective.
- The court is 36 feet wide (x-axis, from 0 to 36) and 78 feet long (y-axis, from 0 to 78).
- Key y-axis landmarks:
  - Near baseline: y=0
  - Near service line: y=21
  - Net: y=39
  - Far service line: y=57
  - Far baseline: y=78
- Player 1 (who hit the key shot) is closer to the camera (lower y-value).
- Player 2 is on the far side of the court (higher y-value).
- Use court lines as the primary reference for estimating coordinates.

Return a JSON object with this exact structure:
{
  "player1_pos": { "x": float, "y": float },
  "player2_pos": { "x": float, "y": float },
  "actual_shot_start": { "x": float, "y": float },
  "actual_shot_end": { "x": float, "y": float },
  "suggested_shot_end": { "x": float, "y": float },
  "analysis": "A brief, conversational analysis explaining why the suggested shot is tactically better."
}

Rules:
- suggested_shot_start is assumed to be the same as actual_shot_start.
- actual_shot_start should be very close to player1_pos.
- Respond with ONLY the JSON object (no markdown, no backticks).
"""

            model_json = genai_client.GenerativeModel("gemini-3-pro-preview")
            json_response = model_json.generate_content(
                contents=[
                    {"parts": [
                        {"file_data": {"file_uri": video_file_obj.uri, "mime_type": video_file_obj.mime_type}},
                        {"text": json_prompt},
                    ]}
                ],
                generation_config={"temperature": 0.06},
            )

            response_text = (json_response.text or "").strip()
            response_text = response_text.replace("```json", "").replace("```", "").strip()

            try:
                viz_data = json.loads(response_text)
            except Exception as e:
                st.error(f"Failed to parse JSON from the AI response: {e}")
                st.text("Model Response:")
                st.code(json_response.text, language="text")
                st.stop()

            player1_pos = validate_coordinates(viz_data.get("player1_pos"), "Player 1")
            player2_pos = validate_coordinates(viz_data.get("player2_pos"), "Player 2")
            actual_shot_start = validate_coordinates(viz_data.get("actual_shot_start"), "Actual shot start")
            actual_shot_end = validate_coordinates(viz_data.get("actual_shot_end"), "Actual shot end")
            suggested_shot_end = validate_coordinates(viz_data.get("suggested_shot_end"), "Suggested shot end")
            shot_analysis_text = viz_data.get("analysis", "No analysis provided.")

            player1_pos, player2_pos = validate_and_fix_player_positions(player1_pos, player2_pos)

            if player1_pos and actual_shot_start:
                actual_shot_start = correct_shot_start_position(actual_shot_start, player1_pos)

            suggested_shot_start = actual_shot_start.copy() if actual_shot_start else None

            if not all([player1_pos, player2_pos, actual_shot_start, actual_shot_end, suggested_shot_start, suggested_shot_end]):
                st.error(
                    "Failed to extract all necessary data points for visualization. "
                    "Please try a different video segment."
                )
                st.stop()

            st.success("Successfully extracted visualization data.")

        # ----------------------------
        # Visualization
        # ----------------------------
        st.subheader("Shot Recommendation Analysis")
        st.markdown(shot_analysis_text)

        with st.spinner("Generating visualization..."):
            try:
                fig, ax = create_tennis_court()

                ax.plot(player1_pos[0], player1_pos[1], "ro", markersize=10, label="Player 1")
                ax.plot(player2_pos[0], player2_pos[1], "bo", markersize=10, label="Player 2")

                # Actual shot
                dx_actual = actual_shot_end[0] - actual_shot_start[0]
                dy_actual = actual_shot_end[1] - actual_shot_start[1]
                ax.arrow(
                    actual_shot_start[0],
                    actual_shot_start[1],
                    dx_actual * 0.95,
                    dy_actual * 0.95,
                    color="red",
                    linestyle="--",
                    width=0.3,
                    head_width=2,
                    head_length=2,
                    length_includes_head=True,
                    label="Actual Shot",
                )

                # Suggested shot
                dx_suggested = suggested_shot_end[0] - suggested_shot_start[0]
                dy_suggested = suggested_shot_end[1] - suggested_shot_start[1]
                ax.arrow(
                    suggested_shot_start[0],
                    suggested_shot_start[1],
                    dx_suggested * 0.95,
                    dy_suggested * 0.95,
                    color="green",
                    width=0.3,
                    head_width=2,
                    head_length=2,
                    length_includes_head=True,
                    label="Suggested Shot",
                )

                ax.legend(loc="best")
                ax.set_title("Shot Visualization - Key Shot Analysis", color="black", fontsize=16)
                plt.tight_layout()

                st.subheader("Shot Visualization")
                st.pyplot(fig)

                analysis_successful = True

            except Exception as e:
                st.error(f"An error occurred while creating the visualization: {e}")
                st.info("The analysis above is still available even though the visualization failed.")
                analysis_successful = True

except Exception as e:
    st.error(f"An error occurred during the analysis process: {e}")
    st.info("Please try uploading a different video segment.")
    st.exception(e)

finally:
    if "video_file_path" in locals() and os.path.exists(video_file_path):
        try:
            os.remove(video_file_path)
        except Exception as e:
            st.warning(f"Could not delete temporary file {video_file_path}: {e}")
