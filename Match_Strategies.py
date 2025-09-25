import streamlit as st
from google.api_core import client_options as client_options_lib
import google.generativeai as genai
import anthropic
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
import tempfile
import re
import numpy as np
import json
import cv2
import base64

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]


client = OpenAI()

def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    return base64Frames, fps

st.set_page_config(layout="wide", page_title="Tennis Strategy Analyzer")
st.title("Tennis Point Strategy Analyzer")

# --- Enhanced Helper Functions with Better Validation ---
def validate_coordinates(coords, coord_type="position"):
    if coords is None:
        return None
    if isinstance(coords, dict):
        coords = [coords.get('x'), coords.get('y')]
    if coords[0] is None or coords[1] is None:
        return None
    x, y = float(coords[0]), float(coords[1])
    x_min_viz, x_max_viz = -10, 50
    y_min_viz, y_max_viz = -10, 90
    if "end" in coord_type.lower():
        court_width = 36
        court_length = 78
        if not (0 <= x <= court_width and 0 <= y <= court_length):
            st.info(f"ℹ️ {coord_type} at [{x:.1f}, {y:.1f}] is outside the court boundaries.")
    if x < x_min_viz or x > x_max_viz or y < y_min_viz or y > y_max_viz:
        st.warning(f"⚠️ {coord_type} coordinates [{x:.1f}, {y:.1f}] are far outside the court. The visualization may be skewed, but proceeding.")
    return [x, y]

def validate_and_fix_player_positions(player1_pos, player2_pos):
    if player1_pos is None or player2_pos is None:
        return player1_pos, player2_pos
    if player1_pos[0] == player2_pos[0] and player1_pos[1] == player2_pos[1]:
        st.error("❌ Both players detected at identical positions. This likely indicates an extraction failure.")
        return None, None
    if player1_pos[1] > player2_pos[1] and player1_pos[1] > 45 and player2_pos[1] < 30:
        st.warning("⚠️ Player positions appear swapped based on y-coordinates. Correcting...")
        return player2_pos, player1_pos
    return player1_pos, player2_pos

def correct_shot_start_position(shot_start, player_pos):
    if None in [shot_start, player_pos]:
        return shot_start
    distance = np.sqrt((shot_start[0] - player_pos[0])**2 + (shot_start[1] - player_pos[1])**2)
    if distance > 8.0:
        st.info(f"📍 Correcting shot start position from [{shot_start[0]:.1f}, {shot_start[1]:.1f}] to player position [{player_pos[0]:.1f}, {player_pos[1]:.1f}]")
        return player_pos.copy()
    return shot_start

def create_tennis_court():
    fig_plot, ax_plot = plt.subplots(figsize=(5, 10))
    ax_plot.set_facecolor('#88b2b9')
    court_width = 36
    court_length = 78
    court = patches.Rectangle((0, 0), court_width, court_length, linewidth=2, edgecolor='white', facecolor='#3a9e5c')
    ax_plot.add_patch(court)
    plt.plot([0, court_width], [0, 0], 'white', linewidth=2)
    plt.plot([0, court_width], [court_length, court_length], 'white', linewidth=2)
    singles_width = 27
    margin = (court_width - singles_width) / 2
    plt.plot([margin, margin], [0, court_length], 'white', linewidth=2)
    plt.plot([court_width - margin, court_width - margin], [0, court_length], 'white', linewidth=2)
    service_line_dist = 21
    plt.plot([margin, court_width - margin], [service_line_dist, service_line_dist], 'white', linewidth=2)
    plt.plot([margin, court_width - margin], [court_length - service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
    plt.plot([court_width / 2, court_width / 2], [service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
    center_mark_width = 0.5
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [0, 0], 'white', linewidth=2)
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [court_length, court_length], 'white', linewidth=2)
    plt.plot([0, court_width], [court_length / 2, court_length / 2], 'white', linewidth=3)
    plt.xlim(-5, court_width + 5)
    plt.ylim(-5, court_length + 5)
    ax_plot.set_xticks([])
    ax_plot.set_yticks([])
    plt.axis('equal')
    return fig_plot, ax_plot

@st.cache_resource
def get_genai_client():
    try:
        genai.configure(
            api_key=GOOGLE_API_KEY,
            transport="rest",
            client_options=client_options_lib.ClientOptions(
                api_endpoint=os.getenv("GOOGLE_API_ENDPOINT"),
            ),
        )
        return genai
    except Exception as e:
        st.error(f"Failed to initialize Google GenAI Client: {e}")
        return None

genai_client = get_genai_client()

uploaded_file = st.file_uploader("Upload a Tennis Point Video (MP4)", type=["mp4", "mov", "avi", "webm", "mkv"])

if uploaded_file is not None and genai_client:
    st.video(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_file_path = tmp_file.name
        if uploaded_file.size > 52 * 1024 * 1024:
            st.warning("This video is too large to process. Please upload a smaller video.")
            st.stop()
        elif uploaded_file.size > 150 * 1024 * 1024:
            st.warning("Large video detected. Processing may take 2-3x longer.")
    st.info("Video uploaded. Analyzing... This may take a few moments.")
    analysis_successful = False

    try:
        # --- 1. Upload file to Google AI ---
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

        # --- 2. Generate Coaching Analysis (Gemini) ---
        with st.spinner("Generating coaching analysis..."):
            coach_prompt = """
You are an elite tennis strategy coach with computer vision capabilities. 
Analyze ONLY the player CLOSEST TO THE CAMERA. 
Speak directly to them as if you were their coach between games. 

🚫 STRICT RULES:
- Do NOT describe or invent individual shots.
- Ignore the first shot completely.
- Base feedback ONLY on clear, observable patterns from the footage.
- If the clip is too short or unclear, reply with: 
  "Not enough reliable data in this clip to give feedback."
- Skip categories you cannot confirm. Never speculate.

---

### What to Analyze

**Strategic Patterns and Shot Selections**
- What overall shot choices or rally patterns did the player lean on? 
- Were they dictating play, reacting defensively, or staying neutral? 
- Did these patterns help or hurt them?

**Court Positioning and Movement**
- Where did they usually stand (deep behind baseline, hugging baseline, moving forward)? 
- Did they recover to an effective court position after shots?
- Was footwork efficient or costing them opportunities? 

**Strategic Effectiveness**
- Did their choices open up the court or surrender control?
- Did they make visible tactical adjustments during the rally? 

**Personalized Coaching Feedback (MOST IMPORTANT)**
- Speak directly to the player (“You should…”, “Next time, focus on…”).
- Connect your advice to what you saw (e.g., “You often stayed two steps behind the baseline even when you had a short ball — that let your opponent reset. Drill: practice taking those balls early and redirecting crosscourt.”).
- Suggest concrete drills or habits that fit the observed issues.
- Highlight one or two strengths, but focus mainly on how to *improve*.

---

⚠️ IMPORTANT:  
Your response must read like a **real coach giving personalized feedback** — clear, specific, actionable. 
No generic advice like “hit deeper” or “move your feet” unless it’s tied to a clear observed pattern.
"""

            try:
                model_coach = genai_client.GenerativeModel('gemini-2.5-pro') 
                response_coach = model_coach.generate_content(
                    contents=[
                        {"parts": [
                            {"file_data": {"file_uri": video_file_obj.uri, "mime_type": video_file_obj.mime_type}},
                            {"text": coach_prompt}
                        ]}
                    ],
                    generation_config={"temperature": 0.2}
                )
                if not response_coach.parts or response_coach.candidates[0].finish_reason != 1:
                    st.error("⚠️ The AI analysis was blocked. Try another video segment.")
                    st.stop()
                coach_text = response_coach.text
            except Exception as e:
                st.error(f"⚠️ Error generating coaching analysis: {str(e)}")
                st.stop()
        st.subheader("Coach's Analysis")
        st.markdown(coach_text)

        # --- 3. Identify Key Shot with OpenAI ---
        base64Frames, fps = process_video(video_file_path)
        def key_shot_identifyer(base64Frames, coach_text, fps):
            frame_sampling_rate = 40
            key_shot_prompt = [
                {
                    "role": "user",
                    "content": [
                         f"""
You are a tennis coach using computer vision to analyze a single tennis point. You are given a sequence of frames from a video, sampled every {frame_sampling_rate} frames. Each frame is approximately {frame_sampling_rate/fps:.2f} seconds apart.

Your task is to identify and clearly describe the single most pivotal groundstroke (not a serve) in this point. This must be a shot hit by the player closest to the camera only. Do not include or describe any shots from the opponent on the far side of the court.

You have access to the video footage.

Your response should focus only on the pivotal shot. Do not describe the full rally. This description will be used by a downstream computer vision model to locate and analyze that specific shot — so be precise, concise, and unambiguous.

A pivotal shot could be:

The shot that directly led to winning or losing the point (e.g., a winner or a final error).

A shot that significantly shifted the advantage (e.g., a great passing shot, a deep approach shot, or a poor shot that gave the opponent control).

A key tactical decision, like changing the direction of the rally or hitting an unexpected shot.

A missed opportunity where a different shot choice could have won the point.

BTW, here is some shot definitions if you need a refresher:
Shot Definitions:

Passing shot: Shot that goes past a net-rushing opponent and is out of their reach.

Approach shot: Aggressive shot hit while moving forward toward the net.

Winner: Any shot that lands in and the opponent cannot touch or return.

Volley: Shot hit before the ball bounces, usually taken at the net.

Half volley: Shot hit right after the bounce, often low and hard to control.

Overhead: Smash shot hit above the head to put away high balls, like a serve in motion.

Lob: High shot over an opponent's head, usually when they are at the net.

Moonball: Very high, slow topspin shot used to reset a rally or push back the opponent.

Format your response exactly like this (no extra explanation or context):

Player 1: [where the player closest to the camera was when hitting the key shot]
Key Shot: [what shot the player closest to the camera hit — be specific but brief; do NOT include any timestamps]
Impact: [why this shot was pivotal and what it reveals about the player's strategy or execution]
Key Shot Timestamp: [Provide the approximate time in seconds from the start of the video when the key shot is executed. For example: "5.4s"]

BTW: Make this description clear and concise so that another AI computer vision model can easily spot this exact moment.


""",
                        *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::40]),
                    ],
                },
            ]
            params = {
                "model": "gpt-5",
                "messages": key_shot_prompt,
                "max_completion_tokens": 2000,
            }
            results = client.chat.completions.create(**params)
            return results.choices[0].message["content"]
        with st.spinner("Identifying key shot (OpenAI)..."):
            key_shot_text_full = key_shot_identifyer(base64Frames, coach_text, fps)
            st.subheader("Key Shot")
            st.markdown(key_shot_text_full)
            
            # Debug raw output
            st.text("Raw Key Shot Response:")
            st.code(key_shot_text_full, language="markdown")
            
            key_shot_timestamp_str = None
            key_shot_text_for_gemini = key_shot_text_full
            
            try:
                # More flexible regex
                timestamp_match = re.search(r"(\d+(?:\.\d+)?)\s*(s|sec|seconds)?", key_shot_text_full.lower())
                if timestamp_match:
                    key_shot_timestamp_str = timestamp_match.group(1) + "s"
                    # remove the line regardless of casing
                    key_shot_text_for_gemini = re.sub(r"key shot timestamp.*", "", key_shot_text_full, flags=re.I)
                    st.info(f"Key shot identified around: {key_shot_timestamp_str}")
                else:
                    st.warning("Could not find timestamp for the key shot in the response.")
            except Exception as e:
                st.warning(f"Error parsing key shot response: {e}")




        # --- 4. Extract Positions and Visualize if Key Shot Found ---
        
        if(analysis_successful ==False):
            with st.spinner("Extracting positions and trajectories..."):
                json_prompt = f"""
You are a tennis analysis system. Using both the video and the key shot description below, identify the moment of the key shot and extract the player positions and shot trajectory coordinates at that moment.
                
Key shot identication:
({key_shot_text_for_gemini})
"""
                if key_shot_timestamp_str:
                    json_prompt += f"\nThe key shot occurs at approximately **{key_shot_timestamp_str}** into the video. Use this to pinpoint the exact moment.\n"

                json_prompt += """
Court Coordinate System for Estimation:
- Origin (0,0) is the bottom-left corner of the court from the camera's perspective.
- The court is 36 feet wide (x-axis, from 0 to 36) and 78 feet long (y-axis, from 0 to 78).
- **Key y-axis landmarks**:
    - Near baseline: y=0
    - Near service line: y=21
    - Net: y=39
    - Far service line: y=57
    - Far baseline: y=78
- Player 1 (who hit the key shot) is closer to the camera (lower y-value).
- Player 2 is on the far side of the court (higher y-value).
- **CRITICAL**: Use the court lines (baselines, service lines, etc.) as your primary reference for estimating coordinates. Estimate a player's y-position based on where they are standing relative to these fixed lines. For instance, a player standing halfway between the near baseline (y=0) and the near service line (y=21) should have a y-coordinate of approximately 10.5.

Your task is to provide a JSON object containing the coordinates for the key moment. Provide your best estimate.

The JSON object must have this exact structure:
{{
  "player1_pos": {{ "x": float, "y": float }},
  "player2_pos": {{ "x": float, "y": float }},
  "actual_shot_start": {{ "x": float, "y": float }},
  "actual_shot_end": {{ "x": float, "y": float }},
  "suggested_shot_end": {{ "x": float, "y": float }},
  "analysis": "A brief, conversational analysis explaining why the suggested shot is tactically better."
}}

RULES:
- The `suggested_shot_start` will be assumed to be the same as `actual_shot_start`.
- The `actual_shot_start` should be very close to `player1_pos`.
- Respond with ONLY the JSON object. Do not include markdown backticks (` ```json `) or any other text.

This is the template of the court so visualize this before you find the JSOn coordinates:
"    fig_plot, ax_plot = plt.subplots(figsize=(5, 10))
    ax_plot.set_facecolor('#88b2b9')
    court_width = 36
    court_length = 78
    court = patches.Rectangle((0, 0), court_width, court_length, linewidth=2, edgecolor='white', facecolor='#3a9e5c')
    ax_plot.add_patch(court)
    plt.plot([0, court_width], [0, 0], 'white', linewidth=2)
    plt.plot([0, court_width], [court_length, court_length], 'white', linewidth=2)
    singles_width = 27
    margin = (court_width - singles_width) / 2
    plt.plot([margin, margin], [0, court_length], 'white', linewidth=2)
    plt.plot([court_width - margin, court_width - margin], [0, court_length], 'white', linewidth=2)
    service_line_dist = 21
    plt.plot([margin, court_width - margin], [service_line_dist, service_line_dist], 'white', linewidth=2)
    plt.plot([margin, court_width - margin], [court_length - service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
    plt.plot([court_width / 2, court_width / 2], [service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
    center_mark_width = 0.5
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [0, 0], 'white', linewidth=2)
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [court_length, court_length], 'white', linewidth=2)
    plt.plot([0, court_width], [court_length / 2, court_length / 2], 'white', linewidth=3)
    plt.xlim(-5, court_width + 5)
    plt.ylim(-5, court_length + 5)
    ax_plot.set_xticks([])
    ax_plot.set_yticks([])
    plt.axis('equal')
    return fig_plot, ax_plot"

"""
                model_json = genai_client.GenerativeModel('gemini-2.5-pro')
                json_response = model_json.generate_content(
                    contents=[
                        {"parts": [
                            {"file_data": {"file_uri": video_file_obj.uri, "mime_type": video_file_obj.mime_type}},
                            {"text": json_prompt}
                        ]}
                    ],
                    generation_config={"temperature": 0.06}
                )
                try:
                    response_text = json_response.text.strip().replace('```json', '').replace('```', '')
                    viz_data = json.loads(response_text)
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
                    if not all([player1_pos, player2_pos, actual_shot_start, actual_shot_end, suggested_shot_end]):
                        st.error("❌ Failed to extract all necessary data points for visualization. The model may have returned incomplete data. Please try a different video.")
                        st.stop()
                    st.success("✅ Successfully extracted all visualization data.")
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    st.error(f"❌ Failed to parse visualization data from the AI. Error: {e}")
                    st.text("Model Response:")
                    st.code(json_response.text, language="text")
                    st.stop()
                if player1_pos and player2_pos and actual_shot_start and actual_shot_end and suggested_shot_start and suggested_shot_end:
                    st.subheader("Shot Recommendation Analysis")
                    st.markdown(shot_analysis_text)
                    with st.spinner("Generating visualization..."):
                        try:
                            fig, ax = create_tennis_court()
                            ax.plot(player1_pos[0], player1_pos[1], 'ro', markersize=10, label='Player 1')
                            ax.plot(player2_pos[0], player2_pos[1], 'bo', markersize=10, label='Player 2')
                            dx_actual = actual_shot_end[0] - actual_shot_start[0]
                            dy_actual = actual_shot_end[1] - actual_shot_start[1]
                            ax.arrow(actual_shot_start[0], actual_shot_start[1], 
                                dx_actual * 0.95, dy_actual * 0.95,
                                color='red', linestyle='--', width=0.3, head_width=2, 
                                head_length=2, length_includes_head=True, label='Actual Shot')
                            dx_suggested = suggested_shot_end[0] - suggested_shot_start[0]
                            dy_suggested = suggested_shot_end[1] - suggested_shot_start[1]
                            ax.arrow(suggested_shot_start[0], suggested_shot_start[1], 
                                dx_suggested * 0.95, dy_suggested * 0.95,
                                color='green', width=0.3, head_width=2, 
                                head_length=2, length_includes_head=True, label='Suggested Shot')
                            ax.legend(loc='best')
                            ax.set_title(f'Shot Visualization - Key Shot Analysis', color='black', fontsize=16)
                            plt.tight_layout()
                            st.subheader("Shot Visualization")
                            st.pyplot(fig)
                            analysis_successful = True
                        except Exception as e:
                            st.error(f"An error occurred while creating the visualization: {e}")
                            st.info("Even though the visualization failed, the data was extracted and the analysis is available above.")
                            analysis_successful = True
    except Exception as e:
        st.error(f"An error occurred during the analysis process: {e}")
        st.info("Please try uploading a different video ")
        st.exception(e)
    finally:
        if 'video_file_path' in locals() and os.path.exists(video_file_path):
            try:
                os.remove(video_file_path)
            except Exception as e:
                st.warning(f"Could not delete temporary file {video_file_path}: {e}")

elif uploaded_file is None:
    st.info("Please upload a tennis point video to start the analysis.")
    st.markdown("""
    ## How This Works
    1. **Upload a Tennis Point Video**: The app accepts MP4 videos showing a single tennis point.
    2. **AI Analysis**: The system will analyze:
       - Your overall game strategy and patterns
       - Court positioning and movement
       - Critical decision-making moments
    3. **Results**: You'll receive:
       - A personalized coaching analysis
       - Identification of a key shot that influenced the point (if any)
       - Visual representation of player positions and shot trajectories
       - Strategic recommendation with explanation
    **Smart Camera Detection** - Automatically detects camera angle and adapts analysis accordingly!
    This tool is designed to help players understand the strategic elements of their game and make better tactical decisions on court.
    """)

else:
    st.error("API clients could not be initialized. Please check API keys/configurations and restart.")
