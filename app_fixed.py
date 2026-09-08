"""
YouTube Brand Monitor - Streamlit App
Monitors YouTube videos for brand logos and keywords INSIDE video content.
Free. Local. No cloud APIs except YouTube Data API.
"""

import streamlit as st
import os
import json
import tempfile
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
import whisper
import imageio_ffmpeg


# ==================== CONFIG ====================
RATE_LIMIT_FILE = ".yt_brand_monitor_state.json"
MAX_SEARCHES_PER_DAY = 2
MAX_LOGOS = 3
MAX_KEYWORDS = 5


# ==================== RATE LIMITER ====================

def get_rate_limit_state():
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(RATE_LIMIT_FILE):
        with open(RATE_LIMIT_FILE, "r") as f:
            state = json.load(f)
        if state.get("date") == today:
            return state
    return {"date": today, "count": 0}


def can_search():
    return get_rate_limit_state()["count"] < MAX_SEARCHES_PER_DAY


def increment_search():
    state = get_rate_limit_state()
    state["count"] = state.get("count", 0) + 1
    with open(RATE_LIMIT_FILE, "w") as f:
        json.dump(state, f)


def searches_remaining():
    return MAX_SEARCHES_PER_DAY - get_rate_limit_state().get("count", 0)


# ==================== YOUTUBE SEARCH ====================

def search_youtube(api_key, keywords, max_per_keyword=50):
    """Search YouTube for videos published in last 2 days."""
    youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    published_after = (datetime.utcnow() - timedelta(days=2)).isoformat("T") + "Z"
    
    all_videos = []
    seen_ids = set()
    
    for keyword in keywords:
        try:
            request = youtube.search().list(
                part="snippet",
                q=keyword,
                type="video",
                publishedAfter=published_after,
                maxResults=max_per_keyword,
                order="date",
            )
            response = request.execute()
            
            for item in response.get("items", []):
                vid = item["id"]["videoId"]
                if vid not in seen_ids:
                    seen_ids.add(vid)
                    thumb = item["snippet"]["thumbnails"]
                    all_videos.append({
                        "video_id": vid,
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "published_at": item["snippet"]["publishedAt"],
                        "channel": item["snippet"]["channelTitle"],
                        "url": f"https://youtube.com/watch?v={vid}",
                        "thumbnail": thumb["high"]["url"] if "high" in thumb else 
                                    (thumb["medium"]["url"] if "medium" in thumb else ""),
                    })
        except Exception as e:
            st.error(f"API Error for keyword '{keyword}': {e}")
    
    return all_videos


# ==================== VIDEO DOWNLOAD ====================

def download_video(video_id, output_dir):
    """Download video using yt-dlp. Returns path or None."""
    url = f"https://youtube.com/watch?v={video_id}"
    output_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    cmd = [
        sys.executable,
        "-m", "yt_dlp",
        "-f", "worst[ext=mp4]",
        "--no-playlist",
        "--quiet",
        "-o", output_path,
        url,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
            return output_path
    except Exception:
        pass
    return None


# ==================== FRAME EXTRACTION ====================

def extract_sample_frames(video_path, output_dir, num_frames=12):
    """Extract evenly spaced frames from video."""
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    
    indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    frame_paths = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            path = os.path.join(frames_dir, f"frame_{int(idx)}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
    
    cap.release()
    return frame_paths


# ==================== LOGO DETECTION ====================

def load_logo(path):
    """Load and preprocess logo image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    max_dim = 300
    h, w = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def detect_logo_orb(frame_gray, logo_gray, min_matches=12):
    """Detect logo using ORB feature matching."""
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(logo_gray, None)
    kp2, des2 = orb.detectAndCompute(frame_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return False, 0
    
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        return len(good) >= min_matches, len(good)
    except Exception:
        return False, 0


def detect_logo_template(frame_gray, logo_gray, threshold=0.6):
    """Detect logo using multi-scale template matching."""
    for scale in [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]:
        resized = cv2.resize(logo_gray, (0, 0), fx=scale, fy=scale)
        h, w = resized.shape
        if h > frame_gray.shape[0] or w > frame_gray.shape[1]:
            continue
        res = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > threshold:
            return True, max_val
    return False, 0


def check_logo_in_frames(frame_paths, logo_paths):
    """Check if any logo appears in any frame."""
    logos = [load_logo(p) for p in logo_paths]
    logos = [l for l in logos if l is not None]
    if not logos:
        return False, []
    
    detected = []
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        
        for i, logo in enumerate(logos):
            found, score = detect_logo_orb(frame, logo)
            if found:
                detected.append(f"Logo-{i+1}(ORB:{score})")
                continue
            found, score = detect_logo_template(frame, logo)
            if found:
                detected.append(f"Logo-{i+1}(TM:{score:.2f})")
    
    return len(detected) > 0, list(set(detected))


# ==================== AUDIO & KEYWORDS ====================

def extract_audio(video_path, output_path):
    """Extract audio from video using FFmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load Whisper model once and cache."""
    return whisper.load_model("base")


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    try:
        model = load_whisper_model()
        result = model.transcribe(audio_path, fp16=False, verbose=False)
        return result["text"].lower()
    except Exception as e:
        st.warning(f"Transcription error: {e}")
        return ""


def find_keywords_in_text(text, keywords):
    """Find which keywords appear in text."""
    found = []
    text = text.lower()
    for kw in keywords:
        if kw.lower() in text:
            found.append(kw)
    return found


# ==================== VIDEO ANALYSIS ====================

def analyze_video(video_info, logo_paths, keywords, work_dir):
    """Full pipeline: download → frames → logo → audio → keywords."""
    vid = video_info["video_id"]
    result = {
        **video_info,
        "logo_detected": False,
        "logo_details": "",
        "keywords_in_audio": [],
        "keywords_found": "",
        "status": "Pending",
    }
    
    # Download
    video_path = download_video(vid, work_dir)
    if not video_path:
        result["status"] = "Download Failed"
        return result
    
    # Frames
    video_subdir = os.path.join(work_dir, vid)
    frame_paths = extract_sample_frames(video_path, video_subdir, num_frames=12)
    
    # Logo check
    if logo_paths:
        found, details = check_logo_in_frames(frame_paths, logo_paths)
        result["logo_detected"] = found
        result["logo_details"] = "; ".join(details) if details else "Not found"
    else:
        result["logo_details"] = "No logos uploaded"
    
    # Audio
    audio_path = os.path.join(work_dir, f"{vid}.wav")
    if extract_audio(video_path, audio_path):
        transcript = transcribe_audio(audio_path)
        found_kws = find_keywords_in_text(transcript, keywords)
        result["keywords_in_audio"] = found_kws
        result["keywords_found"] = ", ".join(found_kws) if found_kws else "None"
        if os.path.exists(audio_path):
            os.remove(audio_path)
    else:
        result["keywords_found"] = "Audio failed"
    
    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(video_subdir):
        shutil.rmtree(video_subdir, ignore_errors=True)
    
    result["status"] = "Analyzed"
    return result


# ==================== STREAMLIT UI ====================

def main():
    st.set_page_config(page_title="YouTube Brand Monitor", layout="wide")
    st.title("🔍 YouTube Brand Monitor")
    st.caption("Finds your brand **inside** videos — not just titles. Free. Local. Last 2 days only.")
    
    # Sidebar
    st.sidebar.header("⚡ Daily Quota")
    remaining = searches_remaining()
    st.sidebar.info(f"Searches remaining today: **{remaining} / {MAX_SEARCHES_PER_DAY}**")
    if remaining <= 0:
        st.sidebar.error("🚫 Limit reached. Try tomorrow.")
    
    st.sidebar.header("🔑 YouTube API Key")
    api_key = st.sidebar.text_input("API Key", type="password",
                                    help="Google Cloud Console → YouTube Data API v3")
    
    st.sidebar.header("🖼️ Brand Logos")
    st.sidebar.caption(f"Max {MAX_LOGOS} images (PNG/JPG)")
    logo_files = st.sidebar.file_uploader("Upload logos", type=["png", "jpg", "jpeg"],
                                          accept_multiple_files=True)
    if logo_files and len(logo_files) > MAX_LOGOS:
        st.sidebar.error(f"Only first {MAX_LOGOS} logos will be used.")
        logo_files = logo_files[:MAX_LOGOS]
    
    st.sidebar.header("📝 Keywords")
    st.sidebar.caption(f"Max {MAX_KEYWORDS} (one per line)")
    kw_input = st.sidebar.text_area("Keywords", height=100,
                                    placeholder="SBI\nState Bank of India\nSBI scam\nएसबीआई")
    keywords = [k.strip() for k in kw_input.split("\n") if k.strip()]
    if len(keywords) > MAX_KEYWORDS:
        st.sidebar.error(f"Only first {MAX_KEYWORDS} keywords will be used.")
        keywords = keywords[:MAX_KEYWORDS]
    
    # Main metrics
    st.header("Search Parameters")
    c1, c2, c3 = st.columns(3)
    c1.metric("Logos", len(logo_files) if logo_files else 0)
    c2.metric("Keywords", len(keywords))
    c3.metric("Window", "Last 48 Hours")
    
    # Search button
    can_click = can_search() and bool(api_key) and len(keywords) > 0
    if st.button("🚀 Search Now", disabled=not can_click, type="primary"):
        if not api_key:
            st.error("Enter your YouTube API key.")
            return
        if len(keywords) == 0:
            st.error("Enter at least 1 keyword.")
            return
        
        # Prep workspace
        work_dir = tempfile.mkdtemp(prefix="yt_monitor_")
        logo_paths = []
        if logo_files:
            for i, lf in enumerate(logo_files):
                path = os.path.join(work_dir, f"logo_{i}.png")
                with open(path, "wb") as f:
                    f.write(lf.getvalue())
                logo_paths.append(path)
        
        try:
            increment_search()
            
            # Step 1: Search
            with st.spinner(f"Searching YouTube for {len(keywords)} keywords (last 2 days)..."):
                videos = search_youtube(api_key, keywords)
            st.success(f"Found {len(videos)} unique videos. Analyzing content...")
            
            if not videos:
                st.info("No videos found. Try broader keywords.")
                return
            
            # Step 2: Analyze each video
            results = []
            prog = st.progress(0)
            status = st.empty()
            
            for i, video in enumerate(videos):
                status.text(f"[{i+1}/{len(videos)}] {video['title'][:55]}...")
                res = analyze_video(video, logo_paths, keywords, work_dir)
                results.append(res)
                prog.progress((i + 1) / len(videos))
            
            prog.empty()
            status.empty()
            
            # Step 3: Filter flagged
            flagged = [r for r in results if r["logo_detected"] or len(r["keywords_in_audio"]) > 0]
            
            st.header(f"🎯 Flagged Videos: {len(flagged)}")
            
            if flagged:
                df = pd.DataFrame(flagged)
                display_cols = ["title", "channel", "published_at", "logo_detected",
                               "logo_details", "keywords_found", "url"]
                display_df = df[[c for c in display_cols if c in df.columns]].copy()
                display_df.rename(columns={
                    "title": "Title", "channel": "Channel", "published_at": "Published",
                    "logo_detected": "Logo Found", "logo_details": "Logo Details",
                    "keywords_found": "Audio Keywords", "url": "URL"
                }, inplace=True)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Excel export
                xls_path = os.path.join(work_dir, "results.xlsx")
                display_df.to_excel(xls_path, index=False)
                with open(xls_path, "rb") as f:
                    st.download_button(
                        "📥 Download Excel",
                        data=f,
                        file_name=f"brand_monitor_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Quick links view
                st.subheader("Video Links")
                for r in flagged:
                    cols = st.columns([1, 6])
                    with cols[0]:
                        if r.get("thumbnail"):
                            st.image(r["thumbnail"], width=100)
                    with cols[1]:
                        badges = []
                        if r["logo_detected"]:
                            badges.append("🖼️ Logo")
                        if r["keywords_in_audio"]:
                            badges.append("🔊 Audio")
                        st.markdown(f"**[{r['title']}]({r['url']})**  \n"
                                   f"`{r['channel']}` | {' | '.join(badges)}")
            else:
                st.info("No videos contained your logos or keywords inside the actual video.")
                with st.expander("See all analyzed videos"):
                    df_all = pd.DataFrame(results)
                    st.dataframe(df_all[["title", "channel", "logo_detected", 
                                        "keywords_found", "url"]], hide_index=True)
        
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()