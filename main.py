import os
import json
import random
import requests
import cv2
import numpy as np
import subprocess
import shutil
import logging
import traceback
import schedule
import time
import math
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from gtts import gTTS
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from git import Repo

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
load_dotenv()

# Environment Variables
FBI_API_URL = os.getenv("FBI_API_URL", "https://api.fbi.gov/wanted/v1/list")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Directory Structure
BASE_DIR = Path(__file__).resolve().parent
DIRS = {
    "images": BASE_DIR / "assets" / "images",
    "audio": BASE_DIR / "assets" / "audio",
    "music": BASE_DIR / "assets" / "music",
    "history": BASE_DIR / "history",
    "output": BASE_DIR / "output",
    "generated": BASE_DIR / "generated_videos",
    "logs": BASE_DIR / "logs"
}

HISTORY_FILE = DIRS["history"] / "used_cases.json"

def init_dirs():
    """Create necessary directories if they don't exist."""
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    
    bgm = DIRS["music"] / "background.mp3"
    if not bgm.exists():
        print(f"Warning: Please place a 'background.mp3' in {DIRS['music']}")

# Initialize Logging
init_dirs()
logging.basicConfig(
    filename=DIRS["logs"] / "run.log", 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================
# 2. HISTORY MANAGER
# ==========================================
def load_history():
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def is_used(case_id):
    history = load_history()
    return any(item.get("case_id") == case_id for item in history)

# ==========================================
# 3. DATA FETCHING (FBI API)
# ==========================================
def get_random_case():
    """Fetches a random unused case with at least 4 images."""
    page = random.randint(1, 40)
    response = requests.get(f"{FBI_API_URL}?page={page}")
    response.raise_for_status()
    items = response.json().get("items", [])
    
    random.shuffle(items)
    for item in items:
        if item.get("images") and len(item["images"]) >= 4 and not is_used(item["uid"]):
            return {
                "case_id": item["uid"],
                "title": item.get("title", "Unknown Subject"),
                "description": item.get("description", ""),
                "place": item.get("place_of_birth", "an unknown location"),
                "caution": item.get("caution", "Details are still unfolding."),
                "images": [img["original"] for img in item["images"]]
            }
    raise ValueError("No suitable case found on this page.")

# ==========================================
# 4. SCRIPT GENERATOR
# ==========================================
def clean_html(raw_html):
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html).replace('&nbsp;', ' ').strip()

def generate_script(case_data):
    """Generates an American True Crime documentary script."""
    title = case_data['title'].title()
    place = case_data.get('place') or "the United States"
    caution = clean_html(case_data.get('caution'))[:200]
    
    script = f"""
    This is a mystery that authorities are still trying to solve. 
    The case centers around {title}, originating near {place}. 
    Investigators discovered a disturbing timeline of events. 
    {caution}
    Despite extensive searches, the suspect remains unidentified or at large. 
    Authorities are asking the public for any information. 
    Do you know something that could close this case?
    """
    
    script = " ".join(script.split()) # Clean up whitespace
    script_path = BASE_DIR / "assets" / "script.txt"
    with open(script_path, "w") as f:
        f.write(script)
        
    return script

# ==========================================
# 5. IMAGE PROCESSOR
# ==========================================
def download_and_format_images(image_urls):
    """Downloads images and formats them to vertical 1080x1920."""
    formatted_paths = []
    target_w, target_h = 1080, 1920
    
    for i, url in enumerate(image_urls[:5]):
        resp = requests.get(url, stream=True)
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Blurred background fill
        bg = cv2.resize(img, (target_w, target_h))
        bg = cv2.GaussianBlur(bg, (99, 99), 30)
        
        # Foreground scale
        scale = min(target_w/w, target_h/h) * 0.9
        new_w, new_h = int(w * scale), int(h * scale)
        fg = cv2.resize(img, (new_w, new_h))
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = fg
        bg = cv2.convertScaleAbs(bg, alpha=1.1, beta=10) # Brightness adjustment
        
        out_path = DIRS["images"] / f"img0{i+1}.jpg"
        cv2.imwrite(str(out_path), bg)
        formatted_paths.append(str(out_path))
        
    return formatted_paths

# ==========================================
# 6. TTS GENERATOR
# ==========================================
def generate_voice(script_text):
    """Generates voiceover audio."""
    tts = gTTS(text=script_text, lang='en', tld='us', slow=False)
    out_path = DIRS["audio"] / "voice.wav"
    temp_mp3 = DIRS["audio"] / "temp.mp3"
    
    tts.save(str(temp_mp3))
    os.system(f"ffmpeg -y -i \"{temp_mp3}\" -ar 44100 \"{out_path}\" -loglevel error")
    os.remove(temp_mp3)
    
    return str(out_path)

# ==========================================
# 7. SUBTITLE GENERATOR
# ==========================================
def generate_srt(script_text, audio_duration):
    """Approximates word timings to generate SRT file."""
    words = script_text.split()
    total_words = len(words)
    time_per_word = audio_duration / total_words
    
    srt_path = BASE_DIR / "assets" / "subtitles.srt"
    chunk_size = 5
    
    with open(srt_path, "w") as f:
        sub_index = 1
        for i in range(0, total_words, chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            start_time = i * time_per_word
            end_time = min((i + chunk_size) * time_per_word, audio_duration)
            
            def format_time(seconds):
                ms = int((seconds % 1) * 1000)
                m, s = divmod(int(seconds), 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
            f.write(f"{sub_index}\n")
            f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            f.write(f"{chunk}\n\n")
            sub_index += 1
            
    return str(srt_path)

# ==========================================
# 8. VIDEO BUILDER
# ==========================================
def build_cinematic_video(image_paths, audio_path):
    """Creates video clips from images and concatenates them."""
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration
    clip_duration = audio_duration / len(image_paths)
    clips = []
    
    for i, img_path in enumerate(image_paths):
        clip = ImageClip(img_path).set_duration(clip_duration)
        if i % 2 == 0:
            clip = clip.resize(lambda t: 1 + 0.02 * t).set_position(('center', 'center'))
        else:
            clip = clip.resize(lambda t: 1.1 - 0.02 * t).set_position(('center', 'center'))
        clips.append(clip)
        
    video = concatenate_videoclips(clips, method="compose")
    video = video.set_audio(audio)
    
    temp_video = DIRS["output"] / "temp_video.mp4"
    video.write_videofile(str(temp_video), fps=30, codec="libx264", audio_codec="aac", logger=None)
    
    return str(temp_video), audio_duration

# ==========================================
# 9. MUSIC MIXER & FINAL RENDER
# ==========================================
def mix_final_video(video_path, srt_path, output_path):
    """Mixes background music, ducks volume, and burns subtitles."""
    bgm_path = DIRS["music"] / "background.mp3"
    
    # Escape path for FFmpeg subtitle filter (Windows paths need double escaping)
    escaped_srt = str(srt_path).replace('\\', '\\\\').replace(':', '\\:')
    
    if bgm_path.exists():
        cmd = [
            'ffmpeg', '-y', 
            '-i', video_path, 
            '-i', str(bgm_path),
            '-filter_complex', 
            "[1:a]volume=0.1[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[a]",
            '-map', '0:v', '-map', '[a]',
            '-vf', f"subtitles={escaped_srt}:force_style='Alignment=2,MarginV=50,Fontsize=18,PrimaryColour=&H00FFFFFF,BorderStyle=3,Outline=1,Shadow=1,BackColour=&H80000000'",
            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', output_path
        ]
    else:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', f"subtitles={escaped_srt}:force_style='Alignment=2,MarginV=50'",
            '-c:v', 'libx264', '-c:a', 'aac', output_path
        ]
        
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# ==========================================
# 10. PUBLISHING & NOTIFICATIONS
# ==========================================
def push_to_github(final_video_path, title):
    """Commits and pushes the generated video to GitHub."""
    if not GITHUB_TOKEN or not REPO_NAME:
        return "Github credentials not set"
        
    date_str = datetime.now().strftime("%Y-%m-%d")
    target_path = DIRS["generated"] / f"{date_str}.mp4"
    shutil.copy(final_video_path, target_path)
    
    try:
        repo = Repo(BASE_DIR)
        repo.git.add(all=True)
        repo.index.commit(f"New true crime video: {title}")
        origin = repo.remote(name='origin')
        origin.push()
        return f"https://raw.githubusercontent.com/{REPO_NAME}/main/generated_videos/{date_str}.mp4"
    except Exception as e:
        logging.error(f"GitHub Push Failed: {e}")
        return "Upload failed"

def send_webhook(data):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json=data)
        except Exception as e:
            logging.error(f"Webhook Failed: {e}")

def send_telegram(video_path, title):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: 
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    caption = f"{title}\n\n#truecrime #crime #mystery #documentary #shorts"
    
    try:
        with open(video_path, 'rb') as video_file:
            requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}, files={'video': video_file})
    except Exception as e:
        logging.error(f"Telegram Post Failed: {e}")

# ==========================================
# 11. MAIN PIPELINE
# ==========================================
def run_pipeline():
    logging.info("--- Starting Daily True Crime Automation Pipeline ---")
    try:
        # 1. Fetch
        case_data = get_random_case()
        logging.info(f"Selected case: {case_data['title']}")
        
        # 2. Script
        script = generate_script(case_data)
        logging.info("Script generated.")
        
        # 3. Images
        image_paths = download_and_format_images(case_data['images'])
        logging.info(f"Processed {len(image_paths)} images.")
        
        # 4. TTS
        audio_path = generate_voice(script)
        logging.info("Audio voiceover generated.")
        
        # 5. Video Base
        temp_video, duration = build_cinematic_video(image_paths, audio_path)
        logging.info(f"Base video built. Duration: {duration}s")
        
        # 6. Subtitles
        srt_path = generate_srt(script, duration)
        logging.info("Subtitles generated.")
        
        # 7. Final Mix
        final_video = str(DIRS["output"] / "final.mp4")
        mix_final_video(temp_video, srt_path, final_video)
        logging.info("Final video rendered and mixed.")
        
        # 8. Upload & Notify
        github_url = push_to_github(final_video, case_data['title'])
        
        webhook_data = {
            "title": case_data['title'],
            "case": case_data['case_id'],
            "video_url": github_url,
            "duration": duration,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        send_webhook(webhook_data)
        send_telegram(final_video, case_data['title'])
        logging.info("Notifications sent.")
        
        # 9. Save History
        save_history({
            "case_id": case_data['case_id'],
            "title": case_data['title'],
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "video_filename": f"{datetime.now().strftime('%Y-%m-%d')}.mp4",
            "github_url": github_url
        })
        logging.info("--- Pipeline completed successfully. ---")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        logging.error(traceback.format_exc())

# ==========================================
# 12. SCHEDULER
# ==========================================
if __name__ == "__main__":
    print("True Crime Automation Started.")
    print("Initial run starting now...")
    
    # Run once immediately on startup
    run_pipeline()
    
    # Then schedule daily
    schedule.every().day.at("10:00").do(run_pipeline)
    print("Scheduler active. Waiting for next daily run at 10:00 AM...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)
