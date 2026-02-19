import os
import json
import random
import requests
import cv2
import numpy as np
import subprocess
import logging
import traceback
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Purane gTTS wale import hata kar ye add karein:
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
import moviepy.audio.fx.all as afx
# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
load_dotenv()

FBI_API_URL = os.getenv("FBI_API_URL", "https://api.fbi.gov/wanted/v1/list")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_DIR = Path(__file__).resolve().parent
DIRS = {
    "images": BASE_DIR / "assets" / "images",
    "audio": BASE_DIR / "assets" / "audio",
    "music": BASE_DIR / "assets" / "music",
    "history": BASE_DIR / "history",
    "output": BASE_DIR / "output",
    "logs": BASE_DIR / "logs"
}

HISTORY_FILE = DIRS["history"] / "used_cases.json"

def init_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Agar history file nahi hai, toh use khali list ke sath banao
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)
            
    bgm = DIRS["music"] / "background.mp3"
    if not bgm.exists():
        print(f"Warning: Please place a 'background.mp3' in {DIRS['music']}")

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
    with open(HISTORY_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

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
    for attempt in range(10): # 10 baar alag-alag page try karega
        page = random.randint(1, 40)
        response = requests.get(f"{FBI_API_URL}?page={page}")
        
        if response.status_code != 200:
            continue
            
        items = response.json().get("items", [])
        random.shuffle(items)
        
        for item in items:
            # CHANGE YAHAN HAI: `>= 4` ki jagah `>= 1` kar diya
            if item.get("images") and len(item["images"]) >= 1 and not is_used(item["uid"]):
                return {
                    "case_id": item["uid"],
                    "title": item.get("title", "Unknown Subject"),
                    "description": item.get("description", ""),
                    "place": item.get("place_of_birth", "an unknown location"),
                    "caution": item.get("caution", "Details are still unfolding."),
                    "images": [img["original"] for img in item["images"]]
                }
                
    raise ValueError("10 attempts ke baad bhi koi suitable case nahi mila.")

# ==========================================
# 4. SCRIPT GENERATOR
# ==========================================
def clean_html(raw_html):
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html).replace('&nbsp;', ' ').strip()

def generate_script(case_data):
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
    
    script = " ".join(script.split()) 
    script_path = BASE_DIR / "assets" / "script.txt"
    with open(script_path, "w") as f:
        f.write(script)
        
    return script
# ==========================================
# 5. IMAGE PROCESSOR
# ==========================================
def download_and_format_images(image_urls):
    formatted_paths = []
    target_w, target_h = 1080, 1920
    # Ye FBI ki website ko lagega ki Google Chrome se request aa rahi hai
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    for i, url in enumerate(image_urls[:4]):
        try:
            resp = requests.get(url, stream=True, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
                
            img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None: continue
                
            h, w = img.shape[:2]
            bg = cv2.resize(img, (target_w, target_h))
            bg = cv2.GaussianBlur(bg, (99, 99), 30)
            
            scale = min(target_w/w, target_h/h) * 0.9
            new_w, new_h = int(w * scale), int(h * scale)
            fg = cv2.resize(img, (new_w, new_h))
            
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = fg
            bg = cv2.convertScaleAbs(bg, alpha=1.1, beta=10)
            
            out_path = DIRS["images"] / f"img0{i+1}.jpg"
            cv2.imwrite(str(out_path), bg)
            formatted_paths.append(str(out_path))
        except Exception as e:
            print(f"Image download fail ho gayi: {e}")
            continue
            
    return formatted_paths

# ==========================================
# 6. TTS & SUBTITLES (NEW SERIOUS VOICE)
# ==========================================
def generate_voice(script_text):
    out_path = DIRS["audio"] / "voice.wav"
    temp_mp3 = DIRS["audio"] / "temp.mp3"
    script_file = BASE_DIR / "assets" / "script.txt"
    
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_text)
        
    # LOGIC: Edge-TTS ka use serious case-study aawaz ke liye ('GuyNeural' voice aur thodi slow speed)
    os.system(f'edge-tts --voice "en-US-GuyNeural" --rate="-10%" --file "{script_file}" --write-media "{temp_mp3}"')
    
    os.system(f"ffmpeg -y -i \"{temp_mp3}\" -ar 44100 \"{out_path}\" -loglevel error")
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
    return str(out_path)

def generate_srt(script_text, audio_duration):
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
# 7. VIDEO BUILDER & FINAL RENDER (MUSIC LOOP & SMALL FONT)
# ==========================================
def build_cinematic_video(image_paths, audio_path):
    if len(image_paths) == 0:
        raise ValueError("Koi valid image process nahi ho payi. Case skip kiya ja raha hai.")

    voice_audio = AudioFileClip(audio_path)
    actual_audio_duration = voice_audio.duration
    
    # Video kam se kam 30 second ka hoga
    video_duration = max(actual_audio_duration, 30.0)
    
    # LOGIC: Background music ko pehle hi load karke loop kar diya (taaki last tak chale)
    bgm_path = DIRS["music"] / "background.mp3"
    if bgm_path.exists():
        bgm_audio = AudioFileClip(str(bgm_path)).fx(afx.volumex, 0.1) # BGM volume 10%
        bgm_audio = afx.audio_loop(bgm_audio, duration=video_duration)
        final_audio = CompositeAudioClip([bgm_audio, voice_audio.set_start(0)])
    else:
        final_audio = voice_audio
        
    clip_duration = video_duration / len(image_paths)
    clips = []
    
    for i, img_path in enumerate(image_paths):
        clip = ImageClip(img_path).set_duration(clip_duration)
        if i % 2 == 0:
            clip = clip.resize(lambda t: 1 + 0.02 * t).set_position(('center', 'center'))
        else:
            clip = clip.resize(lambda t: 1.1 - 0.02 * t).set_position(('center', 'center'))
        clips.append(clip)
        
    video = concatenate_videoclips(clips, method="compose")
    video = video.set_audio(final_audio) # Ab voice aur loop hua BGM dono isme hain
    
    temp_video = DIRS["output"] / "temp_video.mp4"
    video.write_videofile(str(temp_video), fps=30, codec="libx264", audio_codec="aac", logger=None)
    
    return str(temp_video), actual_audio_duration

def mix_final_video(video_path, srt_path, output_path):
    escaped_srt = str(srt_path).replace('\\', '\\\\').replace(':', '\\:')
    
    # LOGIC: BGM pehle add ho gaya, ab yahan sirf Fontsize=12 (Chota font) kiya gaya hai
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vf', f"subtitles={escaped_srt}:force_style='Alignment=2,MarginV=50,Fontsize=12,PrimaryColour=&H00FFFFFF,BorderStyle=3,Outline=1,Shadow=1,BackColour=&H80000000'",
        '-c:v', 'libx264', '-c:a', 'aac', output_path
    ]
        
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# ==========================================
# 7. VIDEO BUILDER & FINAL RENDER
# ==========================================
def build_cinematic_video(image_paths, audio_path):
    if len(image_paths) == 0:
        raise ValueError("Koi valid image process nahi ho payi. Case skip kiya ja raha hai.")

    voice_audio = AudioFileClip(audio_path)
    actual_audio_duration = voice_audio.duration
    
    # LOGIC: Video duration kam se kam 30 second hogi
    video_duration = max(actual_audio_duration, 30.0)
    
    clip_duration = video_duration / len(image_paths)
    clips = []
    
    for i, img_path in enumerate(image_paths):
        clip = ImageClip(img_path).set_duration(clip_duration)
        if i % 2 == 0:
            clip = clip.resize(lambda t: 1 + 0.02 * t).set_position(('center', 'center'))
        else:
            clip = clip.resize(lambda t: 1.1 - 0.02 * t).set_position(('center', 'center'))
        clips.append(clip)
        
    video = concatenate_videoclips(clips, method="compose")
    video = video.set_audio(voice_audio.set_start(0)) # Yahan sirf voice lagayi hai
    
    temp_video = DIRS["output"] / "temp_video.mp4"
    video.write_videofile(str(temp_video), fps=30, codec="libx264", audio_codec="aac", logger=None)
    
    return str(temp_video), actual_audio_duration

def mix_final_video(video_path, srt_path, output_path):
    bgm_path = DIRS["music"] / "background.mp3"
    escaped_srt = str(srt_path).replace('\\', '\\\\').replace(':', '\\:')
    
    if bgm_path.exists():
        print("Background music file mil gayi! FFmpeg se mix aur loop kar rahe hain...")
        # LOGIC: FFmpeg music ko infinite loop (-stream_loop -1) karega aur video ki duration tak chalayega
        cmd = [
            'ffmpeg', '-y', 
            '-i', video_path, 
            '-stream_loop', '-1', '-i', str(bgm_path),
            '-filter_complex', "[1:a]volume=0.1[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[a]",
            '-map', '0:v', '-map', '[a]',
            '-vf', f"subtitles={escaped_srt}:force_style='Alignment=2,MarginV=50,Fontsize=12,PrimaryColour=&H00FFFFFF,BorderStyle=3,Outline=1,Shadow=1,BackColour=&H80000000'",
            '-c:v', 'libx264', '-c:a', 'aac', output_path
        ]
    else:
        print("WARNING: background.mp3 file nahi mili! Bina BGM ke video ban raha hai.")
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', f"subtitles={escaped_srt}:force_style='Alignment=2,MarginV=50,Fontsize=12,PrimaryColour=&H00FFFFFF,BorderStyle=3,Outline=1,Shadow=1,BackColour=&H80000000'",
            '-c:v', 'libx264', '-c:a', 'aac', output_path
        ]
        
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# ==========================================
# 8. TELEGRAM SENDER
# ==========================================
def send_telegram(video_path, title):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: 
        print("Error: Telegram credentials missing!")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    caption = f"🚨 *{title}*\n\n#truecrime #crime #mystery #documentary #shorts"
    
    with open(video_path, 'rb') as video_file:
        response = requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}, files={'video': video_file})
        response.raise_for_status()

# ==========================================
# 9. MAIN EXECUTION
# ==========================================
def run_pipeline():
    print("Starting GitHub Action True Crime Automation...")
    case_data = get_random_case()
    print(f"Case selected: {case_data['title']}")
    
    script = generate_script(case_data)
    image_paths = download_and_format_images(case_data['images'])
    audio_path = generate_voice(script)
    
    temp_video, duration = build_cinematic_video(image_paths, audio_path)
    srt_path = generate_srt(script, duration)
    
    final_video = str(DIRS["output"] / "final.mp4")
    mix_final_video(temp_video, srt_path, final_video)
    
    send_telegram(final_video, case_data['title'])
    print("Video sent to Telegram!")
    
    save_history({
        "case_id": case_data['case_id'],
        "title": case_data['title'],
        "created_date": datetime.now().strftime("%Y-%m-%d")
    })
    print("History updated successfully!")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        traceback.print_exc()
        exit(1)
