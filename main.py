import os
import time
import textwrap
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
from googletrans import Translator
from gtts import gTTS

import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import st7735

import arabic_reshaper
from bidi.algorithm import get_display


# =========================
# SETTINGS
# =========================
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
AUDIO_FILE = "input.wav"
OUTPUT_AUDIO = "arabic.mp3"

# TFT pins
CS_PIN = board.CE0
DC_PIN = board.D25
RST_PIN = board.D24

BAUDRATE = 24000000

WIDTH = 128
HEIGHT = 160


# =========================
# TFT DISPLAY SETUP
# =========================
spi = board.SPI()

cs = digitalio.DigitalInOut(CS_PIN)
dc = digitalio.DigitalInOut(DC_PIN)
rst = digitalio.DigitalInOut(RST_PIN)

display = st7735.ST7735R(
    spi,
    cs=cs,
    dc=dc,
    rst=rst,
    baudrate=BAUDRATE,
    width=WIDTH,
    height=HEIGHT,
    rotation=0
)

image = Image.new("RGB", (WIDTH, HEIGHT), "black")
draw = ImageDraw.Draw(image)

try:
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font_small = ImageFont.load_default()
    font_medium = ImageFont.load_default()


def show_text(title, english_text="", arabic_text=""):
    image = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(image)

    draw.text((5, 5), title, font=font_medium, fill="yellow")

    y = 30

    if english_text:
        draw.text((5, y), "English:", font=font_small, fill="cyan")
        y += 16

        wrapped_en = textwrap.wrap(english_text, width=18)
        for line in wrapped_en[:3]:
            draw.text((5, y), line, font=font_small, fill="white")
            y += 14

    if arabic_text:
        y += 5
        draw.text((5, y), "Arabic:", font=font_small, fill="cyan")
        y += 16

        reshaped_text = arabic_reshaper.reshape(arabic_text)
        bidi_text = get_display(reshaped_text)

        wrapped_ar = textwrap.wrap(bidi_text, width=16)
        for line in wrapped_ar[:4]:
            draw.text((5, y), line, font=font_small, fill="white")
            y += 14

    display.image(image)


# =========================
# AUDIO RECORDING
# =========================
def record_audio():
    show_text("Listening...", "Speak English now")

    print("Recording...")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )
    sd.wait()

    wav.write(AUDIO_FILE, SAMPLE_RATE, audio)
    print("Recording finished")


# =========================
# MAIN PROGRAM
# =========================
def main():
    show_text("Translator Ready", "English to Arabic")
    time.sleep(2)

    print("Loading Whisper model...")
    show_text("Loading AI...", "Please wait")

    model = whisper.load_model("tiny")
    translator = Translator()

    while True:
        try:
            record_audio()

            show_text("Processing...", "Converting speech")

            result = model.transcribe(AUDIO_FILE, language="en")
            english_text = result["text"].strip()

            print("English:", english_text)

            if english_text == "":
                show_text("No Speech Found", "Try again")
                time.sleep(2)
                continue

            show_text("Translating...", english_text)

            translated = translator.translate(
                english_text,
                src="en",
                dest="ar"
            )

            arabic_text = translated.text
            print("Arabic:", arabic_text)

            show_text("Translated", english_text, arabic_text)

            tts = gTTS(text=arabic_text, lang="ar")
            tts.save(OUTPUT_AUDIO)

            os.system(f"mpg123 {OUTPUT_AUDIO}")

            time.sleep(2)

        except KeyboardInterrupt:
            show_text("Stopped", "Program ended")
            break

        except Exception as e:
            print("Error:", e)
            show_text("Error", str(e)[:40])
            time.sleep(3)


if __name__ == "__main__":
    main()
