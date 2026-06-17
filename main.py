import os
import time
import textwrap

import sounddevice as sd
import scipy.io.wavfile as wav
import whisper

from deep_translator import GoogleTranslator
from gtts import gTTS

import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import st7735

import arabic_reshaper
from bidi.algorithm import get_display


# ==========================
# SETTINGS
# ==========================
RECORD_SECONDS = 5
SAMPLE_RATE = 16000

AUDIO_FILE = "input.wav"
OUTPUT_AUDIO = "arabic.mp3"

# ST7735 TFT Pins
CS_PIN = board.CE0
DC_PIN = board.D25
RST_PIN = board.D24

WIDTH = 128
HEIGHT = 160
BAUDRATE = 24000000


# ==========================
# TFT SETUP
# ==========================
spi = board.SPI()

cs = digitalio.DigitalInOut(CS_PIN)
dc = digitalio.DigitalInOut(DC_PIN)
rst = digitalio.DigitalInOut(RST_PIN)

display = st7735.ST7735R(
    spi,
    cs=cs,
    dc=dc,
    rst=rst,
    width=WIDTH,
    height=HEIGHT,
    rotation=0,
    baudrate=BAUDRATE
)

try:
    font_title = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
    )
    font_text = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
    )
except:
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()


def show_text(title, english="", arabic=""):
    image = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(image)

    draw.text((5, 5), title, font=font_title, fill="yellow")

    y = 30

    if english:
        draw.text((5, y), "EN:", font=font_text, fill="cyan")
        y += 15

        for line in textwrap.wrap(english, width=18):
            draw.text((5, y), line, font=font_text, fill="white")
            y += 14

    if arabic:
        y += 5
        draw.text((5, y), "AR:", font=font_text, fill="green")
        y += 15

        reshaped = arabic_reshaper.reshape(arabic)
        bidi_text = get_display(reshaped)

        for line in textwrap.wrap(bidi_text, width=16):
            draw.text((5, y), line, font=font_text, fill="white")
            y += 14

    display.image(image)


# ==========================
# RECORD AUDIO
# ==========================
def record_audio():
    show_text("Listening...", "Speak English")

    print("Recording...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    wav.write(AUDIO_FILE, SAMPLE_RATE, audio)

    print("Saved audio.")


# ==========================
# MAIN PROGRAM
# ==========================
def main():
    show_text("Loading AI...")

    print("Loading Whisper model...")
    model = whisper.load_model("tiny")

    show_text("Ready", "English -> Arabic")
    time.sleep(2)

    while True:
        try:
            record_audio()

            show_text("Processing...")

            result = model.transcribe(
                AUDIO_FILE,
                language="en"
            )

            english_text = result["text"].strip()

            print("English:", english_text)

            if not english_text:
                show_text("No Speech", "Try again")
                time.sleep(2)
                continue

            show_text("Translating...", english_text)

            arabic_text = GoogleTranslator(
                source="en",
                target="ar"
            ).translate(english_text)

            print("Arabic:", arabic_text)

            show_text(
                "Translation",
                english_text,
                arabic_text
            )

            tts = gTTS(
                text=arabic_text,
                lang="ar"
            )

            tts.save(OUTPUT_AUDIO)

            os.system(f"mpg123 {OUTPUT_AUDIO}")

            time.sleep(2)

        except KeyboardInterrupt:
            show_text("Stopped")
            break

        except Exception as e:
            print("Error:", e)

            show_text(
                "Error",
                str(e)[:18]
            )

            time.sleep(3)


if __name__ == "__main__":
    main()
