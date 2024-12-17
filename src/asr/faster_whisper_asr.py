import asyncio

from faster_whisper import WhisperModel
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time

from .asr_interface import ASRInterface

language_codes = {
    "afrikaans": "af",
    "amharic": "am",
    "arabic": "ar",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "belarusian": "be",
    "bulgarian": "bg",
    "bengali": "bn",
    "tibetan": "bo",
    "breton": "br",
    "bosnian": "bs",
    "catalan": "ca",
    "czech": "cs",
    "welsh": "cy",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "estonian": "et",
    "basque": "eu",
    "persian": "fa",
    "finnish": "fi",
    "faroese": "fo",
    "french": "fr",
    "galician": "gl",
    "gujarati": "gu",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "croatian": "hr",
    "haitian": "ht",
    "hungarian": "hu",
    "armenian": "hy",
    "indonesian": "id",
    "icelandic": "is",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "georgian": "ka",
    "kazakh": "kk",
    "khmer": "km",
    "kannada": "kn",
    "korean": "ko",
    "latin": "la",
    "luxembourgish": "lb",
    "lingala": "ln",
    "lao": "lo",
    "lithuanian": "lt",
    "latvian": "lv",
    "malagasy": "mg",
    "maori": "mi",
    "macedonian": "mk",
    "malayalam": "ml",
    "mongolian": "mn",
    "marathi": "mr",
    "malay": "ms",
    "maltese": "mt",
    "burmese": "my",
    "nepali": "ne",
    "dutch": "nl",
    "norwegian nynorsk": "nn",
    "norwegian": "no",
    "occitan": "oc",
    "punjabi": "pa",
    "polish": "pl",
    "pashto": "ps",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "sindhi": "sd",
    "sinhalese": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "shona": "sn",
    "somali": "so",
    "albanian": "sq",
    "serbian": "sr",
    "sundanese": "su",
    "swedish": "sv",
    "swahili": "sw",
    "tamil": "ta",
    "telugu": "te",
    "tajik": "tg",
    "thai": "th",
    "turkmen": "tk",
    "tagalog": "tl",
    "turkish": "tr",
    "tatar": "tt",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "yiddish": "yi",
    "yoruba": "yo",
    "chinese": "zh",
    "cantonese": "yue",
}


class WhisperWorker:
    def __init__(self, model_size):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16",
            num_workers=1,
        )

    def transcribe(self, buffer):
        ndarray = np.frombuffer(buffer, dtype=np.int16)
        segments, info = self.model.transcribe(ndarray)
        segments = list(segments)
        return {
            "text": " ".join([s.text.strip() for s in segments]),
        }


def init_worker(model_size):
    # This will run once per worker process
    global worker
    worker = WhisperWorker(model_size)


def transcribe_worker(buffer):
    # Uses the global worker instance initialized in init_worker
    global worker
    return worker.transcribe(buffer)


class FasterWhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        self.model_size = kwargs.get("model_size", "large-v3")
        # Initialize pool with workers that already have the model loaded
        self.process_pool = ProcessPoolExecutor(
            max_workers=2, initializer=init_worker, initargs=(self.model_size,)
        )

    async def transcribe(self, buffer):
        loop = asyncio.get_running_loop()

        try:
            result = await loop.run_in_executor(
                self.process_pool, transcribe_worker, buffer
            )
            return result
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": ""}

    async def cleanup(self):
        self.process_pool.shutdown(wait=True)
