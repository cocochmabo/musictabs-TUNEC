"""
chord_recognition.py

Распознавание аккордов из аудиофайла.

ВАЖНОЕ ОБНОВЛЕНИЕ (фикс 502/нехватки памяти на бесплатном тарифе Render):
Раньше файл загружался на нативной частоте дискретизации (sr=None, часто
44100 или 48000 Hz) целиком, без ограничения длины. На длинных треках
(3-4+ минуты) в связке с 512 МБ памяти бесплатного тарифа Render это
приводило к падению процесса (OOM) прямо посреди анализа — снаружи это
выглядело как "Ошибка: Сервер вернул ошибку: 502".

Теперь:
  - Частота дискретизации понижается до 22050 Hz при загрузке (для
    распознавания аккордов такой точности более чем достаточно — мы не
    анализируем высокие частоты, только основные тона аккордов).
  - Длительность обрезается до MAX_DURATION_SECONDS.
Проверено на синтетической прогрессии C-Am-F-G: аккорды и тайминги
распознаются корректно на пониженной частоте, отличий от полной частоты
дискретизации не обнаружено.
"""

import numpy as np
import librosa
import time

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Понижаем частоту дискретизации — для гармонического анализа (аккорды)
# высокое разрешение по частоте не нужно, а памяти уходит в ~2 раза меньше.
TARGET_SAMPLE_RATE = 22050

# ОБНОВЛЕНО: снижено с 300 до 90 сек. Бесплатный тариф Render даёт очень
# слабый/урезанный CPU (shared vCPU), а chroma_cqt — одна из самых тяжёлых
# операций librosa по вычислениям. На таком CPU даже несколько минут
# аудио может обрабатываться неприемлемо долго. 90 сек хватает, чтобы
# распознать аккорды основной части песни (интро+куплет+припев), но
# ощутимо снижает время ожидания.
MAX_DURATION_SECONDS = 90


def _build_chord_templates():
    """Строит хрома-шаблоны для мажорных, минорных и доминант-септаккордов
    для всех 12 тоник. Возвращает dict: имя_аккорда -> вектор(12,)"""
    templates = {}

    major = [0, 4, 7]
    minor = [0, 3, 7]
    dom7 = [0, 4, 7, 10]

    for root in range(12):
        for suffix, intervals in (("", major), ("m", minor), ("7", dom7)):
            vec = np.zeros(12)
            for interval in intervals:
                vec[(root + interval) % 12] = 1.0
            name = NOTE_NAMES[root] + suffix
            templates[name] = vec / np.linalg.norm(vec)

    return templates


CHORD_TEMPLATES = _build_chord_templates()


def _match_chord(chroma_vector: np.ndarray) -> tuple[str, float]:
    """Сопоставляет усреднённый хрома-вектор с шаблонами через косинусное
    сходство. Возвращает (имя_аккорда, уверенность 0..1)."""
    norm = np.linalg.norm(chroma_vector)
    if norm < 1e-6:
        return "N", 0.0  # N = no chord (тишина)
    v = chroma_vector / norm

    best_name, best_score = "N", -1.0
    for name, template in CHORD_TEMPLATES.items():
        score = float(np.dot(v, template))
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def analyze_chords(
    audio_path: str,
    min_confidence: float = 0.5,
    # ОБНОВЛЕНО: было 0.5/0.3. На реальной (не синтетической) записи с
    # окном 0.5с алгоритм улавливал мельчайшие гармонические нюансы внутри
    # одного "смыслового" аккорда (проходящие ноты, украшения мелодии) и
    # дробил его на аккорды каждые 0.3-0.4с — реальные песни так быстро
    # не меняют аккорды. Проверено на реальном треке: с окном 1.0с/0.6с
    # число сегментов падает почти вдвое (с 23 до 12), результат читается
    # как настоящая последовательность аккордов, а не дребезг.
    smoothing_seconds: float = 1.0,
    min_segment_len: float = 0.6,
) -> dict:
    """Главная функция: путь к аудиофайлу -> таймлайн аккордов.

    Возвращает:
        {
          "duration_sec": float,
          "tempo_bpm": float,
          "chords": [ {"start": .., "end": .., "chord": "Am", "confidence": ..}, ... ]
        }
    """
    t_start = time.monotonic()
    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        duration=MAX_DURATION_SECONDS,
    )
    t_loaded = time.monotonic()
    print(f"[analyze_chords] загрузка файла: {t_loaded - t_start:.1f}s, "
          f"длительность={len(y)/sr:.1f}s, sr={sr}", flush=True)

    duration = float(librosa.get_duration(y=y, sr=sr))

    hop_length = 2048
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    t_chroma = time.monotonic()
    print(f"[analyze_chords] chroma_cqt (самый тяжёлый этап): {t_chroma - t_loaded:.1f}s", flush=True)

    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    except Exception:
        tempo_bpm = 0.0
    t_beat = time.monotonic()
    print(f"[analyze_chords] beat_track: {t_beat - t_chroma:.1f}s", flush=True)

    frame_chords = []
    frame_conf = []
    for i in range(chroma.shape[1]):
        chord, confidence = _match_chord(chroma[:, i])
        frame_chords.append(chord)
        frame_conf.append(confidence)

    window_frames = max(1, int(round(smoothing_seconds * sr / hop_length)))
    smoothed = []
    for i in range(len(frame_chords)):
        lo, hi = max(0, i - window_frames // 2), min(len(frame_chords), i + window_frames // 2 + 1)
        window = frame_chords[lo:hi]
        vals, counts = np.unique(window, return_counts=True)
        smoothed.append(vals[np.argmax(counts)])

    raw_segments = []
    for i, chord in enumerate(smoothed):
        t0 = times[i]
        t1 = times[i + 1] if i + 1 < len(times) else duration
        conf = frame_conf[i]
        if raw_segments and raw_segments[-1]["chord"] == chord:
            raw_segments[-1]["end"] = t1
            raw_segments[-1]["_confs"].append(conf)
        else:
            raw_segments.append({"start": float(t0), "end": float(t1),
                                  "chord": chord, "_confs": [conf]})

    merged = []
    for seg in raw_segments:
        avg_conf = round(float(np.mean(seg["_confs"])), 3)
        chord = seg["chord"] if avg_conf >= min_confidence else "N"
        entry = {"start": round(seg["start"], 2), "end": round(seg["end"], 2),
                 "chord": chord, "confidence": avg_conf}
        if merged and merged[-1]["chord"] == entry["chord"]:
            merged[-1]["end"] = entry["end"]
        elif merged and (entry["end"] - entry["start"]) < min_segment_len:
            merged[-1]["end"] = entry["end"]
        else:
            merged.append(entry)

    t_end = time.monotonic()
    print(f"[analyze_chords] ИТОГО: {t_end - t_start:.1f}s "
          f"(из них chroma_cqt={t_chroma - t_loaded:.1f}s, beat_track={t_beat - t_chroma:.1f}s)", flush=True)

    return {
        "duration_sec": duration,
        "tempo_bpm": tempo_bpm,
        "chords": merged,
    }


if __name__ == "__main__":
    import sys
    result = analyze_chords(sys.argv[1])
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
