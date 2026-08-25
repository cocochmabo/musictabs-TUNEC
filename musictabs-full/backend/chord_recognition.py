"""
chord_recognition.py

Распознавание аккордов из аудиофайла.

Подход (классический, надёжный, без тяжёлых ML-моделей):
1. Загружаем аудио, приводим к моно.
2. Считаем хромаграмму (CQT-based chroma) — 12 значений энергии
   на каждый момент времени, по одному на каждую ноту (C, C#, D, ...).
3. Определяем биты (beat tracking), чтобы сегментировать трек на
   музыкально осмысленные куски (обычно аккорд держится 1 такт/долю).
4. Усредняем хрому внутри каждого сегмента и сравниваем (косинусное
   сходство) с шаблонами аккордов (24 базовых: 12 мажор + 12 минор,
   плюс опционально septims).
5. Склеиваем соседние сегменты с одинаковым аккордом.

Результат: список {start_sec, end_sec, chord} — то, что нужно для
отображения "аккорды + ритм" в приложении.
"""

import numpy as np
import librosa

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _build_chord_templates():
    """Строит хрома-шаблоны для мажорных, минорных и доминант-септаккордов
    для всех 12 тоник. Возвращает dict: имя_аккорда -> вектор(12,)"""
    templates = {}

    # Интервалы в полутонах от тоники
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


def analyze_chords(audio_path: str, min_confidence: float = 0.5) -> dict:
    """Главная функция: путь к аудиофайлу -> таймлайн аккордов.

    Возвращает:
        {
          "duration_sec": float,
          "tempo_bpm": float,
          "chords": [ {"start": .., "end": .., "chord": "Am", "confidence": ..}, ... ]
        }
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Хромаграмма (CQT-based — устойчивее для гармонического анализа)
    hop_length = 2048
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    # Пытаемся получить темп (может не сработать на материале без чёткой
    # перкуссии — это ок, на разбивку по аккордам это не влияет,
    # используется только как метаданные для UI)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    except Exception:
        tempo_bpm = 0.0

    # Покадровое распознавание аккорда (независимо от beat tracking —
    # надёжнее, работает на любом материале)
    frame_chords = []
    frame_conf = []
    for i in range(chroma.shape[1]):
        chord, confidence = _match_chord(chroma[:, i])
        frame_chords.append(chord)
        frame_conf.append(confidence)

    # Медианное сглаживание по времени (окно ~0.5 сек), чтобы убрать
    # дребезг на переходах и коротких артефактах
    window_frames = max(1, int(round(0.5 * sr / hop_length)))
    smoothed = []
    for i in range(len(frame_chords)):
        lo, hi = max(0, i - window_frames // 2), min(len(frame_chords), i + window_frames // 2 + 1)
        window = frame_chords[lo:hi]
        vals, counts = np.unique(window, return_counts=True)
        smoothed.append(vals[np.argmax(counts)])

    # Собираем смежные одинаковые кадры в сегменты
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
    min_segment_len = 0.3  # сек — отбрасываем совсем короткие дребезжащие сегменты
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
