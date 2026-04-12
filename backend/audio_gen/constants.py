"""Constants for ElevenLabs dataset generation."""

from __future__ import annotations

MANIFEST_VERSION = "v1"

OUTPUT_CLIPS_FILE = "clips.jsonl"
OUTPUT_ERRORS_FILE = "generation_errors.jsonl"
OUTPUT_RUN_METADATA_FILE = "run_metadata.json"

CLEAN_SAMPLE_RATE = 16_000
TELEPHONY_SAMPLE_RATE = 8_000
PCM_CODEC = "pcm_s16le"
WAV_CONTAINER = "wav"
MONO_CHANNELS = 1

REQUIRED_COLUMNS = {
    "clip_id",
    "script_family_id",
    "base_script_id",
    "text",
    "voice_id",
    "voice_type",
    "speech_style",
    "accent",
    "category",
    "difficulty",
    "split",
    "noise_level",
    "has_interruptions",
    "contains_numeric_confusion",
    "numeric_confusion_type",
    "contains_medical_terms",
    "contains_ambiguity",
    "scenario",
    "scenario_group",
    "noise_profile",
    "accent_profile",
    "medical_domain",
    "medical_subtype",
}

REQUIRED_NON_EMPTY_STRING_FIELDS = {
    "clip_id",
    "script_family_id",
    "base_script_id",
    "text",
    "voice_id",
    "voice_type",
    "speech_style",
    "accent",
    "category",
    "difficulty",
    "split",
    "noise_level",
    "numeric_confusion_type",
    "scenario",
    "scenario_group",
    "noise_profile",
}

BOOLEAN_FIELDS = {
    "has_interruptions",
    "contains_numeric_confusion",
    "contains_medical_terms",
    "contains_ambiguity",
    "medical_domain",
}

ALLOWED_SCENARIOS = {
    "clean_speech",
    "noisy_environment",
    "accented_speech",
    "medical_conversation",
}

ALLOWED_SCENARIO_GROUPS = {
    "baseline",
    "noisy",
    "accented",
    "medical",
}

ALLOWED_VOICE_TYPES = {
    "neutral",
    "telephony",
    "accented",
    "clinical",
}

ALLOWED_NUMERIC_CONFUSION_TYPES = {
    "digit_vs_digit",
    "dose_confusion",
    "duration_confusion",
    "none",
}

WORD_TEMPLATE_COLUMNS = [
    "clip_id",
    "word_index",
    "ground_truth",
    "stt_word",
    "is_error",
    "error_type",
    "start_ms",
    "end_ms",
    "duration_ms",
    "speaker",
    "context_prev",
    "context_next",
    "is_medical_term",
    "is_numeric",
    "closest_keyterm",
    "edit_distance",
    "phonetic_match",
    "difficulty_tags",
    "noise_level",
]

NUMERIC_TEMPLATE_COLUMNS = [
    "clip_id",
    "word_index",
    "numeric_value_gt",
    "numeric_value_stt",
    "numeric_error",
    "numeric_type",
]

MEDICAL_TEMPLATE_COLUMNS = [
    "clip_id",
    "word_index",
    "entity_type",
    "entity_gt",
    "entity_stt",
    "entity_verified",
]
