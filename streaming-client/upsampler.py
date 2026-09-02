"""Prompt upsampling: turn a viewer's rough idea into fast-h3-ready scenes.

One LLM call per prompt, against any OpenAI-compatible endpoint. The model
picks the shape the idea calls for — one scene of any legal length, or a
chunked short story of up to `max_chunks` short clips with a setup,
development, and payoff — writes each scene as a self-contained
text-to-video prompt in the configured style/character, and picks each
scene's length in seconds.

Why the prompt is written the way it is — these rules come from how fast-h3
actually behaves, so keep them intact when editing:

  * **The model reads only the scene's own text.** The single biggest
    quality lever. Whatever the generation mode, a scene prompt is the only
    text the model sees for that clip: each scene must re-describe the
    entire setting, subjects, light, and style from scratch. Without
    chaining, "the same forest" renders a *different* forest; with chaining
    the place carries over visually but still mutates unless the text keeps
    naming it. (Learned on the earlier story-livestream client, where
    under-described scenes visibly "lost" characters between cuts.)
  * **Chained scenes must open on a hard cut, and the cut must be
    described.** On a continuation-capable deployment the director chains a
    story's scenes: each clip after the first opens on the previous clip's
    final frame. A chain written as one continuous take degrades visibly —
    every clip re-generates from a generated frame, small errors compound
    link over link, and within a few scenes the picture smears beyond use
    ("cooks"). A hard cut resets that: a scene that opens on a *new shot* —
    different camera angle, distance, or location, described from scratch —
    re-establishes the whole image, so the chain stays sharp indefinitely.
    The chained rules below therefore require every scene after the first
    to open with an explicit cut to a fully described new shot, and forbid
    "the camera continues..." / "still on..." phrasing. This rule is
    load-bearing; never soften it.
  * **800 characters is the model's hard cap per prompt** (`MAX_PROMPT_CHARS`
    in `fasth3_types.py`); the LLM is told 750 to leave headroom, and
    `_sanitize` hard-truncates anyway, because LLMs do not count characters
    reliably.
  * **fast-h3 renders synchronized audio, spoken language included**, so
    the prompt asks for explicit quoted dialogue (who speaks, the exact
    words, the voice's tone) whenever the idea implies speech, and for a
    brief soundscape clause — clips come out flat without them.
  * **Length policy: a single-clip generation always runs the maximum
    length** (the live `clip_seconds_max` from `state_update`), enforced in
    code after validation — long enough for the scene to breathe. The
    short end of the range is reserved for transition chunks inside
    multi-scene stories (an establishing cut, a reaction beat); content
    chunks run long.
  * **Safety is moderation's job, not this prompt's.** The viewer's idea has
    already passed the moderator by the time it arrives here, so the prompt
    asks for faithful staging and never for softening or reinterpreting the
    idea (see `moderator.py`).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# fast-h3's enqueue cap (fasth3_types.MAX_PROMPT_CHARS). Hard limit, enforced
# server-side; _sanitize truncates to it.
MAX_PROMPT_CHARS = 800
# How many LLM calls one idea gets before the raw-prompt fallback. Each
# attempt is warned about individually with the reply's head, so a systematic
# failure is diagnosable from the log.
_MAX_ATTEMPTS = 3

# What the LLM is asked to stay under, leaving headroom for its poor counting.
# Sized so an overshoot still fits under the 800 hard cap: the sanitizer
# truncates mid-word at 800, and what it cuts is the prompt's tail — the
# soundscape sentence the format deliberately puts last.
_TARGET_PROMPT_CHARS = 700

_SYSTEM_PROMPT = """\
You are the scene director of a live, chat-driven AI video stream. Viewers
send short, rough ideas; you turn each one into one or more polished
text-to-video prompts for a model that generates short clips with
synchronized audio.

STYLE / CHARACTER — every scene is rendered in this identity; weave it into
every scene prompt, never contradict it:
{style}

HOW THE VIDEO MODEL WORKS (hard constraints):
- Each scene becomes ONE clip, and the model reads ONLY that scene's text —
  it never sees the other scene prompts. Every scene prompt must be fully
  self-contained and re-describe the entire setting, subjects, lighting,
  palette, mood, and style — even when nothing changed from the previous
  scene. Anything you omit will vanish or mutate between scenes.
- Each scene prompt must be under {target_chars} characters. This is a hard
  limit; prefer cutting adjectives over cutting subjects or setting.
- Each scene has a duration in seconds, between {min_seconds} and
  {max_seconds}; the rules below say how to choose it.
- The model renders picture AND sound, including clear spoken language.
  When the idea involves someone speaking, write the dialogue out
  explicitly and unambiguously — name who speaks and give the exact words
  in quotes (e.g. the fisherman shouts "It's alive!") — and describe the
  voice's tone. Do not paraphrase speech the viewer asked for.
- End each scene prompt with one short clause of soundscape (ambience,
  music mood, or effects) alongside any dialogue.
- Describe only what the camera sees and the microphone hears: no text
  overlays, no UI, no scene numbers, no camera jargon the model cannot show.

{scene_count_rules}

WRITING THE SCENE PROMPTS:
- Be concrete and visual: subject, action, setting, camera angle and motion,
  lighting, color palette, atmosphere, then the soundscape clause.
- Strong nouns and verbs over piles of adjectives; vivid but precise.
- Keep the viewer's idea recognizable — enhance it, do not replace it. The
  idea has already passed moderation before it reaches you; your job is
  faithful staging, not policing.

Reply with ONLY this JSON, nothing else:
{{"title": "short display title for the sequence",
  "scenes": [{{"prompt": "self-contained scene description...", "seconds": 8.0}}]}}
The "scenes" array is REQUIRED even when it holds a single scene; never
flatten a scene's fields to the top level.
"""

_MULTI_SCENE_RULES = """\
HOW MANY SCENES, AND HOW LONG — two shapes; pick what the idea calls for:
- ONE SCENE: a single clip that ALWAYS runs the full {max_seconds} seconds —
  never shorter — with room for the scene to build, land, and breathe.
  Right for a mood, a place, a single action or gag. When in doubt, this.
- CHUNKED SHORT STORY: 3 to {max_chunks} chunks that read as one story with
  a setup, a development, and a payoff. Content chunks run 8-{max_seconds}
  seconds; the short end ({min_seconds}-8 s) is ONLY for transitions — an
  establishing cut, a reaction beat, a snap punchline — never for a chunk
  that carries the story. Choose this shape when the idea implies
  narrative: a journey, a transformation, a chase, a build-up.
- Never more than {max_chunks} scenes. Do not pad a thin idea into many
  chunks; a story earns its chunks or it is one full-length scene.
- Consecutive scenes play back-to-back as one sequence. Make them feel
  continuous: repeat the shared setting and subjects verbatim enough that
  they read as the same place, and change only what the story moves."""

_MULTI_SCENE_RULES_CHAINED = """\
HOW MANY SCENES, AND HOW LONG — two shapes; pick what the idea calls for:
- ONE SCENE: a single clip that ALWAYS runs the full {max_seconds} seconds —
  never shorter — with room for the scene to build, land, and breathe.
  Right for a mood, a place, a single action or gag. When in doubt, this.
- CHUNKED SHORT STORY: 3 to {max_chunks} chunks that read as one story with
  a setup, a development, and a payoff. Content chunks run 8-{max_seconds}
  seconds; the short end ({min_seconds}-8 s) is ONLY for transitions — an
  establishing cut, a reaction beat, a snap punchline — never for a chunk
  that carries the story. Choose this shape when the idea implies
  narrative: a journey, a transformation, a chase, a build-up.
- Never more than {max_chunks} scenes. Do not pad a thin idea into many
  chunks; a story earns its chunks or it is one full-length scene.
- The story's scenes are rendered as ONE continuous video: each scene
  begins on the exact final frame of the scene before it. Because of that,
  EVERY SCENE AFTER THE FIRST MUST OPEN ON A HARD CUT — a new shot with a
  clearly different camera angle, distance, or location — and its prompt
  must describe that new shot in full, as if the camera were set up fresh.
  Open the prompt with the cut itself, e.g. "Hard cut to a wide shot of
  ...", "Cut to: inside the lighthouse, a close-up of ...".
- NEVER write a scene that extends the previous scene's shot. No "the
  camera continues", "still on her face", "the shot lingers", "we keep
  following" — holding one take across scenes makes the picture smear and
  degrade scene over scene, while a clean cut keeps every scene sharp.
  Cuts are also film language: use them for rhythm, not just hygiene.
- Even across cuts the story stays continuous: re-describe the same
  setting and subjects verbatim enough that they read as the same place
  and cast, and change only what the story moves."""

_SINGLE_SCENE_RULES = """\
HOW MANY SCENES, AND HOW LONG:
- Exactly one scene, and it ALWAYS runs the full {max_seconds} seconds.
  Distill the idea into one complete arc that fills that time."""


@dataclass(frozen=True)
class Scene:
    """One upsampled scene: a prompt fast-h3 can take verbatim, and a length."""

    prompt: str
    seconds: float


@dataclass(frozen=True)
class SceneGroup:
    """The scenes one prompt expanded into, played back-to-back.

    ``generated`` marks filler groups made from the idle prompt list rather
    than a viewer request; the director may evict their clips from the
    model's queue to make room for viewer groups.
    """

    group_id: str
    title: str
    author: str
    source: str
    raw_prompt: str
    scenes: list[Scene]
    generated: bool = False


class PromptUpsampler:
    """Expand chat ideas into styled, self-contained fast-h3 scenes."""

    def __init__(
        self,
        api_key: str,
        model: str,
        style: str,
        max_chunks: int,
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._style = style.strip() or "Cinematic, photoreal, rich natural light."
        self._max_chunks = max_chunks

    def set_style(self, style: str) -> None:
        """Swap the style block new scenes are written in (a preset switch).

        Takes effect on the next LLM call; scenes already upsampled or queued
        keep the style they were written in.
        """
        self._style = style.strip() or self._style

    async def upsample(
        self,
        raw_prompt: str,
        author: str,
        source: str,
        min_seconds: float,
        max_seconds: float,
        generated: bool = False,
        max_chunks: int | None = None,
        chained: bool = False,
    ) -> SceneGroup:
        """One idea in, one validated scene group out. Never raises.

        `min_seconds`/`max_seconds` are the live bounds from the model's
        `state_update`, so the LLM always chooses within what the deployment
        actually accepts. `max_chunks` caps this call below the configured
        ceiling (the idle filler passes 1 so its groups stay one-clip and
        evictable). `chained` says the director will chain this group's
        scenes on the model (each clip opening on the previous one's final
        frame), which switches the multi-scene rules to the chained set:
        every scene after the first opens on a described hard cut, never a
        continuation of the same take (see the module docstring for why).
        On any LLM failure the raw prompt (styled, truncated) becomes a
        single scene — the stream keeps moving.
        """
        chunk_cap = min(max_chunks or self._max_chunks, self._max_chunks)
        multi_rules = _MULTI_SCENE_RULES_CHAINED if chained else _MULTI_SCENE_RULES
        scene_count_rules = (
            multi_rules.format(
                max_chunks=chunk_cap,
                min_seconds=f"{min_seconds:g}",
                max_seconds=f"{max_seconds:g}",
            )
            if chunk_cap > 1
            else _SINGLE_SCENE_RULES.format(max_seconds=f"{max_seconds:g}")
        )
        system = _SYSTEM_PROMPT.format(
            style=self._style,
            target_chars=_TARGET_PROMPT_CHARS,
            min_seconds=f"{min_seconds:g}",
            max_seconds=f"{max_seconds:g}",
            scene_count_rules=scene_count_rules,
        )
        group_id = uuid.uuid4().hex[:12]
        title = ""
        scenes: list[Scene] = []
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                title, scenes = await self._attempt(
                    system=system,
                    raw_prompt=raw_prompt,
                    request_tag=f"{group_id}.{attempt}",
                    chunk_cap=chunk_cap,
                    min_seconds=min_seconds,
                    max_seconds=max_seconds,
                )
                break
            except Exception as error:
                logger.warning(
                    "[upsampler] unusable reply, attempt %d/%d for %.60r: %s",
                    attempt, _MAX_ATTEMPTS, raw_prompt, error,
                )
        if not scenes:
            logger.warning(
                "[upsampler] all %d attempts unusable; falling back to the raw prompt",
                _MAX_ATTEMPTS,
            )
            title = raw_prompt[:60]
            # The viewer's idea gets the char budget first; the style fills
            # whatever remains (a long STYLE must never truncate the idea away).
            idea = _sanitize(raw_prompt)
            style_room = MAX_PROMPT_CHARS - len(idea) - 2
            fallback = f"{idea}. {self._style[:style_room]}" if style_room > 20 else idea
            scenes = [
                # A single clip, so it takes the maximum length like every
                # other one-scene generation.
                Scene(prompt=_sanitize(fallback), seconds=max_seconds)
            ]

        group = SceneGroup(
            group_id=group_id,
            title=title,
            author=author,
            source=source,
            raw_prompt=raw_prompt,
            scenes=scenes,
            generated=generated,
        )
        for index, scene in enumerate(group.scenes, start=1):
            logger.info(
                "[upsampler] %s scene %d/%d (%.1fs): %.100s...",
                group_id, index, len(group.scenes), scene.seconds, scene.prompt,
            )
        return group

    async def _attempt(
        self,
        *,
        system: str,
        raw_prompt: str,
        request_tag: str,
        chunk_cap: int,
        min_seconds: float,
        max_seconds: float,
    ) -> tuple[str, list[Scene]]:
        """One LLM call, parsed and validated; raises on an unusable reply.

        The request tag makes every attempt a distinct request — the gateway
        caches identical ones, so a bare retry of a failed prompt would get
        the same failed reply back in milliseconds.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Viewer idea: {raw_prompt}\n\n[request {request_tag}]",
                },
            ],
            temperature=0.8,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content or "{}")
        title = str(data.get("title") or raw_prompt[:60]).strip()
        raw_scenes = data.get("scenes")
        if isinstance(raw_scenes, dict):
            raw_scenes = [raw_scenes]
        if not raw_scenes and "prompt" in data:
            # Some models flatten a single scene's fields to the top level
            # despite the schema; accept it as one scene.
            raw_scenes = [data]
        scenes = self._validate_scenes(
            raw_scenes or [], chunk_cap, min_seconds, max_seconds
        )
        if not scenes:
            raise ValueError(
                "no usable scenes in the reply "
                f"(finish={response.choices[0].finish_reason}, head={content[:200]!r})"
            )
        if len(scenes) == 1:
            # A single-clip generation always runs the maximum length; short
            # clips are reserved for transition chunks in stories.
            scenes = [Scene(prompt=scenes[0].prompt, seconds=max_seconds)]
        return title, scenes

    def _validate_scenes(
        self, raw_scenes: list, chunk_cap: int, min_seconds: float, max_seconds: float
    ) -> list[Scene]:
        """Enforce every constraint the LLM was asked for; trust nothing."""
        scenes: list[Scene] = []
        for raw in raw_scenes[:chunk_cap]:
            if not isinstance(raw, dict):
                continue
            prompt = _sanitize(str(raw.get("prompt", "")))
            if not prompt:
                continue
            try:
                seconds = float(raw.get("seconds", 8.0))
            except (TypeError, ValueError):
                seconds = 8.0
            scenes.append(Scene(prompt=prompt, seconds=_clamp(seconds, min_seconds, max_seconds)))
        return scenes


def _sanitize(prompt: str) -> str:
    """Collapse whitespace and fit under fast-h3's prompt cap, ending clean.

    LLMs overshoot the character target they are given, and a blind cut at
    the cap ends the prompt mid-word — worse for the model than losing the
    final sentence. Over-long prompts are therefore cut at the last sentence
    boundary that fits; the mid-word cut remains
    only as the last resort for a prompt written as one giant sentence.
    """
    collapsed = " ".join(prompt.split())
    if len(collapsed) <= MAX_PROMPT_CHARS:
        return collapsed.strip()
    head = collapsed[:MAX_PROMPT_CHARS]
    boundary = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if boundary > MAX_PROMPT_CHARS // 2:
        return head[: boundary + 1].strip()
    return head.strip()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
