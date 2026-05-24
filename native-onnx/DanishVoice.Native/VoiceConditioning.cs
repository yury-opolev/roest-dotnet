namespace DanishVoice.Native;

/// <summary>
/// Per-voice conditioning tensors loaded once from the refs bundle and reused
/// across sentences. Independent of the text being synthesized.
/// </summary>
internal sealed record VoiceConditioning(
    float[] CondEmb,
    int LenCond,
    float[] PromptFeat,
    float[] Xvector,
    int MelLen1,
    int[] PromptToken);
