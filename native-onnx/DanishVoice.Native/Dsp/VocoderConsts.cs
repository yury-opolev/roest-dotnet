using System.Text.Json;
using System.Text.Json.Serialization;

namespace DanishVoice.Native.Dsp;

/// <summary>Constants the vocoder DSP needs, loaded from refs/vocoder_consts.json.</summary>
internal sealed class VocoderConsts
{
    [JsonPropertyName("n_fft")] public int NFft { get; set; }
    [JsonPropertyName("hop_len")] public int HopLen { get; set; }
    [JsonPropertyName("sampling_rate")] public int SamplingRate { get; set; }
    [JsonPropertyName("harmonic_num")] public int HarmonicNum { get; set; }
    [JsonPropertyName("sine_amp")] public float SineAmp { get; set; }
    [JsonPropertyName("noise_std")] public float NoiseStd { get; set; }
    [JsonPropertyName("voiced_threshold")] public float VoicedThreshold { get; set; }
    [JsonPropertyName("f0_upsample_factor")] public int F0UpsampleFactor { get; set; }
    [JsonPropertyName("audio_limit")] public float AudioLimit { get; set; }
    [JsonPropertyName("lrelu_slope")] public float LreluSlope { get; set; }
    [JsonPropertyName("l_linear_weight")] public float[][] LLinearWeight { get; set; } = [];
    [JsonPropertyName("l_linear_bias")] public float[] LLinearBias { get; set; } = [];

    public static VocoderConsts Load(string refsDir)
    {
        var json = File.ReadAllText(Path.Combine(refsDir, "vocoder_consts.json"));
        return JsonSerializer.Deserialize<VocoderConsts>(json)
               ?? throw new InvalidOperationException("could not parse vocoder_consts.json");
    }
}
