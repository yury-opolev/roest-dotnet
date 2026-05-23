using System.Text.Json;
using DanishVoice.Native.Flow;
using DanishVoice.Native.T3;
using DanishVoice.Native.Tokenizer;

namespace DanishVoice.Native;

/// <summary>
/// Full native C# Danish TTS: text -> punc_norm -> tokenizer -> T3 (greedy) ->
/// flow -> vocoder -> waveform. Per-voice conditioning is loaded from the dumped
/// tensors (Mic/Nic). No Python at inference time.
/// </summary>
internal sealed class SynthPipeline : IDisposable
{
    private readonly string refsDir;
    private readonly MtlTokenizer tokenizer;
    private readonly T3Model t3;
    private readonly FlowModel flow;
    private readonly Vocoder vocoder;
    private readonly int startTextToken;
    private readonly int stopTextToken;
    private readonly int stopSpeechToken;

    public SynthPipeline(string onnxDir, string refsDir)
    {
        this.refsDir = refsDir;
        foreach (var s in new[] { "shapes.json", "shapes_t3.json", "shapes_t3loop.json",
                                  "shapes_flow.json", "shapes_flow2.json", "shapes_cond.json" })
        {
            TensorIo.AddShapes(refsDir, s);
        }

        this.tokenizer = new MtlTokenizer(Path.Combine(refsDir, "grapheme_tokenizer.json"));
        this.t3 = new T3Model(onnxDir, refsDir);
        this.flow = new FlowModel(onnxDir, refsDir);
        this.vocoder = new Vocoder(onnxDir, refsDir);

        using var sp = JsonDocument.Parse(File.ReadAllText(Path.Combine(refsDir, "text_specials.json")));
        this.startTextToken = sp.RootElement.GetProperty("start_text_token").GetInt32();
        this.stopTextToken = sp.RootElement.GetProperty("stop_text_token").GetInt32();
        this.stopSpeechToken = sp.RootElement.GetProperty("stop_speech_token").GetInt32();
    }

    public float[] Synth(string text, string voice, int maxNewTokens = 600, int seed = 1234)
    {
        // 1. text prep
        var norm = PuncNorm.Apply(text);
        var ids = this.tokenizer.Encode(norm, "da");
        var textTokens = new int[ids.Length + 2];
        textTokens[0] = this.startTextToken;
        Array.Copy(ids, 0, textTokens, 1, ids.Length);
        textTokens[^1] = this.stopTextToken;

        // 2. per-voice conditioning
        var condEmb = TensorIo.Load(this.refsDir, $"cond_{voice}_t3_cond_emb");
        var lenCond = TensorIo.Shape($"cond_{voice}_t3_cond_emb")[1];
        var promptFeat = TensorIo.Load(this.refsDir, $"cond_{voice}_prompt_feat");
        var xvector = TensorIo.Load(this.refsDir, $"cond_{voice}_xvector");
        using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(this.refsDir, $"cond_{voice}_meta.json")));
        var melLen1 = meta.RootElement.GetProperty("mel_len1").GetInt32();
        using var ptDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(this.refsDir, $"cond_{voice}_prompt_token.json")));
        var promptToken = ptDoc.RootElement[0].EnumerateArray().Select(e => e.GetInt32()).ToArray();

        // 3. T3 greedy decode
        Console.WriteLine($"  T3 decoding (greedy, CPU)...");
        var speech = this.t3.Generate(condEmb, lenCond, textTokens, maxNewTokens);
        // drop trailing stop-speech token
        var speechList = speech.ToList();
        while (speechList.Count > 0 && speechList[^1] == this.stopSpeechToken)
        {
            speechList.RemoveAt(speechList.Count - 1);
        }
        Console.WriteLine($"  T3 produced {speechList.Count} speech tokens");

        // 4. flow: token = prompt_token ++ speech
        var tokenConcat = new int[promptToken.Length + speechList.Count];
        Array.Copy(promptToken, tokenConcat, promptToken.Length);
        for (var i = 0; i < speechList.Count; i++)
        {
            tokenConcat[promptToken.Length + i] = speechList[i];
        }

        var t2 = 2 * tokenConcat.Length;
        var z = GaussianNoise(80 * t2, seed);
        var flowTrace = this.flow.Run(tokenConcat, promptFeat, melLen1, xvector, z);

        // 5. vocoder
        var outFrames = flowTrace.T2 - melLen1;
        var voc = this.vocoder.Run(flowTrace.MelOut, 80, outFrames);
        return voc.Wav;
    }

    private static float[] GaussianNoise(int n, int seed)
    {
        var rng = new Random(seed);
        var z = new float[n];
        for (var i = 0; i < n; i++)
        {
            // Box-Muller
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();
            z[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return z;
    }

    public void Dispose()
    {
        this.t3.Dispose();
        this.flow.Dispose();
        this.vocoder.Dispose();
    }
}
