namespace DanishVoice.Native;

/// <summary>ONNX Runtime execution provider for the native TTS pipeline.</summary>
public enum ExecutionProvider
{
    /// <summary>CPU (default). Correct everywhere; ~10x slower than real time.</summary>
    Cpu,

    /// <summary>CUDA GPU. Requires CUDA 12.x + cuDNN 9 runtime libraries on the host.</summary>
    Cuda,
}

/// <summary>
/// Public entry point for the native, in-process Danish text-to-speech engine
/// (CoRal Røst-v3 via ONNX Runtime — no Python at inference time).
///
/// Point it at a folder containing <c>onnx_models/</c> and <c>refs/</c> (the
/// release runtime bundle), then call <see cref="Synthesize"/>.
/// </summary>
public sealed class DanishVoiceTts : IDisposable
{
    /// <summary>Output sample rate (Hz). Mono.</summary>
    public const int SampleRate = 24000;

    private readonly SynthPipeline pipeline;

    /// <summary>The execution provider actually in use (may differ from the request on CUDA fallback).</summary>
    public ExecutionProvider ActiveProvider { get; }

    /// <param name="modelRoot">Folder containing <c>onnx_models/</c> and <c>refs/</c>.</param>
    /// <param name="provider">Preferred execution provider; falls back to CPU if CUDA init fails.</param>
    public DanishVoiceTts(string modelRoot, ExecutionProvider provider = ExecutionProvider.Cpu)
    {
        var onnxDir = Path.Combine(modelRoot, "onnx_models");
        var refsDir = Path.Combine(modelRoot, "refs");

        if (provider == ExecutionProvider.Cuda)
        {
            try
            {
                this.pipeline = new SynthPipeline(onnxDir, refsDir, useCuda: true);
                this.ActiveProvider = ExecutionProvider.Cuda;
                return;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[DanishVoiceTts] CUDA EP unavailable, falling back to CPU: {ex.Message}");
            }
        }

        this.pipeline = new SynthPipeline(onnxDir, refsDir, useCuda: false);
        this.ActiveProvider = ExecutionProvider.Cpu;
    }

    /// <summary>
    /// Synthesize Danish <paramref name="text"/> in the given <paramref name="voice"/>
    /// (<c>"mic"</c> = female, <c>"nic"</c> = male). Returns 24 kHz mono samples in [-1, 1].
    /// </summary>
    public float[] Synthesize(string text, string voice = "mic", int maxNewTokens = 600, int seed = 1234)
    {
        return this.pipeline.Synth(text, voice, maxNewTokens, seed);
    }

    /// <summary>Synthesize and write a 16-bit PCM WAV file.</summary>
    public void SynthesizeToWav(string text, string voice, string wavPath, int maxNewTokens = 600, int seed = 1234)
    {
        var samples = this.Synthesize(text, voice, maxNewTokens, seed);
        WavWriter.Write(wavPath, samples, SampleRate);
    }

    public void Dispose()
    {
        this.pipeline.Dispose();
    }
}
