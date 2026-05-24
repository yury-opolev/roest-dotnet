using System.Runtime.CompilerServices;

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
    private readonly SemaphoreSlim gate = new(1, 1);

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
        // Serialize against the streaming path: the ONNX sessions are not
        // concurrency-safe, so all synthesis goes through the same gate.
        this.gate.Wait();
        try
        {
            return this.pipeline.Synth(text, voice, maxNewTokens, seed);
        }
        finally
        {
            this.gate.Release();
        }
    }

    /// <summary>Synthesize and write a 16-bit PCM WAV file.</summary>
    public void SynthesizeToWav(string text, string voice, string wavPath, int maxNewTokens = 600, int seed = 1234)
    {
        // Gated transitively via Synthesize — do not acquire the gate here
        // (SemaphoreSlim is non-reentrant; a second acquire would deadlock).
        var samples = this.Synthesize(text, voice, maxNewTokens, seed);
        WavWriter.Write(wavPath, samples, SampleRate);
    }

    /// <summary>
    /// Stream synthesis sentence by sentence. Splits <paramref name="text"/> into
    /// sentences and yields one 16-bit little-endian PCM chunk (24 kHz mono) per
    /// sentence as it is ready — first audio after the first sentence. The next
    /// sentence synthesizes while the caller consumes the current chunk.
    /// </summary>
    public async IAsyncEnumerable<byte[]> SynthesizeStreamingAsync(
        string text, string voice = "mic", int maxNewTokens = 600, int seed = 1234,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var sentences = SentenceChunker.Split(text);

        await this.gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await foreach (var pcm in PcmSentenceStream.StreamAsync(
                sentences,
                (sentence, _) => this.pipeline.Synth(sentence, voice, maxNewTokens, seed),
                cancellationToken: cancellationToken).ConfigureAwait(false))
            {
                yield return pcm;
            }
        }
        finally
        {
            this.gate.Release();
        }
    }

    /// <summary>
    /// Synthesize the full <paramref name="text"/> and return one concatenated
    /// 16-bit PCM (24 kHz mono) buffer. Convenience wrapper over the stream.
    /// </summary>
    public async Task<byte[]> SynthesizeAsync(
        string text, string voice = "mic", int maxNewTokens = 600, int seed = 1234,
        CancellationToken cancellationToken = default)
    {
        using var ms = new MemoryStream();
        await foreach (var pcm in this.SynthesizeStreamingAsync(text, voice, maxNewTokens, seed, cancellationToken).ConfigureAwait(false))
        {
            ms.Write(pcm);
        }
        return ms.ToArray();
    }

    public void Dispose()
    {
        this.pipeline.Dispose();
        this.gate.Dispose();
    }
}
