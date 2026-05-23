using DanishVoice.Native.Dsp;

namespace DanishVoice.Native;

/// <summary>
/// Native C# HiFT vocoder: ONNX f0_predictor + C# NSF/STFT + ONNX conv stack +
/// C# iSTFT. Mirrors HiFTGenerator.inference (deterministic NSF variant).
/// </summary>
internal sealed class Vocoder : IDisposable
{
    private readonly OnnxModel f0Predictor;
    private readonly OnnxModel convStack;
    private readonly VocoderConsts consts;
    private readonly NsfSource nsf;
    private readonly ShortTimeFourier stft;

    public Vocoder(string onnxDir, string refsDir, bool useCuda = false)
    {
        this.f0Predictor = new OnnxModel(Path.Combine(onnxDir, "voc_f0_predictor.onnx"), useCuda);
        this.convStack = new OnnxModel(Path.Combine(onnxDir, "voc_conv_stack.onnx"), useCuda);
        this.consts = VocoderConsts.Load(refsDir);
        this.nsf = new NsfSource(this.consts);
        this.stft = new ShortTimeFourier(
            this.consts.NFft, this.consts.HopLen,
            ShortTimeFourier.HannPeriodic(this.consts.NFft));
    }

    public VocoderConsts Consts => this.consts;

    /// <summary>Intermediate results, exposed for stage-by-stage parity checks.</summary>
    public sealed class Trace
    {
        public float[] F0 = [];
        public float[] Source = [];
        public float[] SStft = [];
        public float[] Wav = [];
    }

    /// <summary>speechFeat: flat (1, 80, T) row-major. Returns waveform + trace.</summary>
    public Trace Run(float[] speechFeat, int melChannels, int frames)
    {
        var trace = new Trace();

        // Graph A: f0_predictor
        var f0Out = this.f0Predictor.Run(new Dictionary<string, (float[], int[])>
        {
            ["speech_feat"] = (speechFeat, [1, melChannels, frames]),
        });
        var f0 = f0Out["f0"].Data; // (1, T) -> length T
        trace.F0 = f0;

        // C# NSF source
        var source = this.nsf.Generate(f0);
        trace.Source = source;

        // C# forward STFT -> s_stft = concat(real, imag) along channel
        var (real, imag, sFrames) = this.stft.Forward(source);
        var nBins = this.consts.NFft / 2 + 1;
        var sStft = new float[2 * nBins * sFrames];
        Array.Copy(real, 0, sStft, 0, real.Length);
        Array.Copy(imag, 0, sStft, real.Length, imag.Length);
        trace.SStft = sStft;

        // Graph B: conv stack -> magnitude, phase
        var convOut = this.convStack.Run(new Dictionary<string, (float[], int[])>
        {
            ["speech_feat"] = (speechFeat, [1, melChannels, frames]),
            ["s_stft"] = (sStft, [1, 2 * nBins, sFrames]),
        });
        var magnitude = convOut["magnitude"].Data;
        var phase = convOut["phase"].Data;
        var mFrames = convOut["magnitude"].Dims[2];

        // C# iSTFT
        var wav = this.stft.Inverse(magnitude, phase, mFrames);
        for (var i = 0; i < wav.Length; i++)
        {
            wav[i] = Math.Clamp(wav[i], -this.consts.AudioLimit, this.consts.AudioLimit);
        }
        trace.Wav = wav;
        return trace;
    }

    public void Dispose()
    {
        this.f0Predictor.Dispose();
        this.convStack.Dispose();
    }
}
