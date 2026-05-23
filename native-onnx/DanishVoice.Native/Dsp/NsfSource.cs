namespace DanishVoice.Native.Dsp;

/// <summary>
/// Deterministic Neural Source Filter harmonic-source generator, matching the
/// (noise/phase-zeroed) HiFT SourceModuleHnNSF used for parity.
///
/// Pipeline: f0 (per mel frame) -> nearest upsample -> harmonic sine bank ->
/// linear(9->1) + tanh -> source excitation signal.
/// </summary>
internal sealed class NsfSource
{
    private readonly VocoderConsts c;

    public NsfSource(VocoderConsts consts)
    {
        this.c = consts;
    }

    /// <summary>
    /// f0: one value per mel frame. Returns the source excitation signal of length
    /// f0.Length * f0UpsampleFactor.
    /// </summary>
    public float[] Generate(ReadOnlySpan<float> f0)
    {
        var factor = this.c.F0UpsampleFactor;
        var len = f0.Length * factor;
        var harmonics = this.c.HarmonicNum + 1;
        var sr = this.c.SamplingRate;

        // nearest upsample
        var f0up = new float[len];
        for (var i = 0; i < len; i++)
        {
            f0up[i] = f0[i / factor];
        }

        var w = this.c.LLinearWeight[0]; // [harmonics]
        var b = this.c.LLinearBias[0];

        var s = new float[len];
        // per-harmonic running cumulative phase (cumsum of F_mat along time)
        var cum = new double[harmonics];
        for (var t = 0; t < len; t++)
        {
            var uv = f0up[t] > this.c.VoicedThreshold ? 1.0 : 0.0;
            double acc = b;
            for (var h = 0; h < harmonics; h++)
            {
                var fMat = (double)f0up[t] * (h + 1) / sr;
                cum[h] += fMat;
                var frac = cum[h] - Math.Floor(cum[h]); // cumsum % 1
                var theta = 2.0 * Math.PI * frac;
                var sine = this.c.SineAmp * Math.Sin(theta) * uv; // phase=0, noise=0
                acc += sine * w[h];
            }
            s[t] = (float)Math.Tanh(acc);
        }
        return s;
    }
}
